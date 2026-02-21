# MoE 大模型卸载策略：协同缓存实验失败分析与流水线架构重构展望

本报告综合呈现了关于 Mixtral-8x7B (MoE) 离线推理引擎中“空间协同缓存策略（EBCO）”的算法思想、物理实验验证结果，以及在极度受限显存下的瓶颈反思与架构演进方向。

---

## 一、 算法思想与数学形式化 (Algorithm Formulation)

在大规模 Batch 推理中，传统的 LRU（最近最少使用）驱逐算法仅考虑时序局部性，忽略了模型参数自身的语义关联。我们的创新点源于一个核心假设：**同层专家间存在强烈的“空间共现性 (Spatial Co-occurrence)”**，即部分专家经常在同一 Batch 或同一 Token 的 Top-K 路由中被联合激活。

### 1.1 共现图构建 (Co-occurrence Graph Construction)
我们为每一层网络构建了一个无向加权图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中顶点集合 $\mathcal{V}$ 代表该层的所有专家。
边上的权重矩阵（相似度矩阵）定义为 $W \in \mathbb{R}^{N \times N}$（$N=8$ 为专家总数）。

对于输入的 Prompt/Batch，Router 网络输出该层令牌选择专家的概率矩阵 $P \in \mathbb{R}^{B \times N}$，其中 $B$ 为 Batch 内的 token 数量。
我们的算法尝试了两种构建 $W$ 的方式：

*   **Soft Cosine 相似度**：利用全量 Softmax 概率的点乘。
    $$C = \frac{1}{B} P^T P$$
*   **Hard 二值共现（修正后）**：仅使用严格路由的 Top-2 索引，消除概率平滑噪声。
    令 $\tilde{P} \in \{0,1\}^{B \times N}$ 为通过 `topk(P)` 截断后的 One-Hot 矩阵。
    $$C = \frac{1}{B} \tilde{P}^T \tilde{P}$$

矩阵 $W$ 通过指数移动平均 (EMA) 在不同批次 $t$ 间进行在线更新：
$$W_{t} = \alpha C_t + (1 - \alpha) W_{t-1}$$

### 1.2 协同置换策略 (Collaborative Eviction)
当发生 Cache Miss 且显存容量不足时，我们需要在当前显存内的专家集合 $\mathcal{M}$ (Main Memory) 中挑选一个牺牲者 (Victim) 踢入 CPU。
给定当前帧所需要的专家集合 $\mathcal{A}$ (Active Set)，我们计算 $\mathcal{M}$ 中每个候选专家 $e_i$ 的**协同留存得分 (Collaborative Score)**：
$$Score(e_i) = \sum_{e_a \in \mathcal{A}} W(e_i, e_a)$$
算法选择得分**最低**的专家进行驱逐，以期保留与当前 Active Set 紧密绑定的“兄弟”专家。

---

## 二、 实验结果与假设证伪 (Empirical Results & Falsification)

我们在单卡 RTX 4090 (限制配置：`Main=64`，即平均每层仅分到 2 个常驻专家槽位) 的环境下运行了真实数据集 (Wikitext-2) 的推理基准测试。

### 2.1 性能极化与核心数据
| 策略             | 相似度获取方式     | 缓存命中率          | 吞吐量 (TPS) | CPU 算法开销 |
| :------------- | :---------- | :------------- | :-------- | :------- |
| **LRU 基线**     | 纯时序 (无图)    | **36.86%**     | **2.84**  | 极低       |
| **EBCO Graph** | Soft Cosine | 35.83% (-1.0%) | 2.40      | 中等       |
| **EBCO Graph** | Hard Binary | 34.67% (-2.2%) | 2.39      | 中等       |

### 2.2 假设证伪分析
实验数据**无可辩驳地证伪**了我们的核心创新点（同层空间协同）。
数据表明，即便是消除了所有数学噪声的绝对二值共现统计（Hard Binary），命中率不仅没有超越无脑的 LRU，反而跌到底部。

*   **特征正交性 (Orthogonality)**：相似度矩阵 $W$ 非对角线元素基本在 $10^{-3}$ 到 $10^{-2}$ 之间。这证明了 Mixtral 在设计与训练时，其路由机制极度鼓励**专家的解耦与特异化**。同层内不同专家负责绝对正交的语义空间，不存在互相绑定的“帮派”现象。
*   **时序远大于空间**：LRU 的胜出证明，相比于“谁和谁一起出现”（空间维），路由器的行为更倾向于“刚才那个 Token 去了哪，紧跟着的下一个 Token 也大概率去哪”（时间局部性）。

**结论**：在 Mixtral 此类 MoE 模型中，同层专家间的空间共现性属于**弱信噪比/无效信号**。基于此构建预测缓存，属于方向性偏差。

---

## 三、 显存物理瓶颈深度剖析 (Bottleneck Diagnosis)

通过 `time.perf_counter()` 的极低延迟埋点，我们获取了一份针对 TPS=2.39 这个惨淡数据的“黑盒尸检报告”：

### 3.1 遥测数据 (Telemetry)
*   总计运行耗时: $62.74$ 秒
*   CPU 图算法计算开销 (Graph Update + Evict Search): $1.02$ 秒 ($1.6\%$)
*   **真正的 PCIe 同步死锁 (Implicit Wait) + 少量 GPU 算力: $61.39$ 秒 ($97.8\%$)**

### 3.2 痛点根因：容量死锁 (Capacity Deadlock)
算法并非变慢的原因（CPU 开销仅占 1%）。真正的痛点是由**显卡物理容量配置**直接导致的疯狂换页颠簸 (Thrashing)。

*   **物理定律**：设定值 `Main=64` 导致分摊到每层的**容量仅为 2**。
*   **负载需求**：Mixtral 是 Top-2 路由。对于每一个输入，**活跃需求刚性为 2**。

**需求量 (2) == 最大缓存容量 (2)**。
这导致缓存的“自由周旋度 (Degrees of Freedom)”为 0。不管采用 LRU 还是神级预测器，一旦当前需要的新专家不在 GPU 内，系统**必须全盘洗牌**，根本留不出任何闲置槽位来“提前贮藏”那些高分预测专家。
在这 $61.39$ 秒的庞大耗时中，怪兽级的 RTX 4090 有将近 60 秒是**空转闲置的**，干等庞大的 GB 级权重通过狭窄的 PCIe 总线传输。

---

## 四、 破局之道：宏流水线架构 (MoE-Lightning 带来的重构思路)

当本地容量刚性受限，纯靠“智能替换算法”已经走到死胡同时，顶会级论文 *MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs* 给出了标准解法：**打破 PCIe 串行，引入 CPU-GPU-I/O 调度流水线 (CGOPipe)**。

### 4.1 传统阻滞噩梦 (Current State)
`发起 PCIe 传输专家A -> (GPU完全死等这几十毫秒) -> 传输完成 -> GPU 开始算专家A`

### 4.2 解决思路：基于宏微批次的异步流水线 (Macro-Micro Batch Pipelining)

借鉴 MoE-Lightning 的核心思想（Pre-attention + Interleaved Paging），在不修改 HuggingFace 原生权重的前提下，我们需要对本项目进行如下彻底重构：

#### A. 第一步：前端预测剥离 (Lookahead Gate / Pre-Routing)
抛弃走一步看一步的做法。当大 Batch Token 进来时，在执行真正的 Transformer 层注意力计算之前，我们强制**用极小的计算量（仅提取 Gate 层）瞬间将此 Batch 全路程的路由结果推演完毕**。
这赐予了我们“预知未来全局视野”的能力，这是发起预取的先决条件。

#### B. 第二步：微批次拆解与流式交错 (Micro-Batching & Stream Overlap)
针对某个极度饥饿的 `Layer i`，假设容量为 2，但全局需要 6 个专家。
我们将计算流暴力切碎 (Chunking)，利用我们在代码中已经搭建的 `torch.cuda.Stream`，实现算力与 I/O 的双龙出海：

1.  **I/O Stream** 发起对于 `Chunk 1 (Expert 1, 2)` 的异步拷贝。
2.  `Chunk 1` 到达，交棒给 **Compute Stream** 开始无情计算。
3.  **【核心重叠】** 在 Compute Stream 满载运算 `Chunk 1` 这几十毫秒窗口内，**I/O Stream** 绝非闲置，而是立刻发起静默后台拉取指令，将 `Chunk 2 (Expert 3, 4)` 强行塞向预留显存。
4.  当 Compute Stream 算完 `Chunk 1` 抬头时，`Chunk 2` 刚好落地，实现**无缝衔接，消融泡沫 (Pipeline Bubble)**。

通过这种“以算掩传 (Compute Hiding Transfer)”的极致重叠，即使在容量极度恶劣的环境下，大 Batch 的吞吐量也将实现数量级跃迁。这将是我们告别虚假相关性实验，迈向大型工业级引擎的下一步宏伟蓝图。

---

## 五、 终极物理叹息：理论公式与 I/O 墙的残酷现实 (The I/O Wall)

最后，我们在模型底层（[custom_layers.py](file:///d:/moe_offloading/mixtral-offloading/src/custom_layers.py)）注入了严苛的硬件流水级探针，以验证第四部分流水线重构的投入产出比。核心目标是剥离出“纯 PCIe 阻塞时间” $\mathcal{T}_{Wait}$ 和“纯 GPU 计算时间” $\mathcal{T}_{Compute}$。

我们定义了**可量化 Overlap 成本收益公式**：
*   **串行基线耗时**：$\mathcal{T}_{Current} = \mathcal{T}_{Wait} + \mathcal{T}_{Compute}$
*   **完美流水线极限**：$\mathcal{T}_{Ideal} = \max(\mathcal{T}_{Wait}, \mathcal{T}_{Compute})$
*   **流水线最大可挽回时间 (Bubble 容量)**：$\Delta\mathcal{T}_{Gain} = \min(\mathcal{T}_{Wait}, \mathcal{T}_{Compute})$

### 5.1 残酷的测定结果
执行 [benchmark.py](file:///d:/moe_offloading/mixtral-offloading/benchmark.py) 后，真实的物理切片数据如下：
*   $\mathcal{T}_{Current} = 10.4961$ 秒
*   **$\mathcal{T}_{Wait} = 10.2588$ 秒**
*   **$\mathcal{T}_{Compute} = 0.2373$ 秒**
*   $\Delta\mathcal{T}_{Gain} = 0.2373$ 秒

### 5.2 结论：为什么单纯的 Pipelining 救不了小 Batch 的命？
数据宣告了一个沉重的系统工程学事实：**PCIe 的物理传输耗时达到了底层算力耗时的 43.2 倍！**
换句话说，就算我们不眠不休写出了这个世界上最完美的异步流水线，让 GPU 算力的这 $0.2373$ 秒 100% 完美掩藏在 PCIe 的后台搬运之中，我们所作的系统级重构，**仅仅能在总耗时中提升可怜的 2% ($10.49s \rightarrow 10.25s$)**。

系统当前并未受困于气泡 (Bubble) 或 CPU 调度，而是被一堵绝对无法被代码撼动的纯物理墙——**PCIe 4.0 总线带宽天花板 (I/O Wall)** 死死封印住了。

### 5.3 真正走向工业级的终局思路
在单节点容量极度受限且遭遇绝对 I/O 墙时，Pipelining（例如 MoE-Lightning 的机制）必须与特定负载配合才能发威。我们的最终突围路线有且仅有以下三个维度：
1. **暴力提升并发 Batch Size (大吞吐推断)**：GPU 计算时间 $\mathcal{T}_{Compute}$ 是随 Batch Size 几何增长的，而传输时间 $\mathcal{T}_{Wait}$（搬运一整个专家权重）在此阶段不随输入大小变化。**只有当线上并发 Batch Size 急剧飙升数十倍，使得 $\mathcal{T}_{Compute}$ 也膨胀到 10 秒级别（与 $\mathcal{T}_{Wait}$ 齐平），流水线重叠机制（Pipelining）才能爆发出其 1+1=1 的真正威力（即吞吐翻倍）。**这也是所有大型高吞吐调度系统（vLLM/MoE-Lightning）能够存活的核心土壤。
2. **更高维度的极端量化 (Bits-and-Bytes)**：将专家从常规的 4-bit 继续无情压缩至 2-bit (即应用我们代码库里的 HQQ 深度压缩)，通过折损部分模型智商，来实质性削薄单专家的物理体积，从公式源头上成倍削减 $\mathcal{T}_{Wait}$。
3. **放弃单机扩展：分布式显存池化**：不再依靠龟速的系统内存 PCIe 通道，而是通过高速互联骨干网（NVLink/InfiniBand），将多张廉价 GPU（如多卡 4090）组成物理显存资源池，通过堆硬件使得 $Capacity > Active\ Set$，将 Miss 降到 0，强行让 $\mathcal{T}_{Wait}$ 归零。
