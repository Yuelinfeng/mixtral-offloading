
import sys
import time
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from src.build_model import build_model, OffloadConfig, QuantConfig
from src.utils import with_default_dtype
from hqq.core.quantize import BaseQuantizeConfig

# ==========================================
# 配置部分 (Configuration)
# ==========================================

# 模型名称
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# 权重路径 (请根据实际情况修改)
state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo" 

# 量化配置 (Quantization Config)
# 注意：这里使用演示中的典型配置
# Attention 层使用 4-bit 量化，Group Size 64
attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)

# FFN 层 (专家层) 使用 2-bit 量化，Group Size 16
ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)

quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

# 获取模型配置以计算总专家数
config = AutoConfig.from_pretrained(model_name)
num_hidden_layers = config.num_hidden_layers
num_local_experts = config.num_local_experts

# 每层卸载数量
offload_per_layer = 6

# 卸载配置 (Offload Config) - EBCO 核心
# 注意：ExpertCache 是全局共享的，所以 main_size 和 offload_size 必须是所有层的总和
offload_config = OffloadConfig(
    main_size=num_hidden_layers * (num_local_experts - offload_per_layer),
    offload_size=num_hidden_layers * offload_per_layer,
    buffer_size=4,     # 缓冲区大小 (适当增加以应对并行)
    offload_per_layer=offload_per_layer
)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="ebco", choices=["lru", "ebco", "embed", "markov"])
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--use_prefetch", action="store_true")
    args = parser.parse_args()

    print(f"正在配置模型运行环境...")
    print(f"设备: {device}")
    print(f"Offload 配置: Main={offload_config.main_size}, Offload={offload_config.offload_size}")
    print(f"缓存策略 (Policy): {args.policy.upper()}")
    if args.policy == "ebco":
        print(f"Lambda 权重: {args.lambda_weight}")
        print(f"启用预取 (Prefetch): {args.use_prefetch}")
    
    # 1. 加载 Tokenizer
    print("加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"加载 Tokenizer 失败，请检查网络或路径: {e}")
        return

    # 2. 构建模型 (Build Model)
    print("构建模型中 (这可能需要一些时间)...")
    try:
        model = build_model(
            device, quant_config, offload_config, state_path, 
            cache_policy=args.policy, 
            lambda_weight=args.lambda_weight, 
            use_prefetch=args.use_prefetch
        )
    except Exception as e:
        print(f"构建模型失败: {e}")
        print("提示: 请确保 'state_path' 指向正确的权重目录。")
        return

    print("模型构建完成！")

    # 3. 准备推理输入 (使用 Wikitext-2 真实数据集)
    print("\n正在加载 Wikitext-2 数据集用于真实测试...")
    try:
        from datasets import load_dataset
        # 使用 wikitext-2-raw-v1 的 test 集，只取前几个段落
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # 过滤掉短句，只保留有意义的段落
        texts = [x['text'] for x in dataset if len(x['text']) > 100][:3] 
        print(f"成功加载 {len(texts)} 个样本进行测试。")
    except ImportError:
        print("缺少 datasets 库，请先安装: pip install datasets")
        print("降级使用固定 Prompt 进行测试。")
        texts = ["Explain the concept of 'Expert-Based Collaborative Offloading' in simple terms."]
    except Exception as e:
        print(f"加载数据集失败: {e}")
        texts = ["Explain the concept of 'Expert-Based Collaborative Offloading' in simple terms."]

    total_new_tokens = 0
    total_time = 0
    
    # 4. 预热 (Warmup)
    print("\n开始预热 (Warmup)...")
    dummy_input = tokenizer("Warmup", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**dummy_input, max_new_tokens=5)
    print("预热完成。")

    # 5. 正式 Benchmark
    print("\n开始正式推理测试...")
    
    # 使用 Streamer 实时显示输出 (可选，这里为了测速可能不显示全部)
    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    start_time_global = time.time()
    
    for i, text in enumerate(texts):
        print(f"\n[Case {i+1}] 输入长度: {len(text)} 字符")
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=50, 
                pad_token_id=tokenizer.eos_token_id,
                # streamer=streamer 
            )
        end_time = time.time()
        
        step_time = end_time - start_time
        new_tokens = output.shape[1] - inputs.input_ids.shape[1]
        
        total_new_tokens += new_tokens
        total_time += step_time
        
        print(f"  -> 生成 {new_tokens} tokens, 耗时 {step_time:.2f}s, 当前速度: {new_tokens/step_time:.2f} TPS")

    avg_tps = total_new_tokens / total_time
    
    print("\n" + "="*30)
    print("真实数据集测试结果 (Real Benchmark Results)")
    print("="*30)
    print(f"测试样本数: {len(texts)}")
    print(f"总生成 Token 数: {total_new_tokens}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均推理速度 (TPS): {avg_tps:.2f} tokens/s")
    print("="*30)
    
    # 尝试访问缓存统计信息 (如果可用)
    hits = 0
    misses = 0
    try:
        cache = model.model.layers[0].mlp.experts
        
        # 统计所有驱逐组
        for group_info in cache.group_infos.values():
            hits += group_info.hits
            misses += group_info.misses
            
        print(f"缓存命中 (Hits): {hits}")
        print(f"缓存未命中 (Misses): {misses}")
        if hits + misses > 0:
            print(f"缓存命中率: {hits / (hits + misses) * 100:.2f}%")
            
        # --- Telemetry Analysis ---
        all_active_sizes = []
        theoretical_hits = 0.0
        total_requested = 0
        total_t_graph_update = 0.0
        total_t_evict_search = 0.0
        
        for group_info in cache.group_infos.values():
            all_active_sizes.extend(group_info.active_set_sizes)
            theoretical_hits += group_info.theoretical_hits
            total_requested += group_info.total_requested
            total_t_graph_update += getattr(group_info, 't_graph_update', 0.0)
            total_t_evict_search += getattr(group_info, 't_evict_search', 0.0)
            
        if all_active_sizes:
            avg_active_size = sum(all_active_sizes) / len(all_active_sizes)
            pct_gt_2 = sum(1 for s in all_active_sizes if s > 2) / len(all_active_sizes) * 100
            theoretical_max_rate = (theoretical_hits / total_requested * 100) if total_requested > 0 else 0.0
            
            # Simple quantiles
            all_active_sizes.sort()
            p50 = all_active_sizes[len(all_active_sizes) // 2]
            p90 = all_active_sizes[int(len(all_active_sizes) * 0.9)]
            
            print("-" * 20)
            print("工作负载特征分析 (Workload Telemetry):")
            print(f"  均值 |Active Set|: {avg_active_size:.2f} (每层每步需求专家数)")
            print(f"  P50 / P90: {p50} / {p90}")
            print(f"  超载比例 (|Active Set| > 2): {pct_gt_2:.2f}%")
            print(f"  理论最大命中率 (容量受限上限): {theoretical_max_rate:.2f}%")
            print("-" * 20)
            print("精细化耗时分析 (Fine-grained Latency Profiling):")
            print(f"  CPU图更新总耗时 (Graph Update): {total_t_graph_update:.4f} 秒")
            print(f"  CPU驱逐搜索总耗时 (Evict Search): {total_t_evict_search:.4f} 秒")
            t_swap_setup = getattr(cache, 'timers', {}).get('t_swap_setup', 0.0)
            t_routing_cpu = getattr(cache, 'timers', {}).get('t_routing_cpu', 0.0)
            t_pcie_wait = getattr(cache, 'timers', {}).get('t_pcie_wait', 0.0)
            t_gpu_compute = getattr(cache, 'timers', {}).get('t_gpu_compute', 0.0)
            print(f"  PCIe装载发起 & 路由开销 (CPU Setup & Route): {t_swap_setup + t_routing_cpu:.4f} 秒")
            print(f"  纯GPU计算耗时 (Pure GPU Compute): {t_gpu_compute:.4f} 秒")
            print(f"  物理PCIe阻塞死等耗时 (Pure PCIe Wait): {t_pcie_wait:.4f} 秒")
            
            # Formula Evaluation
            t_actual = t_pcie_wait + t_gpu_compute
            t_ideal_perfect_overlap = max(t_pcie_wait, t_gpu_compute)
            print("-" * 20)
            print("★ 流水线潜力分析与 Overlap 成本公式 (Pipeline Potential Formula):")
            print("  当前串行执行模式: T_Current = T_Wait + T_Compute = {:.4f} 秒".format(t_actual))
            print("  理论完美掩盖极限: T_Ideal = Max(T_Wait, T_Compute) = {:.4f} 秒".format(t_ideal_perfect_overlap))
            print("  通过实现 Macro-Pipelining 极度压榨出的性能空间 (可挽救时间 Bubble) = {:.4f} 秒".format(t_actual - t_ideal_perfect_overlap))
            print("  (注: 若 T_Wait 远大于 T_Compute，说明存在纯粹的 IO 墙，即使完美流水线，TPS 上限也将由带宽决定！)")
        # ---------------------------
            
        print("-" * 20)
        print("调试信息 (Debug Info): Layer 0 Similarity Matrix")
        first_group = next(iter(cache.group_infos.values()))
        if hasattr(first_group, "similarity_matrix") and first_group.similarity_matrix is not None:
             mat = first_group.similarity_matrix
             print(f"Shape: {mat.shape}")
             print(f"Max value: {mat.max().item():.4f}")
             print(f"Min value: {mat.min().item():.4f}")
             print(f"Mean value: {mat.mean().item():.4f}")
             print(f"Sample (Top-left 4x4):\n{mat[:4, :4]}")
        else:
             print("Similarity Matrix is None!")
             
        # --- [Stage 0] Export Markov Transition Graph ---
        if hasattr(cache, 'transition_matrix'):
             matrix_path = "markov_transition_matrix.pt"
             torch.save(cache.transition_matrix.cpu(), matrix_path)
             print(f"\n[Stage 0] 已成功将 32x8x8 时序转移矩阵导出至: {matrix_path}")
             print("下一步：请运行 `python analyze_markov_graph.py` 来计算图的 SCCs 和谱间隙！")
        # ------------------------------------------------
    except Exception as e:
        print(f"无法获取缓存统计信息: {e}")

if __name__ == "__main__":
    main()
