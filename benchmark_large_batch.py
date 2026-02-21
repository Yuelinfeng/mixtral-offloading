import os
import torch
import time
import argparse
from transformers import AutoTokenizer
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

# Try to import building blocks
try:
    from src.build_model import OffloadConfig, QuantConfig, build_model
    from hqq.core.quantize import BaseQuantizeConfig
except ImportError as e:
    print(f"导入失败: {e}. 请确保你在 mixtral-offloading 项目根目录下运行。")
    exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Large Batch Prefill Benchmark for EBCO vs LRU")
    parser.add_argument('--policy', type=str, default='ebco', choices=['lru', 'ebco'], help='Cache replacement policy')
    parser.add_argument('--main_size', type=int, default=128, help='Global main buffer size (128 = 4 experts per layer on avg)')
    parser.add_argument('--seq_len', type=int, default=4096, help='Length of the massive input sequence (simulates large batch)')
    args = parser.add_argument_group()
    args = parser.parse_args()

    print(f"=== 大型 Batch (长序列预填充) 测试启动 ===")
    print(f"策略: {args.policy.upper()}")
    print(f"全局缓存容量: {args.main_size} 专家")
    print(f"目标序列长度: {args.seq_len} Tokens")
    print("=========================================\n")

    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

    quant_config = QuantConfig(
        ffn_config=BaseQuantizeConfig(nbits=2, group_size=16, quant_zero=True, quant_scale=True),
        attn_config=BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True), # Match benchmark.py
    )
    
    num_hidden_layers = 32
    num_local_experts = 8
    total_experts = num_hidden_layers * num_local_experts # 256
    
    main_size = args.main_size
    offload_size = total_experts - main_size
    offload_per_layer = num_local_experts - (main_size // num_hidden_layers)
    
    offload_config = OffloadConfig(
        main_size=main_size,
        offload_size=offload_size,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"构建模型 (Policy: {args.policy})...")
    model = build_model(device, quant_config, offload_config, state_path, cache_policy=args.policy)
    print("模型构建完成！")

    print(f"\n加载 Wikitext-2 构造大段连续长文本...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x['text'] for x in dataset if len(x['text']) > 50]
        long_text = " ".join(texts[:200]) # Join many paragraphs
    except Exception as e:
        print(f"加载数据集失败 ({e}), 使用基础连词...")
        long_text = "The quick brown fox jumps over the lazy dog. " * 500

    inputs = tokenizer(long_text, return_tensors="pt", max_length=args.seq_len, truncation=True).to(device)
    actual_len = inputs.input_ids.shape[1]
    print(f"构造完毕。实际喂入 Sequence Length = {actual_len} (等效于巨型单批次 Batch Size)")

    print("\n--- 开始执行流水线微批次 (Micro-Batch) 前向传播 ---")
    
    # 核心修正：不能把 4096 个 Token 塞进一个前向传播里！
    # 因为 4096 个多样化 Token 会在每一层瞬间激活所有 8 个专家 (Active Set = 8/8)
    # 这导致缓存算法 (LRU/EBCO) 彻底失效，因为所有专家全是 "刚需"，只能硬替换 50%。
    # 模拟真实的高并发推断系统 (如 vLLM)，我们会将输入切分为 Chunk (例如每 256 个 Token 一批)
    # 连续灌入，这样才能考察出缓存策略在时间序列上的“留存能力”。
    
    chunk_size = 256
    num_chunks = actual_len // chunk_size
    print(f"切分策略: 将序列划分为 {num_chunks} 个微序列 (Micro-Batches), 每个长度 {chunk_size}")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_chunks):
            chunk_inputs = {
                "input_ids": inputs.input_ids[:, i*chunk_size:(i+1)*chunk_size],
                "attention_mask": inputs.attention_mask[:, i*chunk_size:(i+1)*chunk_size] if 'attention_mask' in inputs else None,
            }
            # 清理无用键
            chunk_inputs = {k: v for k, v in chunk_inputs.items() if v is not None}
            _ = model(**chunk_inputs)
            print(f"  > 完成 Chunk {i+1}/{num_chunks}")
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n大段前向传播完成！")
    print(f"处理海量 Token: {actual_len} 个")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"相当于吞吐率: {actual_len / total_time:.2f} Tokens/s")
    
    # 获取缓存信息
    hits = 0
    misses = 0
    try:
        cache = model.model.layers[0].mlp.experts
        
        for group_info in cache.group_infos.values():
            hits += group_info.hits
            misses += group_info.misses
            
        print("\n=== 缓存战报 ===")
        print(f"总命中 (Hits): {hits}")
        print(f"总未命中 (Misses): {misses}")
        if hits + misses > 0:
            print(f"缓存命中率: {hits / (hits + misses) * 100:.2f}%")
            
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
            t_max_rate = (theoretical_hits / total_requested * 100) if total_requested > 0 else 0.0
            print(f"单层平均 Active Set (活跃专家数): {avg_active_size:.2f} / 8")
            print(f"理论极限命中率 (若神级算法): {t_max_rate:.2f}%")
            
        print("\n=== 耗时剖析 ===")
        print(f"CPU 图更新: {total_t_graph_update:.4f}s")
        print(f"CPU 驱逐搜索: {total_t_evict_search:.4f}s")
        t_swap_setup = getattr(cache, 'timers', {}).get('t_swap_setup', 0.0)
        t_routing_cpu = getattr(cache, 'timers', {}).get('t_routing_cpu', 0.0)
        t_pcie_wait = getattr(cache, 'timers', {}).get('t_pcie_wait', 0.0)
        t_gpu_compute = getattr(cache, 'timers', {}).get('t_gpu_compute', 0.0)
        
        print(f"纯 PCIe 物理阻塞等候: {t_pcie_wait:.4f}s")
        print(f"纯 GPU 计算: {t_gpu_compute:.4f}s")
        print("===================")
            
    except Exception as e:
        print(f"获取缓存统计信息失败: {e}")

if __name__ == "__main__":
    main()
