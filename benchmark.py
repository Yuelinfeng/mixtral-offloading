
import sys
import time
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
    print(f"正在配置模型运行环境...")
    print(f"设备: {device}")
    print(f"Offload 配置: Main={offload_config.main_size}, Offload={offload_config.offload_size}")
    
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
        model = build_model(device, quant_config, offload_config, state_path)
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
    except Exception as e:
        print(f"无法获取缓存统计信息: {e}")

if __name__ == "__main__":
    main()
