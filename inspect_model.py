
import torch
from transformers import AutoConfig, AutoModelForCausalLM

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

print(f"Loading config for {model_name}...")
try:
    config = AutoConfig.from_pretrained(model_name)
    # 不加载权重，只加载结构，通过 meta device 或不加载 state_dict
    # 但 AutoModelForCausalLM.from_config 是最快的
    print("Initializing model structure (no weights)...")
    model = AutoModelForCausalLM.from_config(config)
    
    print("\n" + "="*50)
    print("MixtralDecoderLayer Structure Analysis")
    print("="*50)
    
    first_layer = model.model.layers[0]
    print(f"Layer Class: {type(first_layer).__name__}")
    print("\nAttributes:")
    
    # 打印所有属性，过滤掉私有属性
    for attr in dir(first_layer):
        if not attr.startswith("_"):
            val = getattr(first_layer, attr)
            print(f"- {attr}: {type(val).__name__}")
            
    print("="*50)
    
    # 检查是否有 mo 相关的属性
    possible_names = ["block_sparse_moe", "experts", "moe", "sparse_moe", "mlp"]
    print("\nChecking for MoE related attributes:")
    for name in possible_names:
        if hasattr(first_layer, name):
            print(f"[FOUND] {name}")
        else:
            print(f"[MISSING] {name}")

except Exception as e:
    print(f"Error: {e}")
