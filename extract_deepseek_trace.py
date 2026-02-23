import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import tqdm
import os
import transformers.utils.import_utils

# Monkey-patch for older transformers versions
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

def extract_trace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Base")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Loading tokenizer {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, args.dataset_name, split="train")
    
    # Pack dataset into a single string for continuous sampling
    text = "\n\n".join(dataset["text"][:1000])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    print(f"Loading model {args.model_id} to {args.device} (This might take a while if downloading)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        trust_remote_code=True, 
        device_map=args.device,
        torch_dtype=torch.bfloat16 # Use bfloat16 to save RAM (16B model = ~32GB RAM)
    )
    
    # Automatically determine routing top-k and expert count from model config if possible
    # DeepSeekV2 config uses n_routed_experts and num_experts_per_tok
    num_routed_experts = getattr(model.config, 'n_routed_experts', 64)
    top_k = getattr(model.config, 'num_experts_per_tok', 6)
    num_moe_layers = getattr(model.config, 'num_hidden_layers', 27) # Usually first few layers are dense, but we check len(router_logits)
    
    print(f"Model properties: Expected Routed Experts={num_routed_experts}, Top-K={top_k}")
    
    # Will be assigned once we know the exact number of MoE layers from the forward pass
    global_transition_matrix = None 
    
    print(f"Extracting traces over {args.num_samples} sequences of length {args.seq_len}...")
    
    # Use forward hooks to capture router logits regardless of model architecture support
    captured_logits = []
    
    def hook_fn(module, input, output):
        # Print progress every few layers to show it's not stuck
        if len(captured_logits) % 5 == 0:
            print(f"    -> [Progress] Captured logits for layer {len(captured_logits)}")
            
        # output is usually the router logits directly or a tuple containing it.
        # Deepseek-V2 usually returns (router_logits) or similar from the gate.
        if isinstance(output, tuple):
            captured_logits.append(output[0].detach().cpu())
        else:
            captured_logits.append(output.detach().cpu())

    # Register hooks on all MLP gates
    hooks = []
    for name, module in model.named_modules():
        # The gate is usually named 'gate' in MoE layers. 
        # MUST use endswith to avoid matching "mlp.gate_proj" which is the dense FFN layer!
        if name.endswith("mlp.gate"):
            hooks.append(module.register_forward_hook(hook_fn))
            
    if not hooks:
        print("Warning: Could not find any modules named 'mlp.gate'. Looking for 'gate'...")
        for name, module in model.named_modules():
            if name.endswith("gate"):
                hooks.append(module.register_forward_hook(hook_fn))
                
    print(f"Registered {len(hooks)} forward hooks on MoE gates.")

    with torch.no_grad():
        for i in tqdm.tqdm(range(args.num_samples)):
            start_idx = i * args.seq_len
            end_idx = start_idx + args.seq_len
            input_ids = tokens[start_idx:end_idx].unsqueeze(0).to(args.device)
            
            captured_logits.clear() # Clear before each forward pass
            
            # Forward pass (disable KV cache to bypass `DynamicCache` compatibility bugs in old transformers)
            model(input_ids, use_cache=False)
            
            if not captured_logits:
                raise ValueError("Hooks did not capture any router logits.")
            
            # DeepSeek V2 gate outputs typically have shape (batch_size * seq_len, num_experts)
            router_logits = captured_logits
            
            actual_moe_layers = len(router_logits)
            
            if global_transition_matrix is None:
                # Actual number of experts from the logits tensor
                actual_num_experts = router_logits[0].shape[-1]
                print(f"\nDiscovered {actual_moe_layers} MoE layers with {actual_num_experts} experts per layer from logits.")
                global_transition_matrix = torch.zeros((actual_moe_layers, actual_num_experts, actual_num_experts), dtype=torch.float32)

            # For each MoE layer, extract the top-k selections
            for l_idx, layer_logits in enumerate(router_logits):
                # layer_logits shape: (batch_size * seq_len, num_experts)
                layer_logits = layer_logits.view(args.seq_len, -1)
                
                # FIX A: Extract strictly Top-1 semantic routing to avoid rank-1 marginal homogenization
                # caused by Top-K cross Cartesian products over adjacent tokens.
                _, selected_experts = torch.topk(layer_logits, k=1, dim=-1)
                
                # Update transition matrix using strictly Top-1 to Top-1 sequence jumps
                for t in range(args.seq_len - 1):
                    u = selected_experts[t, 0]
                    v = selected_experts[t+1, 0]
                    global_transition_matrix[l_idx, u.item(), v.item()] += 1.0
                            
    out_path = "deepseek_markov_transition_matrix.pt"
    torch.save(global_transition_matrix, out_path)
    print(f"\n[Success] Extraction complete! Transition matrix saved to {out_path}.")
    print(f"Run `python analyze_markov_graph.py` but remember to modify it to read this new '{out_path}' file and handle the 64x64 output!")

if __name__ == "__main__":
    extract_trace()
