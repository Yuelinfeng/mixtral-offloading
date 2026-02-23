import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import tqdm
import os

def extract_trace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
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
    
    text = "\n\n".join(dataset["text"][:1000])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    print(f"Loading model {args.model_id} to {args.device}...")
    # This model is ~5.7GB, it should easily fit on disk and in memory
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        trust_remote_code=True, 
        device_map=args.device,
        torch_dtype=torch.float32 # Use standard float for better CPU compat
    )
    
    print(f"Extracting traces over {args.num_samples} sequences of length {args.seq_len}...")
    
    captured_logits = []
    
    def hook_fn(module, input, output):
        if len(captured_logits) % 5 == 0:
            print(f"    -> [Progress] Captured logits for layer {len(captured_logits)}")
            
        # output is usually the router logits directly or a tuple.
        if isinstance(output, tuple):
            captured_logits.append(output[0].detach().cpu())
        else:
            captured_logits.append(output.detach().cpu())

    hooks = []
    for name, module in model.named_modules():
        if "mlp.gate" in name:
            hooks.append(module.register_forward_hook(hook_fn))
            
    if not hooks:
        print("Warning: Could not find any modules named 'mlp.gate'. Looking for 'gate'...")
        for name, module in model.named_modules():
            if name.endswith("gate"):
                hooks.append(module.register_forward_hook(hook_fn))
                
    print(f"Registered {len(hooks)} forward hooks on MoE gates.")

    global_transition_matrix = None

    with torch.no_grad():
        for i in tqdm.tqdm(range(args.num_samples)):
            start_idx = i * args.seq_len
            end_idx = start_idx + args.seq_len
            input_ids = tokens[start_idx:end_idx].unsqueeze(0).to(args.device)
            
            captured_logits.clear()
            
            # KV cache disabled
            model(input_ids, use_cache=False)
            
            if not captured_logits:
                raise ValueError("Hooks did not capture any router logits.")
            
            router_logits = captured_logits
            actual_moe_layers = len(router_logits)
            
            if global_transition_matrix is None:
                # Actual number of experts from the logits tensor (60 for Qwen1.5-MoE routed experts)
                actual_num_experts = router_logits[0].shape[-1]
                print(f"\nDiscovered {actual_moe_layers} MoE layers with {actual_num_experts} experts per layer from logits.")
                global_transition_matrix = torch.zeros((actual_moe_layers, actual_num_experts, actual_num_experts), dtype=torch.float32)

            for l_idx, layer_logits in enumerate(router_logits):
                layer_logits = layer_logits.view(args.seq_len, -1)
                # FIX A: Extract strictly Top-1 semantic routing to avoid rank-1 homogenization
                # caused by Top-K cross Cartesian products.
                _, selected_experts = torch.topk(layer_logits, k=1, dim=-1)
                
                for t in range(args.seq_len - 1):
                    u = selected_experts[t, 0]
                    v = selected_experts[t+1, 0]
                    global_transition_matrix[l_idx, u.item(), v.item()] += 1.0
                            
    out_path = "qwen_markov_transition_matrix.pt"
    torch.save(global_transition_matrix, out_path)
    print(f"\n[Success] Extraction complete! Transition matrix saved to {out_path}.")
    print(f"Run `python analyze_markov_graph.py --matrix {out_path}` to analyze it!")

if __name__ == "__main__":
    extract_trace()
