import torch
import argparse
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import tqdm
import os
import transformers.utils.import_utils

# Monkey-patch just in case
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

def extract_trace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/switch-base-128")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Loading tokenizer {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, args.dataset_name, split="train")
    
    text = "\n\n".join(dataset["text"][:1000])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    print(f"Loading model {args.model_id} to {args.device}...")
    model = AutoModel.from_pretrained(
        args.model_id, 
        device_map=args.device,
        torch_dtype=torch.float32 
    )
    
    # We'll extract only the ENCODER routing to keep it simple and consistent.
    print(f"Extracting traces over {args.num_samples} sequences of length {args.seq_len}...")
    
    captured_logits = []
    
    def hook_fn(module, input, output):
        if len(captured_logits) % 3 == 0:
            print(f"    -> [Progress] Captured logits for router {len(captured_logits)}")
            
        # The true logits come from the linear classifier layer natively
        if isinstance(output, tuple):
            captured_logits.append(output[0].detach().cpu())
        else:
            captured_logits.append(output.detach().cpu())

    hooks = []
    # Hook the actual Linear layer that outputs the 128 dimensions directly
    for name, module in model.encoder.named_modules():
        if name.endswith("router.classifier"):
            hooks.append(module.register_forward_hook(hook_fn))
            
    if not hooks:
        print("Warning: Could not find routers in encoder. Registering blindly...")
        for name, module in model.named_modules():
            if name.endswith("router"):
                hooks.append(module.register_forward_hook(hook_fn))
                
    print(f"Registered {len(hooks)} forward hooks on MoE routers in the encoder.")

    global_transition_matrix = None

    with torch.no_grad():
        for i in tqdm.tqdm(range(args.num_samples)):
            start_idx = i * args.seq_len
            end_idx = start_idx + args.seq_len
            input_ids = tokens[start_idx:end_idx].unsqueeze(0).to(args.device)
            
            captured_logits.clear()
            
            # Forward pass ONLY through Encoder to capture sequential routing
            # KV cache disabled natively in encoder usually
            model.encoder(input_ids=input_ids)
            
            if not captured_logits:
                raise ValueError("Hooks did not capture any router logits.")
            
            router_logits = captured_logits
            actual_moe_layers = len(router_logits)
            
            if global_transition_matrix is None:
                # Actual number of experts from the logits tensor (128 for Switch Base)
                actual_num_experts = router_logits[0].shape[-1]
                print(f"\nDiscovered {actual_moe_layers} encoder MoE layers with {actual_num_experts} experts per layer.")
                global_transition_matrix = torch.zeros((actual_moe_layers, actual_num_experts, actual_num_experts), dtype=torch.float32)

            # Switch Transformer is distinctly Top-1 Routing
            top_k = 1
            
            for l_idx, layer_logits in enumerate(router_logits):
                layer_logits = layer_logits.view(args.seq_len, -1)
                
                _, selected_experts = torch.topk(layer_logits, k=top_k, dim=-1)
                
                for t in range(args.seq_len - 1):
                    u = selected_experts[t, 0]
                    v = selected_experts[t+1, 0]
                    global_transition_matrix[l_idx, u.item(), v.item()] += 1.0
                            
    out_path = "switch_markov_transition_matrix.pt"
    torch.save(global_transition_matrix, out_path)
    print(f"\n[Success] Extraction complete! Transition matrix saved to {out_path}.")
    print(f"Run `python analyze_markov_graph.py --matrix {out_path}` to analyze it!")

if __name__ == "__main__":
    extract_trace()
