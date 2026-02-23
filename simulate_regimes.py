import torch
import numpy as np
import json
import scipy.linalg as linalg
import tqdm
import os

def compute_spectral_gap(P_tensor):
    """Computes the average spectral gap (delta) across all layers of the Markov transition matrix."""
    gaps = []
    L = P_tensor.shape[0]
    for l in range(L):
        P = P_tensor[l].numpy()
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_norm = P / row_sums
        
        try:
            eigenvalues = linalg.eigvals(P_norm)
            sorted_evals = np.sort(np.abs(eigenvalues))[::-1]
            if len(sorted_evals) > 1:
                gaps.append(1.0 - sorted_evals[1])
            else:
                gaps.append(0.0)
        except Exception as e:
            pass
    return float(np.mean(gaps)) if gaps else 0.0

def generate_batched_trace(P_tensor, seq_len, batch_size):
    """Generates synthetic expert routing traces based on empirical transition matrix."""
    L, E, _ = P_tensor.shape
    traces = np.zeros((batch_size, seq_len, L), dtype=int)
    
    P_norm_layers = []
    for l in range(L):
        P = P_tensor[l].numpy()
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_norm = P / row_sums
        P_norm_layers.append(P_norm)
        
    for b in range(batch_size):
        curr_states = np.random.randint(0, E, size=L)
        traces[b, 0, :] = curr_states
        
        for t in range(1, seq_len):
            for l in range(L):
                p_dist = P_norm_layers[l][curr_states[l]]
                if p_dist.sum() == 0 or np.isnan(p_dist.sum()):
                    next_state = np.random.randint(0, E)
                else:
                    p_dist = p_dist / p_dist.sum()
                    next_state = np.random.choice(E, p=p_dist)
                traces[b, t, l] = next_state
                curr_states[l] = next_state
                
    return traces

def simulate_cache(traces, capacity, policy="lru", admission_threshold=0.0, P_tensor=None):
    """
    Simulates cache behavior over the given traces.
    Returns metrics suitable for INFORMS evaluation proxy.
    """
    B, S, L = traces.shape
    
    hits = 0
    misses = 0
    evictions = 0
    bypassed = 0
    
    caches = [[] for _ in range(L)]
    
    P_norm_layers = []
    if policy == "markov" and P_tensor is not None:
        for l in range(L):
            P = P_tensor[l].numpy()
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            P_norm_layers.append(P / row_sums)

    for t in range(S):
        for l in range(L):
            reqs = traces[:, t, l]
            unique_reqs = set(reqs)
            
            admitted_reqs = set()
            for r in unique_reqs:
                if policy == "markov":
                    admit = False
                    if not caches[l]:
                        admit = True
                    else:
                        for c in caches[l]:
                            if P_norm_layers[l][c, r] >= admission_threshold:
                                admit = True
                                break
                    
                    if admit:
                        admitted_reqs.add(r)
                    else:
                        bypassed += 1
                        misses += 1
                else:
                    admitted_reqs.add(r)
            
            for r in admitted_reqs:
                if r in caches[l]:
                    hits += 1
                    caches[l].remove(r)
                    caches[l].append(r)
                else:
                    misses += 1
                    caches[l].append(r)
                    if len(caches[l]) > capacity:
                        caches[l].pop(0)
                        evictions += 1

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "bypassed": bypassed,
        "churn_per_step_layer": evictions / (S * L) if S * L > 0 else 0,
        "miss_rate": misses / (hits + misses) if (hits + misses) > 0 else 0
    }

def main():
    models = {
        "Mixtral": {"path": "mixtral_markov_transition_matrix.pt", "E": 8},
        "Qwen": {"path": "qwen_markov_transition_matrix.pt", "E": 60},
        "Switch": {"path": "switch_markov_transition_matrix.pt", "E": 128}
    }
    
    results = {}
    seq_len = 100
    batch_sizes = [16, 64, 128, 256]
    
    for m_name, m_info in models.items():
        print(f"\n[{m_name}] Loading module transition graph...")
        if not os.path.exists(m_info["path"]):
            print(f"  Missing {m_info['path']}! Please ensure extraction scripts have been run.")
            continue
            
        P_tensor = torch.load(m_info["path"])
        # Fix dtype if needed (Qwen extraction could save as float16/32)
        P_tensor = P_tensor.float()
        
        delta = compute_spectral_gap(P_tensor)
        print(f"  -> Spectral Gap ($\delta$): {delta:.4f}")
        
        results[m_name] = {
            "delta": delta,
            "experts": m_info["E"],
            "sweeps": []
        }
        
        # 10 capacity points from 10% to 100%
        capacities = [int(m_info["E"] * r) for r in np.linspace(0.1, 1.0, 10)]
        capacities = sorted(list(set(capacities)))
        if 0 in capacities: capacities.remove(0)
        
        for b in batch_sizes:
            print(f"  -> Simulating traces for Batch Size: {b}...")
            traces = generate_batched_trace(P_tensor, seq_len, b)
            
            for c in capacities:
                lru_res = simulate_cache(traces, c, policy="lru")
                # Admission threshold set conservatively (e.g. 0.05 probability)
                markov_res = simulate_cache(traces, c, policy="markov", admission_threshold=0.05, P_tensor=P_tensor)
                
                results[m_name]["sweeps"].append({
                    "batch_size": b,
                    "capacity": c,
                    "normalized_capacity": c / m_info["E"],
                    "lru": lru_res,
                    "markov": markov_res
                })
                
    out_file = "informs_simulation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Success] All regimes simulated. Output saved to {out_file}.")

if __name__ == "__main__":
    main()
