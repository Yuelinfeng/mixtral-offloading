import numpy as np
import matplotlib.pyplot as plt
import os
import copy

MODELS = {
    "DeepSeek": {"N": 160, "S": 16, "C": 40, "gap": 0.925},
    "Mixtral":  {"N": 8,   "S": 4,   "C": 4,  "gap": 0.858},
    "Switch":   {"N": 128, "S": 16,  "C": 32, "gap": 0.583},
    "Qwen":     {"N": 60,  "S": 10,  "C": 15, "gap": 0.497},
}

class ExpertCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []
        self.misses = 0
        self.bypassed = 0
        
    def access_lru(self, requested_experts):
        for e in requested_experts:
            if e in self.cache:
                self.cache.remove(e)
                self.cache.append(e)
            else:
                self.misses += 1
                if len(self.cache) >= self.capacity:
                    self.cache.pop(0)
                self.cache.append(e)

    def access_optimal_admission(self, requested_experts, expected_hits):
        # The Formal INFORMS Admission Control Policy (Top-C Bounded)
        # S* = argmax_{|S| = C} ( expected_hits(e) )
        # We only admit an expert if it ranks in the global Top-C of expected hits.
        
        # Determine the cutoff score for the Top-C experts globally
        all_experts = list(expected_hits.keys())
        all_experts.sort(key=lambda x: expected_hits[x], reverse=True)
        top_c_experts = set(all_experts[:self.capacity])

        for e in requested_experts:
            if e in self.cache:
                self.cache.remove(e)
                self.cache.append(e)
            else:
                self.misses += 1 # A physical fetch occurs regardless
                if e in top_c_experts:
                    # Admit to cache
                    if len(self.cache) >= self.capacity:
                        self.cache.pop(0)
                    self.cache.append(e)
                else:
                    # Bypassed via Streaming Bypass (No cache pollution!)
                    self.bypassed += 1

def simulate(model_name, B, T=100):
    N = MODELS[model_name]["N"]
    S = MODELS[model_name]["S"]
    C = MODELS[model_name]["C"]
    gap = MODELS[model_name]["gap"]
    
    num_communities = N // S
    assert num_communities > 0
    
    tokens_comm = np.random.randint(0, num_communities, size=B)
    
    lru_cache = ExpertCache(C)
    ac_cache = ExpertCache(C)
    
    for step in range(T):
        requested = set()
        
        # Calculate exactly P(return) state for AC *before* moving tokens (zero-lag radar)
        comm_counts = np.bincount(tokens_comm, minlength=num_communities)
        expected_hits = {}
        for c in range(num_communities):
            tokens_in_c = comm_counts[c]
            for offset in range(S):
                e = c * S + offset
                local_hits = tokens_in_c * (1 - gap) / S
                escape_hits = B * gap / N
                expected_hits[e] = local_hits + escape_hits
                
        # Transition tokens
        for i in range(B):
            if np.random.rand() < gap:
                next_e = np.random.randint(0, N)
                tokens_comm[i] = next_e // S
            else:
                offset = np.random.randint(0, S)
                next_e = tokens_comm[i] * S + offset
            requested.add(next_e)
            
        lru_cache.access_lru(requested)
        ac_cache.access_optimal_admission(requested, expected_hits)
        
    return lru_cache.misses, ac_cache.misses, ac_cache.bypassed

def run_sweeps():
    batches = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    results = {m: {"lru": [], "ac": []} for m in MODELS.keys()}
    
    print(f"{'Model':<10} | {'Batch':<5} | {'LRU_Miss':<10} | {'AC_Miss':<10} | {'AC_Bypassed':<10}")
    print("-" * 65)
    
    for m in MODELS.keys():
        for b in batches:
            lru_misses, ac_misses, bypassed = simulate(m, b, T=200)
            results[m]["lru"].append(lru_misses)
            results[m]["ac"].append(ac_misses)
            print(f"{m:<10} | {b:<5} | {lru_misses:<10} | {ac_misses:<10} | {bypassed:<10}")
            
    # Visualize the Phase Transition / Avalanche
    plt.figure(figsize=(14, 10))
    for i, m in enumerate(MODELS.keys()):
        plt.subplot(2, 2, i+1)
        plt.title(f"{m} (Gap: {MODELS[m]['gap']})")
        plt.plot(batches, results[m]["lru"], label="LRU", marker='o', color='red')
        plt.plot(batches, results[m]["ac"], label="Admission Control", marker='s', color='blue')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.xlabel("Batch Size (B)")
        plt.ylabel("H2D Load Events (Misses)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig("regimes_sweep_top_c.png", dpi=300)

if __name__ == "__main__":
    run_sweeps()
