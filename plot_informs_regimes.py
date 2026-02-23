import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(path="informs_simulation_results.json"):
    with open(path, "r") as f:
        return json.load(f)

def plot_delta_vs_collapse(data):
    """
    Fig 1: Delta vs Empirical Collapse Point
    Collapse point is defined as the normalized capacity required for LRU to reach < 10% miss rate (B=16)
    """
    models = list(data.keys())
    deltas = [data[m]["delta"] for m in models]
    
    collapse_points = []
    for m in models:
        sweeps = [s for s in data[m]["sweeps"] if s["batch_size"] == 16]
        c_point = 1.0 # default to 100% capacity
        for s in sweeps:
            if s["lru"]["miss_rate"] < 0.10:
                c_point = s["normalized_capacity"]
                break
        collapse_points.append(c_point)
        
    plt.figure(figsize=(6, 5))
    plt.scatter(deltas, collapse_points, color='blue', s=100, zorder=5)
    
    for i, m in enumerate(models):
        plt.annotate(m, (deltas[i], collapse_points[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
        
    # Fit a trendline
    if len(models) >= 2:
        z = np.polyfit(deltas, collapse_points, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(deltas)*0.9, max(deltas)*1.1, 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.6, label="Trendline")
        
    plt.title("Empirical Avalanche Threshold vs. Spectral Gap ($\delta$)")
    plt.xlabel("Markov Spectral Gap ($\delta$)")
    plt.ylabel("Avalanche Threshold Capacity (B*)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig1_delta_collapse.png", dpi=300)
    plt.close()

def plot_admission_comparison(data, model="Qwen", batch_size=64):
    """
    Fig 2: Admission ON vs OFF for Churn and Miss Rate (H2D/P99 proxy)
    """
    sweeps = [s for s in data[model]["sweeps"] if s["batch_size"] == batch_size]
    
    caps = [s["normalized_capacity"] for s in sweeps]
    
    lru_churn = [s["lru"]["churn_per_step_layer"] for s in sweeps]
    markov_churn = [s["markov"]["churn_per_step_layer"] for s in sweeps]
    
    lru_miss = [s["lru"]["miss_rate"] for s in sweeps]
    markov_miss = [s["markov"]["miss_rate"] for s in sweeps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Churn plot
    ax1.plot(caps, lru_churn, 'o-', label='LRU (Admission OFF)', linewidth=2, color='#ee6666')
    ax1.plot(caps, markov_churn, 's-', label='Markov (Admission ON)', linewidth=2, color='#5470c6')
    ax1.set_title(f"Cache Churn ({model}, B={batch_size})")
    ax1.set_xlabel("Normalized Cache Capacity")
    ax1.set_ylabel("Evictions per step per layer")
    ax1.grid(True, alpha=0.5)
    ax1.legend()
    
    # H2D/Miss Rate plot
    ax2.plot(caps, lru_miss, 'o-', label='LRU (Admission OFF)', linewidth=2, color='#ee6666')
    ax2.plot(caps, markov_miss, 's-', label='Markov (Admission ON)', linewidth=2, color='#5470c6')
    ax2.set_title(f"Miss Rate / H2D Proxy ({model}, B={batch_size})")
    ax2.set_xlabel("Normalized Cache Capacity")
    ax2.set_ylabel("Cache Miss Rate")
    ax2.grid(True, alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("fig2_admission_comparison.png", dpi=300)
    plt.close()
    
def plot_multi_model(data, batch_size=128, target_cap=0.5):
    """
    Fig 3: Multi-model comparison at fixed batch size and capacity.
    """
    models = list(data.keys())
    lru_misses = []
    markov_misses = []
    
    for m in models:
        sweeps = [s for s in data[m]["sweeps"] if s["batch_size"] == batch_size]
        # Find closest capacity
        closest = min(sweeps, key=lambda x: abs(x["normalized_capacity"] - target_cap))
        lru_misses.append(closest["lru"]["miss_rate"] * 100)
        markov_misses.append(closest["markov"]["miss_rate"] * 100)
        
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, lru_misses, width, label='LRU (OFF)', color='#ee6666')
    rects2 = ax.bar(x + width/2, markov_misses, width, label='Markov (ON)', color='#5470c6')
    
    ax.set_ylabel('Miss Rate (%)')
    ax.set_title(f'Multi-Model Admission Impact (B={batch_size}, Cap={target_cap*100:.0f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.1f', padding=3)
    ax.bar_label(rects2, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    plt.savefig("fig3_multimodel.png", dpi=300)
    plt.close()

def plot_batch_sweep(data, model="Qwen", target_cap=0.5):
    """
    Fig 4: Batch Size Sweep
    """
    # Group by batch size at closest capacity
    batch_sizes = sorted(list(set([s["batch_size"] for s in data[model]["sweeps"]])))
    
    lru_miss = []
    markov_miss = []
    lru_churn = []
    markov_churn = []
    
    for b in batch_sizes:
        sweeps = [s for s in data[model]["sweeps"] if s["batch_size"] == b]
        closest = min(sweeps, key=lambda x: abs(x["normalized_capacity"] - target_cap))
        
        lru_miss.append(closest["lru"]["miss_rate"])
        markov_miss.append(closest["markov"]["miss_rate"])
        
        # Normalize churn by batch size to get churn per token
        lru_churn.append(closest["lru"]["churn_per_step_layer"] / b)
        markov_churn.append(closest["markov"]["churn_per_step_layer"] / b)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(batch_sizes, lru_miss, 'o-', label='LRU', color='#ee6666')
    ax1.plot(batch_sizes, markov_miss, 's-', label='Markov', color='#5470c6')
    ax1.set_title(f"Miss Rate vs Batch Size ({model}, Cap={target_cap*100:.0f}%)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Miss Rate")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(batch_sizes)
    ax1.grid(True, alpha=0.5)
    ax1.legend()
    
    ax2.plot(batch_sizes, lru_churn, 'o-', label='LRU', color='#ee6666')
    ax2.plot(batch_sizes, markov_churn, 's-', label='Markov', color='#5470c6')
    ax2.set_title(f"Token-Normalized Churn vs Batch Size")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Evictions per token per layer")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(batch_sizes)
    ax2.grid(True, alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("fig4_batch_sweep.png", dpi=300)
    plt.close()

def main():
    data = load_data()
    print("Generating Figure 1: Delta vs Collapse Point...")
    plot_delta_vs_collapse(data)
    
    print("Generating Figure 2: Admission ON vs OFF (Qwen)...")
    plot_admission_comparison(data, model="Qwen", batch_size=64)
    
    print("Generating Figure 3: Multi-model Comparison...")
    plot_multi_model(data, batch_size=128, target_cap=0.4)
    
    print("Generating Figure 4: Batch Size Sweep...")
    plot_batch_sweep(data, model="Qwen", target_cap=0.5)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
