import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_metrics(P_tensor):
    # Normalize to probability matrix P
    row_sums = P_tensor.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0 
    P = (P_tensor / row_sums).numpy()
    
    # Compute SCC
    binary_P = (P > 0.05).astype(int)
    n_components, _ = connected_components(csgraph=sp.csr_matrix(binary_P), directed=True, connection='strong')
    
    # Compute Spectral Gap
    spectral_gap = 0
    try:
        eigenvalues, _ = linalg.eig(P)
        sorted_evals = np.sort(np.abs(eigenvalues))[::-1]
        if len(sorted_evals) > 1:
            spectral_gap = 1.0 - sorted_evals[1]
    except Exception:
        pass
        
    return P, n_components, spectral_gap

def plot_matrices():
    # Attempt to load whatever traces have been successfully extracted
    files = {
        "Mixtral-8x7B\n(8 Experts, Top-2)": "mixtral_markov_transition_matrix.pt",
        "Qwen1.5-MoE\n(60 Experts, Top-4)": "qwen_markov_transition_matrix.pt",
        "DeepSeek-V2-Lite\n(64 Experts, Top-6)": "deepseek_markov_transition_matrix.pt",
        "Switch-Base-128\n(128 Experts, Top-1)": "switch_markov_transition_matrix.pt"
    }
    
    existing_files = {k: v for k, v in files.items() if os.path.exists(v)}
    
    if not existing_files:
        print("No transition matrix files found in the current directory!")
        return
        
    print(f"Found {len(existing_files)} matrices to plot.")
    
    # Set up matplotlib figure
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, len(existing_files), figsize=(6 * len(existing_files), 5.5))
    if len(existing_files) == 1:
        axes = [axes]
        
    for ax, (title_base, filepath) in zip(axes, existing_files.items()):
        print(f"Processing {filepath}...")
        tensor = torch.load(filepath)
        
        # Take a layer from the middle of the network as a representative sample
        l_idx = tensor.shape[0] // 2  
        layer_tensor = tensor[l_idx]
        
        P, scc, gap = compute_metrics(layer_tensor)
        
        # Plot Heatmap
        sns.heatmap(P, ax=ax, cmap="Blues", cbar=True, 
                    xticklabels=False, yticklabels=False,
                    vmin=0, vmax=0.5) # Cap vmax to 0.5 to make sparse connections visible
        
        ax.set_title(f"{title_base}\nMiddle Layer: {l_idx}\nSCC Count: {scc} | Spectral Gap: {gap:.3f}", 
                     fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel("Next Expert $v$", fontsize=11)
        ax.set_ylabel("Current Expert $u$", fontsize=11)
        
        # Add a thick border around the heatmap for aesthetics
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

    plt.suptitle("INFORMS Topology Validation: Transition Phase & Working Set Mixing", 
                 fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    # Save the figure
    out_file = "moe_topology_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Visualization saved to {out_file}")

if __name__ == "__main__":
    plot_matrices()
