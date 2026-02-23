import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import scipy.linalg as linalg
import os

def analyze_markov_layer(P_tensor, layer_idx):
    """
    Analyzes the structure of a single layer's transition matrix.
    Args:
        P_tensor: Shape (N, N) raw transition weights.
    """
    print(f"\n{'='*40}")
    print(f"Layer {layer_idx} Analysis")
    print(f"{'='*40}")
    
    # Add a small epsilon to avoid divide by zero if a row is all zeros
    row_sums = P_tensor.sum(dim=1, keepdim=True)
    # To avoid NaNs for entirely unvisited states
    row_sums[row_sums == 0] = 1.0 
    
    # 1. Normalize to Probability Matrix P
    P = (P_tensor / row_sums).numpy()
    N = P.shape[0]
    
    print("Transition Matrix P (Row-Normalized):")
    np.set_printoptions(precision=3, suppress=True)
    print(P)
    
    # 2. Extract out-degree distribution (How many non-zero outgoing transitions)
    # We consider a transition "existent" if it's > 5% probability
    binary_P = (P > 0.05).astype(int)
    out_degrees = binary_P.sum(axis=1)
    print(f"\nOut-degrees (threshold > 0.05): {out_degrees}")
    print(f"Average out-degree: {out_degrees.mean():.2f}")
    
    # 3. Compute Strongly Connected Components (SCC)
    # We use the binary adjacency matrix
    n_components, labels = connected_components(csgraph=sp.csr_matrix(binary_P), directed=True, connection='strong')
    print(f"\nStrongly Connected Components (SCC count): {n_components}")
    print(f"SCC Labels per node: {labels}")
    if n_components == 1:
        print("-> The graph is strongly connected (no isolated macro-clusters).")
    elif n_components == N:
        print("-> The graph has no meaningful cycles (completely disjoint or DAG).")
    else:
        print("-> The graph has partial clustering/communities.")
        
    # 4. Compute Spectral Gap (Mixing time proxy)
    # We look at the eigenvalues of the transition matrix P
    try:
        eigenvalues, _ = linalg.eig(P)
        # Sort eigenvalues by magnitude
        sorted_evals = np.sort(np.abs(eigenvalues))[::-1]
        
        # The largest eigenvalue for a Markov matrix is theoretically 1
        lambda_1 = sorted_evals[0]
        # The second largest eigenvalue magnitude governs the mixing rate
        if len(sorted_evals) > 1:
            lambda_2 = sorted_evals[1]
            spectral_gap = lambda_1 - lambda_2
            print(f"\nSpectral Properties:")
            print(f"  |lambda_1|: {lambda_1:.4f}")
            print(f"  |lambda_2|: {lambda_2:.4f}")
            print(f"  Spectral Gap (1 - |lambda_2|): {spectral_gap:.4f}")
            
            if spectral_gap > 0.5:
                print("  -> Fast mixing (Rapid convergence to stationary). Working set will expand VERY quickly. (Large B* is catastrophic)")
            else:
                print("  -> Slow mixing (Stronger temporal locality). Working set expands slowly. (Large B* is tolerable)")
    except Exception as e:
        print(f"Could not compute eigenvalues: {e}")
        
    print(f"-"*40)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, default="markov_transition_matrix.pt")
    args = parser.parse_args()
    
    matrix_path = args.matrix
    if not os.path.exists(matrix_path):
        print(f"File not found: {matrix_path}")
        print("Please run the benchmark with --policy markov first.")
        exit(1)
        
    print(f"Loading transition matrix from {matrix_path}...")
    # Shape should be [32, 8, 8]
    transition_matrix = torch.load(matrix_path)
    print(f"Tensor shape: {transition_matrix.shape}")
    
    num_layers = transition_matrix.shape[0]
    
    # Analyze first, middle, and last layer to get a representative look
    layers_to_analyze = [0, num_layers // 2, num_layers - 1]
    
    for l_idx in layers_to_analyze:
        analyze_markov_layer(transition_matrix[l_idx], l_idx)
    
    print("\n[GLOBAL CONCLUSION]")
    print("If SCC count is mostly 1, and Spectral Gap is large, it proves the INFORMS structural hypothesis:")
    print("1. MoE routing graphs lack macro-community structure (no static clusters to bundle).")
    print("2. The graph mixes rapidly, explaining why Batched Union Working Set rapidly expands and collapses Cache.")
    print("-> Justifies why Return-Probability Admission Control (Strategy B) is the mathematically correct optimization.")
