import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .expert_cache import ExpertCache, ExpertUID
from .expert_wrapper import MixtralExpertWrapper

class EmbeddingExpertCache(ExpertCache):
    """
    Expert Cache that tracks Embedding Centroids and uses Hidden States 
    to perform cross-layer lookahead prefetching.
    """
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, num_layers: int = 32, num_experts: int = 8, hidden_dim: int = 4096, device: str = "cuda:0"):
        super().__init__(make_module, main_size, offload_size, buffer_size)
        
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Centroid Tracking: [num_layers, num_experts, hidden_dim]
        # Initialized to zero. Will be populated via EMA.
        self.centroids = torch.zeros((num_layers, num_experts, hidden_dim), device=self.device, dtype=torch.float16)
        
        # Track whether a centroid has been initialized (to avoid EMA bias on start)
        self.centroid_initialized = torch.zeros((num_layers, num_experts), device=self.device, dtype=torch.bool)
        
        self.alpha_ema = 0.1 # Weight for new data
        self.prediction_timers = {'t_predict': 0.0, 't_update': 0.0}
        
    def update_centroids(self, layer_idx: int, hidden_states: torch.Tensor, selected_experts: torch.Tensor):
        """
        Update the semantic centroids of experts based on the tokens routed to them.
        
        Args:
            layer_idx: Current layer index.
            hidden_states: (Batch*SeqLen, HiddenDim) tensor composed of tokens.
            selected_experts: (Batch*SeqLen, TopK) tensor of which experts were chosen.
        """
        t0 = time.perf_counter()
        
        # Determine which experts were active
        active_experts = selected_experts.flatten().unique()
        
        for expert_idx in active_experts:
            expert_idx_item = expert_idx.item()
            # Find tokens routed to this expert
            mask = (selected_experts == expert_idx).any(dim=-1)
            tokens_for_expert = hidden_states[mask]
            
            if tokens_for_expert.shape[0] == 0:
                continue
                
            # Compute mean of tokens routed to this expert in this batch
            batch_centroid = tokens_for_expert.mean(dim=0)
            
            if not self.centroid_initialized[layer_idx, expert_idx_item]:
                self.centroids[layer_idx, expert_idx_item] = batch_centroid
                self.centroid_initialized[layer_idx, expert_idx_item] = True
            else:
                # Exponential Moving Average
                self.centroids[layer_idx, expert_idx_item] = (
                    (1 - self.alpha_ema) * self.centroids[layer_idx, expert_idx_item] + 
                    self.alpha_ema * batch_centroid
                )
        
        self.prediction_timers['t_update'] += time.perf_counter() - t0

    def predict_next_layer(self, current_layer_idx: int, hidden_states: torch.Tensor, top_k: int = 2) -> List[ExpertUID]:
        """
        Predict which experts will be needed in the NEXT layer based on the current hidden states trajectory.
        
        Args:
            current_layer_idx: The layer the tokens just passed through.
            hidden_states: Current token embeddings.
            
        Returns:
            List of ExpertUIDs to prefetch.
        """
        t0 = time.perf_counter()
        target_layer_idx = current_layer_idx + 1
        
        if target_layer_idx >= self.num_layers:
            return []
            
        # Check if target layer centroids are roughly initialized
        if not self.centroid_initialized[target_layer_idx].any():
             # Cannot predict if next layer has no profile
             return []
             
        # Aggregate the hidden states of the batch to represent the "current semantic direction"
        # Since we use greedy routing, predicting based on the batch mean or batch distinct points.
        # For simplicity and speed, let's take the mean trajectory of the entire active prompt.
        batch_mean_state = hidden_states.mean(dim=0, keepdim=True) # (1, HiddenDim)
        
        target_centroids = self.centroids[target_layer_idx] # (NumExperts, HiddenDim)
        
        # Calculate Cosine Similarity
        # Normalize vectors for cosine sim: (A dot B) / (|A| |B|)
        batch_mean_norm = torch.nn.functional.normalize(batch_mean_state, p=2, dim=-1)
        target_centroids_norm = torch.nn.functional.normalize(target_centroids, p=2, dim=-1)
        
        similarities = torch.matmul(batch_mean_norm, target_centroids_norm.transpose(0, 1)).squeeze(0) # (NumExperts,)
        
        # Ignore uninitialized centroids by setting their sim to -inf
        uninitialized_mask = ~self.centroid_initialized[target_layer_idx]
        similarities.masked_fill_(uninitialized_mask, float('-inf'))
        
        # Get Top-K
        if uninitialized_mask.all():
            return []
            
        actual_top_k = min(top_k, (~uninitialized_mask).sum().item())
        if actual_top_k <= 0:
            return []
            
        _, top_expert_indices = torch.topk(similarities, actual_top_k)
        
        predicted_uids = [(target_layer_idx, idx.item()) for idx in top_expert_indices]
        
        self.prediction_timers['t_predict'] += time.perf_counter() - t0
        return predicted_uids
