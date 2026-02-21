import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .expert_cache import ExpertCache, ExpertUID
from .expert_wrapper import MixtralExpertWrapper

class MarkovExpertCache(ExpertCache):
    """
    Expert Cache that tracks Temporal Markov Transitions (Directed Graph) 
    to perform O(1) cross-step lookahead prefetching based on Token T to Token T+1 transitions.
    """
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, num_layers: int = 32, num_experts: int = 8):
        super().__init__(make_module, main_size, offload_size, buffer_size)
        
        self.num_layers = num_layers
        self.num_experts = num_experts
        
        # Directed Transition Matrix: [num_layers, num_experts, num_experts]
        # P[layer_idx, from_expert, to_expert] = transition weight
        # Kept entirely on CPU for O(1) fast indexing
        self.transition_matrix = torch.zeros((num_layers, num_experts, num_experts), dtype=torch.float32)
        
        # Track the active experts from the previous token step for each layer
        self.last_active_experts: Dict[int, torch.Tensor] = {}
        
        self.alpha_transition = 1.0  # Weight added on observed transition
        self.decay_factor = 0.99     # Decay factor applied periodically
        self.step_counter = 0
        self.decay_interval = 100
        
        self.prediction_timers = {'t_markov_route': 0.0}
        
    def update_and_predict(self, layer_idx: int, curr_active_experts: torch.Tensor, top_k: int = 2) -> List[ExpertUID]:
        """
        Update the Markov transition matrix based on Token T-1 -> Token T, 
        and predict the experts for Token T+1.
        
        Args:
            layer_idx: The current layer being processed.
            curr_active_experts: Tensor containing expert indices active for the current token.
            
        Returns:
            List of ExpertUIDs to prefetch for the NEXT token at THIS layer.
        """
        t0 = time.perf_counter()
        
        # Flatten and unique the currently active experts
        curr_active = curr_active_experts.flatten().unique()
        
        # 1. UPDATE PHASE
        if layer_idx in self.last_active_experts:
            last_active = self.last_active_experts[layer_idx]
            
            # Simple O(1) indexing to update probabilities
            # For every expert active in T-1, they point to every expert active in T
            for u in last_active:
                for v in curr_active:
                    self.transition_matrix[layer_idx, u.item(), v.item()] += self.alpha_transition
                    
        # Store current as last for the next token step
        self.last_active_experts[layer_idx] = curr_active.clone()
        
        # Apply periodic decay to prioritize recent temporal locality
        self.step_counter += 1
        if self.step_counter % self.decay_interval == 0:
            self.transition_matrix *= self.decay_factor
            
        # 2. PREDICT PHASE (Predicting for Token T+1)
        # We look at the rows corresponding to currently active experts
        # and sum them to get the projected distribution for the next step.
        
        curr_indices = curr_active.tolist()
        if not curr_indices:
            self.prediction_timers['t_markov_route'] += time.perf_counter() - t0
            return []
            
        # Sum transition probabilities emanating from current active experts
        # Shape: (Num_Experts)
        predicted_scores = self.transition_matrix[layer_idx, curr_indices, :].sum(dim=0)
        
        # Optimization: ignore experts that are currently active (they are already loaded/locked)
        # Though technically they could be evicted and needed again, usually they are retained if recently used
        
        # Find the Top-K experts with highest transition probability
        predicted_uids = []
        if (predicted_scores > 0).any():
            # Get Number of experts with non-zero scores
            num_nonzero = (predicted_scores > 0).sum().item()
            actual_k = min(top_k, num_nonzero)
            
            if actual_k > 0:
                _, top_indices = torch.topk(predicted_scores, actual_k)
                predicted_uids = [(layer_idx, idx.item()) for idx in top_indices]
                
        self.prediction_timers['t_markov_route'] += time.perf_counter() - t0
        return predicted_uids
