import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .expert_cache import ExpertCache, ExpertUID, EvictionGroupInfo, ExpertInfo
from .expert_wrapper import MixtralExpertWrapper

class MarkovEvictionGroupInfo(EvictionGroupInfo):
    def __init__(self, admission_threshold=0.0):
        super().__init__()
        self.expert_return_probs = {} # Dict[ExpertUID, float]
        self.admission_threshold = admission_threshold
        self.bypassed_count = 0
        self.t_graph_update = 0.0
        
    def admit(self, info: ExpertInfo) -> bool:
        if self.admission_threshold <= 0.0:
            return True
        prob = self.expert_return_probs.get(info.uid, 0.0)
        # If the expected return probability is >= threshold, admit it. Otherwise bypass.
        return prob >= self.admission_threshold

    def choose_expert_to_evict(self) -> ExpertInfo:
        # Instead of strict LRU, evict the expert with the lowest return probability
        t0 = time.perf_counter()
        if not self.main_infos:
            raise ValueError("No evictable experts")
            
        lowest_prob = float('inf')
        victim = None
        
        # main_infos is OrderedDict, iteration is inherently LRU to MRU.
        # This breaks ties (e.g. probs=0) by picking the least recently used zero-prob expert.
        for uid, info in self.main_infos.items():
            prob = self.expert_return_probs.get(uid, 0.0)
            if prob < lowest_prob:
                lowest_prob = prob
                victim = info
                
        self.t_evict_search += time.perf_counter() - t0
        return victim

class MarkovExpertCache(ExpertCache):
    """
    Expert Cache that tracks Temporal Markov Transitions (Directed Graph) 
    to perform O(1) cross-step lookahead prefetching based on Token T to Token T+1 transitions.
    """
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, num_layers: int = 32, num_experts: int = 8, admission_threshold: float = 0.0):
        super().__init__(make_module, main_size, offload_size, buffer_size)
        
        self.admission_threshold = admission_threshold
        
        # Override defaultdict to use MarkovEvictionGroupInfo
        from collections import defaultdict
        self.group_infos = defaultdict(lambda: MarkovEvictionGroupInfo(admission_threshold=self.admission_threshold))
        
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
        
    def update_and_predict(self, layer_idx: int, curr_selected_experts: torch.Tensor, top_k: int = 2) -> List[ExpertUID]:
        """
        Update the Markov transition matrix based on Token T -> Token T+1 inside the micro-batch,
        and predict the most likely experts for the overall batch.
        
        Args:
            layer_idx: The current layer being processed.
            curr_selected_experts: Tensor of shape (Batch*SeqLen, Top_K) containing expert selections.
            
        Returns:
            List of ExpertUIDs to prefetch for the NEXT token at THIS layer.
        """
        t0 = time.perf_counter()
        
        # 1. UPDATE PHASE (Token T to Token T+1 Temporal Tracking)
        # To avoid the KxK cross-product rank-1 homogenization (similar to our Trace findings),
        # we will track the strict Top-1 to Top-1 semantic jumps inside this sequence.
        seq_len = curr_selected_experts.shape[0]
        
        if seq_len > 1:
            # Vectorized or fast looped transition tracking
            top1_experts = curr_selected_experts[:, 0]
            
            # Count transitions
            for t in range(seq_len - 1):
                u = top1_experts[t].item()
                v = top1_experts[t+1].item()
                self.transition_matrix[layer_idx, u, v] += self.alpha_transition
                
            # Inter-batch transition (last token of prev batch -> first token of this batch)
            if layer_idx in self.last_active_experts:
                u = self.last_active_experts[layer_idx].item()
                v = top1_experts[0].item()
                self.transition_matrix[layer_idx, u, v] += self.alpha_transition
                
            self.last_active_experts[layer_idx] = top1_experts[-1]
            
        # Flatten and unique the currently active experts across the entire micro-batch
        curr_active_batch = curr_selected_experts.flatten().unique()
        
        # Apply periodic decay to prioritize recent temporal locality
        self.step_counter += 1
        if self.step_counter % self.decay_interval == 0:
            self.transition_matrix *= self.decay_factor
            
        # 2. PREDICT PHASE (Predicting for the active experts)
        # We look at the rows corresponding to currently active experts
        # and sum them to get the projected distribution.
        
        curr_indices = curr_active_batch.tolist()
        if not curr_indices:
            self.timers['t_markov_route'] += time.perf_counter() - t0
            return []
            
        # Sum transition probabilities emanating from current active experts
        # Shape: (Num_Experts)
        predicted_scores = self.transition_matrix[layer_idx, curr_indices, :].sum(dim=0)
        
        # --- Update P(Return) into the Admission Controller ---
        if layer_idx in self.group_infos:
            group = self.group_infos[layer_idx]
            if isinstance(group, MarkovEvictionGroupInfo):
                sum_scores = predicted_scores.sum()
                if sum_scores > 1e-6:
                    probs = predicted_scores / sum_scores
                    for idx, p in enumerate(probs.tolist()):
                        group.expert_return_probs[(layer_idx, idx)] = p
        # --------------------------------------------------------
        
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
                
        self.timers['t_markov_route'] += time.perf_counter() - t0
        # Add to graph update timer
        self.group_infos[layer_idx].t_graph_update += time.perf_counter() - t0
        
        return predicted_uids
