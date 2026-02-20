from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List, Set
from collections import deque, defaultdict, OrderedDict
import torch
from torch import nn

from .expert_cache import ExpertCache, ExpertUID, ExpertInfo, EvictionGroupInfo
from .expert_wrapper import MixtralExpertWrapper


@dataclass
class GraphEvictionGroupInfo(EvictionGroupInfo):
    """
    Extends EvictionGroupInfo to support Graph-based eviction.
    Maintains a similarity matrix between experts.
    """
    # Expert Similarity Matrix W (EMA updated)
    # Shape: (total_experts, total_experts)
    # We need to map ExpertUID -> functional index (0..N-1) for matrix operations
    # However, since this group is for a specific layer, and experts are indexed 0..7 usually,
    # we can just use the expert_index from the UID (layer_idx, expert_idx).
    
    # We assume standard Mixtral with 8 experts per layer. 
    # If dynamic, we might need a mapping. For now, let's assume max 8.
    # To be safe, we can use a slightly larger matrix or resize.
    # But usually `num_local_experts` is constant.
    
    similarity_matrix: Optional[torch.Tensor] = None
    alpha: float = 0.1  # EMA decay factor
    
    # Track current active experts for the ongoing forward pass (set by load_experts)
    current_active_experts: Set[int] = field(default_factory=set)

    def update_graph(self, gating_probs: torch.Tensor):
        """
        Updates the Expert Co-occurrence Graph (Similarity Matrix) using EMA.
        
        Args:
            gating_probs: (Batch_Size, Num_Experts) float tensor on CPU or GPU.
                          Columns correspond to expert indices.
        """
        # Ensure computation is on CPU to avoid GPU overhead for graph logic
        # and because we might be in the middle of GPU op.
        # Although, moving large batch to CPU might be slow?
        # The prompt says: "Computation MUST be on CPU".
        
        with torch.no_grad():
            probs = gating_probs.detach().cpu().float()
            batch_size, num_experts = probs.shape
            
            # Initialize matrix if needed
            if self.similarity_matrix is None or self.similarity_matrix.shape[0] != num_experts:
                self.similarity_matrix = torch.eye(num_experts, dtype=torch.float32)
                
            # 1. Normalize columns (experts) to unit vectors for Cosine Similarity
            # expert_vectors: (Batch, N) -> each column is an expert vector
            # We want W[i, j] = CosSim(Expert_i, Expert_j)
            
            # epsilon to avoid division by zero
            norm = torch.norm(probs, p=2, dim=0, keepdim=True) + 1e-8
            normalized_probs = probs / norm
            
            # 2. Compute Cosine Similarity Matrix: C = V^T @ V
            # Shape: (N, B) @ (B, N) -> (N, N)
            current_similarity = torch.mm(normalized_probs.T, normalized_probs)
            
            # 3. EMA Update: W_new = alpha * C + (1 - alpha) * W_old
            # We use specific alpha for EMA
            self.similarity_matrix = self.alpha * current_similarity + (1 - self.alpha) * self.similarity_matrix
            
            # Ensure diagonal is 1 (self-similarity) - optional but good for stability
            self.similarity_matrix.fill_diagonal_(1.0)

    def choose_expert_to_evict(self) -> ExpertInfo:
        """
        Selects the expert to evict based on Collaborative Score.
        Score(e) = sum(Similarity(e, a) for a in Active_Set)
        We evict the expert in `main_infos` (GPU) with the LOWEST score.
        """
        if not self.main_infos:
            raise ValueError("No evictable experts")
            
        # If no similarity matrix yet (or cold start), fallback to LRU (first in OrderedDict)
        if self.similarity_matrix is None:
            # OrderedDict in ExpertCache is ordered by access time (LRU is first)
            # We treat the first item as LRU.
            k = next(iter(self.main_infos))
            return self.main_infos[k]

        # Calculate scores for all candidates in main_infos
        # Candidates are experts currently on GPU
        candidates = list(self.main_infos.values())
        
        scores = []
        for info in candidates:
            # Expert UID is (layer_idx, expert_idx)
            # We need expert_idx
            e_idx = info.uid[1]
            
            # If expert index is out of bounds of our matrix (shouldn't happen if consistent), handle safely
            if e_idx >= self.similarity_matrix.shape[0]:
                scores.append(-1.0) # Evict this anomaly?
                continue

            # Collaborative Score: Sum of similarities to currently active experts
            # We only care about correlation with the *fresh* active set of this batch
            score = 0.0
            for active_idx in self.current_active_experts:
                if active_idx < self.similarity_matrix.shape[0]:
                    score += self.similarity_matrix[e_idx, active_idx].item()
            
            scores.append(score)
            
        # Find index of minimum score
        # If there are ties, we might want to use LRU as tie-breaker.
        # Since candidates are ordered by LRU (from OrderedDict), `min` will pick the first one if ties?
        # Python's min is stable for ties? No, it returns the first one encountered.
        # Candidates list respects OrderedDict order? Yes.
        # So if we iterate, the first one (LRU) with min score is picked.
        
        min_score_idx = 0
        min_score = scores[0]
        
        for i in range(1, len(scores)):
            if scores[i] < min_score:
                min_score = scores[i]
                min_score_idx = i
                
        return candidates[min_score_idx]


class GraphExpertCache(ExpertCache):
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int):
        super().__init__(make_module, main_size, offload_size, buffer_size)
        # Override the default EvictionGroupInfo with our Graph-based one
        # access base class attr? No, just re-initialize or make `group_infos` use the new class
        self.group_infos: Dict[int, GraphEvictionGroupInfo] = defaultdict(GraphEvictionGroupInfo)

    def load_experts(
            self, 
            *uids: ExpertUID, 
            unordered: bool = False, 
            gating_probs: Optional[torch.Tensor] = None
        ) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        Loads experts with awareness of Gating Probabilities for Graph Update.
        
        Args:
            uids: List of ExpertUIDs to load (the 'active' experts for this batch).
            unordered: If True, yields in optimal order.
            gating_probs: Tensor of shape (Batch, Num_Experts).
        """
        if len(uids) == 0:
            return
            
        # Identify the eviction group (should be same for all uids)
        # We peek at the first uid to get the group
        # NOTE: self.registered_experts might not be populated if we haven't added them?
        # But `load_experts` assumes they are registered.
        
        first_info = self.registered_experts[uids[0]]
        group_id = first_info.eviction_group
        eviction_group = self.group_infos[group_id]
        
        # 1. Update Graph if probs provided
        if gating_probs is not None:
            eviction_group.update_graph(gating_probs)

        # 2. Set current active experts for this batch (for score calculation)
        active_indices = {uid[1] for uid in uids} # uid is (layer, expert_idx)
        eviction_group.current_active_experts = active_indices
        
        # 3. Proceed with standard loading (which uses choose_expert_to_evict internally)
        # We call the super class method. 
        # But wait, super().load_experts doesn't accept `gating_probs`.
        # And keep in mind Python doesn't support overloading well.
        # If I call super().load_experts(..., gating_probs=...), it will fail.
        
        # We need to rely on the fact that `choose_expert_to_evict` is called by the base class logic.
        # Since we injected GraphEvictionGroupInfo, the base class will call OUR `choose_expert_to_evict`.
        # So we just need to pass the arguments that the base class accepts.
        
        return super().load_experts(*uids, unordered=unordered)
