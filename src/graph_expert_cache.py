from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List, Set
from collections import deque, defaultdict, OrderedDict
import torch
from torch import nn
import time

from .expert_cache import ExpertCache, ExpertUID, ExpertInfo, EvictionGroupInfo
from .expert_wrapper import MixtralExpertWrapper


@dataclass
class GraphEvictionGroupInfo(EvictionGroupInfo):
    """
    Extends EvictionGroupInfo to support Graph-based eviction.
    Maintains a similarity matrix between experts.
    """
    lambda_weight: float = 1.0
    similarity_matrix: Optional[torch.Tensor] = None
    alpha: float = 0.6  # EMA decay factor
    current_active_experts: Set[int] = field(default_factory=set)
    t_graph_update: float = 0.0
    t_evict_search: float = 0.0

    # Expert Similarity Matrix W (EMA updated)
    # Shape: (total_experts, total_experts)
    # We need to map ExpertUID -> functional index (0..N-1) for matrix operations
    # However, since this group is for a specific layer, and experts are indexed 0..7 usually,
    # we can just use the expert_index from the UID (layer_idx, expert_idx).
    
    # We assume standard Mixtral with 8 experts per layer. 
    # If dynamic, we might need a mapping. For now, let's assume max 8.
    # To be safe, we can use a slightly larger matrix or resize.
    # But usually `num_local_experts` is constant.
    
    # Track current active experts for the ongoing forward pass (set by load_experts)

    # Profiling Timers


    def update_graph(self, gating_probs: torch.Tensor):
        """
        Updates the Expert Co-occurrence Graph (Similarity Matrix) using EMA.
        
        Args:
            gating_probs: (Batch_Size, Num_Experts) float tensor on CPU or GPU.
                          Columns correspond to expert indices.
        """
        # Move computation to CPU as requested to avoid GPU overhead and interference
        t0 = time.perf_counter()
        with torch.no_grad():
            # Transfer probs to CPU
            probs = gating_probs.detach().cpu().float()
            batch_size, num_experts = probs.shape
            
            # Initialize matrix on CPU if needed
            if self.similarity_matrix is None or self.similarity_matrix.shape[0] != num_experts:
                self.similarity_matrix = torch.eye(num_experts, dtype=torch.float32, device='cpu')
            
            # Ensure matrix is on CPU (in case it was previously on GPU)
            if self.similarity_matrix.device.type != 'cpu':
                self.similarity_matrix = self.similarity_matrix.to('cpu')
                
            # 1. Compute HARD Co-occurrence Matrix
            # Find the actual Top-2 activated experts for each token
            topk_indices = torch.topk(probs, 2, dim=1).indices
            binary_probs = torch.zeros_like(probs)
            # Scatter 1.0 precisely at the Top-2 indices
            binary_probs.scatter_(1, topk_indices, 1.0)
            
            # C = P^T @ P (Binary Dot Product = Co-occurrence Count)
            current_similarity = torch.mm(binary_probs.T, binary_probs)
            
            # Normalize by batch size
            current_similarity = current_similarity / batch_size
            
            # 3. EMA Update: W_new = alpha * C + (1 - alpha) * W_old
            # Entirely on CPU
            self.similarity_matrix = self.alpha * current_similarity + (1 - self.alpha) * self.similarity_matrix
            
            # Ensure diagonal is 1
            self.similarity_matrix.fill_diagonal_(1.0)
        self.t_graph_update += time.perf_counter() - t0

    def choose_expert_to_evict(self) -> ExpertInfo:
        """
        Selects the expert to evict based on Collaborative Score.
        """
        t0 = time.perf_counter()
        if not self.main_infos:
            self.t_evict_search += time.perf_counter() - t0
            raise ValueError("No evictable experts")
            
        # If no similarity matrix yet, fallback to LRU
        if self.similarity_matrix is None:
            k = next(iter(self.main_infos))
            self.t_evict_search += time.perf_counter() - t0
            return self.main_infos[k]

        candidates = list(self.main_infos.values())
        
        # Prepare indices for scoring on CPU
        # Ensure similarity matrix is on CPU
        if self.similarity_matrix.device.type != 'cpu':
             self.similarity_matrix = self.similarity_matrix.to('cpu')
             
        device = 'cpu'
        
        # Candidate indices (experts currently on GPU)
        candidate_indices = torch.tensor([info.uid[1] for info in candidates], device=device, dtype=torch.long)
        
        # Active indices (experts needed for current batch)
        if not self.current_active_experts:
            # If no active experts (rare), fallback to LRU
            k = next(iter(self.main_infos))
            return self.main_infos[k]
            
        active_indices = torch.tensor(list(self.current_active_experts), device=device, dtype=torch.long)
        
        # Vectorized Score Calculation
        # Select submatrix: rows=candidates, cols=active_experts
        # Shape: (Num_Candidates, Num_Active)
        submatrix = self.similarity_matrix[candidate_indices][:, active_indices].clone()
        
        # --- Action 1: Normalize and Amplify Collaborative Score ---
        # Find the maximum non-diagonal value in the whole layer's similarity matrix 
        # to normalize the scores. (We use the whole matrix to get a stable base max).
        # Since we want non-diagonal MAX, we can temporarily set diagonal to 0
        assert self.similarity_matrix is not None
        temp_sim = self.similarity_matrix.clone()
        temp_sim.fill_diagonal_(0.0)
        max_off_diag = temp_sim.max().item()
        
        if max_off_diag > 1e-6:
            # Normalize the submatrix
            normalized_submatrix = submatrix / max_off_diag
        else:
            normalized_submatrix = submatrix
            
        # Sum over active experts to get Collaborative Score
        # Shape: (Num_Candidates,)
        collab_scores = normalized_submatrix.sum(dim=1)
        
        # We heavily weigh the collab_score with lambda_weight.
        # We could also add a Frequency/Temporal term here if we want (e.g., LFU/LRU base score)
        # But for now, we just let EBCO make the decision based on scaled collab_scores.
        final_scores = collab_scores * self.lambda_weight
        
        # Find index of minimum score (kick out the one with lowest collab correlation)
        min_score_idx = torch.argmin(final_scores).item()
        
        self.t_evict_search += time.perf_counter() - t0
        return candidates[min_score_idx]


class GraphExpertCache(ExpertCache):
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, lambda_weight: float = 1.0):
        super().__init__(make_module, main_size, offload_size, buffer_size)
        self.lambda_weight = lambda_weight
        # Override the default EvictionGroupInfo with our Graph-based one
        # The defaultdict lambda correctly initializes GraphEvictionGroupInfo with the cache's lambda_weight
        self.group_infos: Dict[int, GraphEvictionGroupInfo] = defaultdict(lambda: GraphEvictionGroupInfo(lambda_weight=self.lambda_weight))
        
        # Inter-Layer Graph State
        # Map layer_idx -> Transition Matrix (Num_Experts_Current, Num_Experts_Next)
        self.inter_layer_similarity: Dict[int, torch.Tensor] = {} 
        self.last_layer_context: Optional[Tuple[int, torch.Tensor]] = None # (layer_idx, probs)
        self.alpha_inter: float = 0.6 # EMA decay for inter-layer

    def update_inter_layer_graph(self, curr_layer_idx: int, curr_probs: torch.Tensor):
        """
        Updates the transition graph from (curr_layer_idx - 1) to curr_layer_idx.
        """
        # Move inputs to CPU
        with torch.no_grad():
            curr_probs_cpu = curr_probs.detach().cpu().float()
            
            if self.last_layer_context is not None:
                last_layer_idx, last_probs_cpu = self.last_layer_context
                
                # We only update if layers are sequential (e.g. 0->1, 5->6)
                # This handles the case where we might jump layers or reset (new batch)
                if last_layer_idx == curr_layer_idx - 1:
                    batch_size = curr_probs_cpu.shape[0]
                    num_experts_last = last_probs_cpu.shape[1]
                    num_experts_curr = curr_probs_cpu.shape[1]
                    
                    # Initialize transition matrix if needed
                    # Matrix Shape: (Num_Experts_Last, Num_Experts_Curr)
                    if last_layer_idx not in self.inter_layer_similarity:
                        self.inter_layer_similarity[last_layer_idx] = torch.zeros(
                            (num_experts_last, num_experts_curr), 
                            dtype=torch.float32, device='cpu'
                        )
                    
                    transition_matrix = self.inter_layer_similarity[last_layer_idx]
                    
                    # Compute correlation: Last_P^T @ Curr_P
                    # Shape: (N_last, B) @ (B, N_curr) -> (N_last, N_curr)
                    current_transition = torch.mm(last_probs_cpu.T, curr_probs_cpu)
                    current_transition = current_transition / batch_size
                    
                    # EMA Update
                    self.inter_layer_similarity[last_layer_idx] = (
                        self.alpha_inter * current_transition + 
                        (1 - self.alpha_inter) * transition_matrix
                    )

            # Update context for next call
            self.last_layer_context = (curr_layer_idx, curr_probs_cpu)

    def predict_next_layer(self, curr_layer_idx: int, curr_probs: torch.Tensor, k: int = 2) -> List[ExpertUID]:
        """
        Predicts top-k experts for the NEXT layer (curr_layer_idx + 1).
        returns: List of ExpertUIDs (layer_idx+1, expert_idx)
        """
        if curr_layer_idx not in self.inter_layer_similarity:
            return []
            
        with torch.no_grad():
            curr_probs_cpu = curr_probs.detach().cpu().float()
            transition_matrix = self.inter_layer_similarity[curr_layer_idx]
            
            # Predict scores for next layer
            # Score = Curr_P @ Transition_Matrix
            # Shape: (B, N_curr) @ (N_curr, N_next) -> (B, N_next)
            t0 = time.perf_counter()
            next_scores = torch.mm(curr_probs_cpu, transition_matrix)
            
            # Aggregate scores over batch (sum) to find generally most needed experts
            # Shape: (N_next,)
            total_scores = next_scores.sum(dim=0)
            
            # Top-K indices
            # If k is larger than num experts, clamp it
            k = min(k, total_scores.shape[0])
            topk_values, topk_indices = torch.topk(total_scores, k)
            
            next_layer_idx = curr_layer_idx + 1
            predicted_uids = [(next_layer_idx, idx.item()) for idx in topk_indices]
            # print(f"DEBUG: Layer {curr_layer_idx} Prediction took {(time.perf_counter() - t0)*1000:.2f}ms")
            return predicted_uids

    def load_experts(
            self, 
            *uids: ExpertUID, 
            unordered: bool = False, 
            gating_probs: Optional[torch.Tensor] = None,
            layer_idx: Optional[int] = None # Added layer_idx for context
        ) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        Loads experts with awareness of Gating Probabilities for Graph Update.
        
        Args:
            uids: List of ExpertUIDs to load (the 'active' experts for this batch).
            unordered: If True, yields in optimal order.
            gating_probs: Tensor of shape (Batch, Num_Experts).
        """
        if len(uids) == 0:
            return iter([])
            
        # Identify the eviction group (should be same for all uids)
        # We peek at the first uid to get the group
        # NOTE: self.registered_experts might not be populated if we haven't added them?
        # But `load_experts` assumes they are registered.
        
        first_info = self.registered_experts[uids[0]]
        group_id = first_info.eviction_group
        eviction_group = self.group_infos[group_id]
        
        # 1. Update Graph if probs provided
        if gating_probs is not None:
            # The defaultdict creates the group info if it doesn't exist.
            # We need to ensure it's created with the correct capacity (main_size)
            # and lambda_weight. The defaultdict lambda already handles this.
            eviction_group.update_graph(gating_probs)
            
            # Also update inter-layer graph if layer_idx is known
            # We can infer layer_idx from uids[0] which is (layer_idx, expert_idx)
            # Or use the passed argument
            curr_layer = uids[0][0]
            self.update_inter_layer_graph(curr_layer, gating_probs)

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
        
        # print(f"DEBUG: load_experts called with {len(uids)} uids")
        res = super().load_experts(*uids, unordered=unordered)
        if res is None:
            print("CRITICAL: super().load_experts returned None!")
            return iter([])
        return res
