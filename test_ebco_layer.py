import sys
import os
import torch
import unittest
from unittest.mock import MagicMock

# Adjust path to include src
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mocking modules before import if necessary, but we can just use the classes
# We need to make sure torch is available (it is)

# Import the class under test
# Note: we might need to mock internal imports if they fail
from src.graph_expert_cache import GraphExpertCache, GraphEvictionGroupInfo, ExpertInfo

# Mock MixtralExpertWrapper for type checks
class MockExpert:
    def __init__(self, device='cpu'):
        # The cache expects torch.UntypedStorage
        # In recent PyTorch, UntypedStorage is the base for typed storages.
        # We can construct it directly.
        self.storage = torch.UntypedStorage(1024)
        if device == 'cuda' and torch.cuda.is_available():
             self.storage = self.storage.cuda()
        # Mock pin_memory support on the storage object if needed, 
        # but UntypedStorage has pin_memory.
        
        self.device = device

class TestEBCO(unittest.TestCase):
    def setUp(self):
        # Setup Cache
        # Main size 2, Offload size 2
        self.cache = GraphExpertCache(
            make_module=lambda: MockExpert(),
            main_size=2,
            offload_size=2,
            buffer_size=1
        )
        
        # Add 4 experts (0, 1, 2, 3) all in layer 0
        for i in range(4):
            expert = MockExpert()
            # We need to manually set offload state because add_expert logic is strict
            # First 2 go to main, next 2 to offload
            offload = True if i >= 2 else False
            self.cache.add_expert(uid=(0, i), module=expert, eviction_group=0, offload=offload)
            
    def test_graph_update(self):
        eviction_group = self.cache.group_infos[0]
        
        # Simulate batch where Expert 0 and 1 are highly correlated
        # Batch size 2. 
        # Item 0: Experts 0, 1 active
        # Item 1: Experts 0, 1 active
        probs = torch.zeros(2, 4)
        probs[0, 0] = 0.9
        probs[0, 1] = 0.8
        probs[1, 0] = 0.8
        probs[1, 1] = 0.9
        
        eviction_group.update_graph(probs)
        
        W = eviction_group.similarity_matrix
        print(f"Similarity Matrix after update:\n{W}")
        
        # Check self similarity
        self.assertAlmostEqual(W[0, 0].item(), 1.0)
        
        # Check 0-1 similarity (should be high)
        sim_0_1 = W[0, 1].item()
        self.assertGreater(sim_0_1, 0.0)
        
        # Check 0-2 similarity (should be 0 or very low/negative depending on initialization)
        sim_0_2 = W[0, 2].item()
        # Since initialized as Identity * (1-alpha) + Update * alpha
        # And update had 0 correlation for 0-2 (dot product of [0.9, 0.8] and [0,0] is 0)
        # It should be 0.
        self.assertLess(sim_0_2, 0.1)

    def test_eviction_logic(self):
        # Current Layout:
        # Main (GPU): [Expert 0, Expert 1]
        # Offload (CPU): [Expert 2, Expert 3]
        
        eviction_group = self.cache.group_infos[0]
        
        # 1. Establish strong correlation between Expert 0 and Expert 2
        # And weak correlation between Expert 1 and Expert 2
        # We manually set W for predictability
        eviction_group.similarity_matrix = torch.eye(4)
        eviction_group.similarity_matrix[0, 2] = 0.9 # Expert 0 and 2 are friends
        eviction_group.similarity_matrix[2, 0] = 0.9
        
        eviction_group.similarity_matrix[1, 2] = 0.1 # Expert 1 and 2 are not friends
        eviction_group.similarity_matrix[2, 1] = 0.1
        
        # 2. Request to load Expert 2.
        # Active set for this request is [2].
        # We need to evict someone from Main ([0, 1]) to make room for 2.
        # Score(0) = Sim(0, 2) = 0.9
        # Score(1) = Sim(1, 2) = 0.1
        # The logic should evict the one with LOWEST score -> Expert 1.
        
        print("Initial Main:", self.cache.main_infos)
        
        # Execute load_experts
        # We pass uids=[(0, 2)]
        # We also pass dummy probs just to satisfy interface, but current active set 
        # will be derived from uids.
        
        # Note: load_experts calls `choose_expert_to_evict`.
        # `choose_expert_to_evict` uses `self.current_active_experts`.
        # `load_experts` sets `current_active_experts` to {(0, 2)}.
        
        gen = self.cache.load_experts((0, 2), gating_probs=torch.zeros(1, 4))
        for _ in gen: pass # Consume iterator
        
        # Check who is in main now
        # Expert 2 should be in main.
        # Expert 0 should be in main (high score).
        # Expert 1 should be evicted (low score).
        
        main_uids = [info.uid for info in self.cache.main_infos if info is not None]
        print("Final Main UIDs:", main_uids)
        
        self.assertIn((0, 2), main_uids)
        self.assertIn((0, 0), main_uids)
        self.assertNotIn((0, 1), main_uids)

if __name__ == '__main__':
    unittest.main()
