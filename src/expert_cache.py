from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List
from collections import deque, defaultdict, OrderedDict
import time
from .expert_wrapper import MixtralExpertWrapper

import torch
from torch import nn

ExpertUID = Any


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    index: int


@dataclass
class EvictionGroupInfo:
    # infos in main and offload devices; ordered from least recently used to most
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)
    
    # Telemetry
    active_set_sizes: List[int] = field(default_factory=list)
    theoretical_hits: float = field(default=0.0)
    total_requested: int = field(default=0)
    
    # Base Profiling Timer
    t_evict_search: float = field(default=0.0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        t0 = time.perf_counter()
        for uid, info in self.main_infos.items():
            self.t_evict_search += time.perf_counter() - t0
            return info  # least recently used
        self.t_evict_search += time.perf_counter() - t0
        raise ValueError("No evictable experts")

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class ExpertCache:
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        self.main_modules = [self._check_module(make_module()) for i in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = [
            torch.UntypedStorage(self.module_size).pin_memory() for _ in range(offload_size)]
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        # temporary storage to shave off latency
        self.device_expert_buffers = deque([self._check_module(make_module()) for _ in range(buffer_size)])
        self.offloaded_storage_buffers = deque([
            torch.UntypedStorage(self.module_size).pin_memory() for _ in range(buffer_size)])
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)
        self.prefetch_stream = torch.cuda.Stream()
        
        self.timers = defaultdict(float)

    def _check_module(self, module: MixtralExpertWrapper):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def add_expert(self, uid: ExpertUID, module: MixtralExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=False, index=i)
                    self.registered_experts[uid] = self.main_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found spot on device
        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, index=i)
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False, gating_probs: Optional[torch.Tensor] = None) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        :example:
        >>> for uid, expert in expert_cache.load_experts(*list_of_uids, unordered=True):
        >>>     for uid, expert in expert_iter:
        >>>         result += expert(x) * get_moe_weight(uid)

        :param uids: iterate over the specified expert uids. Same uids as in add_expert
        :param unordered: if True, allows cache to iterate experts in arbitrary order
            The order is chosen to minimize the total wait time.
        :param gating_probs: (Optional) raw gating probabilities for graph-based strategies.
        :returns: an iterator that yields (uid, expert) pairs, only usable inside the for loop

        """
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:  # yield non-offloaded experts first
            try:
                uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
            except KeyError: # Handle case where uids might not be registered yet? Unlikely.
                 pass

        infos = [self.registered_experts[uid] for uid in uids]
        
        # --- Telemetry Recording ---
        active_set_size = len(infos)
        if active_set_size > 0:
             group_id_tmp = infos[0].eviction_group
             eg_tmp = self.group_infos[group_id_tmp]
             eg_tmp.active_set_sizes.append(active_set_size)
             eg_tmp.total_requested += active_set_size
             # Theoretical Upper Bound: (Assuming capacity is len(self.main_modules) // num_layers, but let's just 
             # use the actual len(main_infos) of this eviction group)
             capacity = len(eg_tmp.main_infos)
             eg_tmp.theoretical_hits += min(capacity, active_set_size)
        # ---------------------------

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        
        # Ensure any pending prefetches on these experts are done
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        
        # Base implementation ignores gating_probs, but subclasses use it.
        # We need to make sure we don't break existing logic.
        
        for info in infos:
            eviction_group.mark_used(info)

        try:
            self.active = True
            # save pre-loaded experts before they can be swapped
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.index] for info in pre_loaded_infos])

            # begin loading experts into free buffers in background (via non-blocking copy)
            infos_to_load_queue = deque([info for info in infos if info.offloaded])
            
            # --- Aggressive Prefetching Logic ---
            prefetch_candidates = []
            sim_matrix = getattr(eviction_group, 'similarity_matrix', None)
            if getattr(self, 'use_prefetch', False) and sim_matrix is not None:
                req_indices = [info.uid[1] for info in infos]
                for req_idx in req_indices:
                    try:
                        top_idx = torch.argmax(sim_matrix[req_idx]).item()
                        if top_idx != req_idx and top_idx not in req_indices:
                            uid = (infos[0].uid[0], top_idx)
                            p_info = self.registered_experts[uid]
                            if p_info.offloaded and p_info not in infos_to_load_queue and p_info not in prefetch_candidates:
                                prefetch_candidates.append(p_info)
                    except Exception:
                        pass
            # ------------------------------------
            
            infos_in_loading = deque([])
            experts_in_loading = deque([])
            bypassed_flags = deque([])
            
            def dispatch_load(info):
                is_prefetch = info not in infos
                if hasattr(eviction_group, 'admit') and not eviction_group.admit(info):
                    if is_prefetch:
                        return False # Don't prefetch bypassed experts
                    infos_in_loading.append(info)
                    experts_in_loading.append(self._stream_bypass(info))
                    bypassed_flags.append(True)
                    if hasattr(eviction_group, 'bypassed_count'): 
                        eviction_group.bypassed_count += 1
                else:
                    infos_in_loading.append(info)
                    experts_in_loading.append(self._swap(info, eviction_group.choose_expert_to_evict()))
                    bypassed_flags.append(False)
                return True

            window_size = min(len(self.device_expert_buffers), len(infos_to_load_queue))
            
            dispatched_count = 0
            while dispatched_count < window_size and len(infos_to_load_queue) > 0:
                info_to_dispatch = infos_to_load_queue.popleft()
                if dispatch_load(info_to_dispatch):
                    dispatched_count += 1

            for info in infos:
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                elif len(infos_in_loading) > 0 and info is infos_in_loading[0]:
                    infos_in_loading.popleft()
                    module = experts_in_loading.popleft()
                    is_bypassed = bypassed_flags.popleft()
                    
                    yield (info.uid, module)
                    
                    if is_bypassed:
                        # Return the streaming buffer back to the pool immediately!
                        self.device_expert_buffers.append(module)
                        
                    while len(infos_to_load_queue) > 0:
                        next_info = infos_to_load_queue.popleft()
                        if dispatch_load(next_info):
                            break
                else:
                    raise RuntimeError("internal error: caching algorithm failed")
                    
            # --- Launch remaining prefetches ---
            for p_info in prefetch_candidates:
                try:
                    if hasattr(eviction_group, 'admit') and not eviction_group.admit(p_info):
                        continue
                    self._swap(p_info, eviction_group.choose_expert_to_evict())
                except ValueError:
                    break
            # -----------------------------------
        finally:
            self.active = False

    def _stream_bypass(self, info_to_load: ExpertInfo) -> nn.Module:
        """Stream an offloaded expert without evicting a resident expert. (Admission Control Bypass)"""
        t0 = time.perf_counter()
        device_expert_buffer = self.device_expert_buffers.popleft()
        device_expert_buffer.storage.copy_(self.offloaded_storages[info_to_load.index], non_blocking=True)
        self.timers['t_stream_setup'] += time.perf_counter() - t0
        return device_expert_buffer

    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        """Swap an offloaded expert (info_to_load) with an on-device expert (info_to_evict) return the loaded expert"""
        t0 = time.perf_counter()
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        offloaded_storage_buffer = self.offloaded_storage_buffers.popleft()
        device_expert_buffer = self.device_expert_buffers.popleft()
        device_expert_buffer.storage.copy_(self.offloaded_storages[info_to_load.index], non_blocking=True)
        offloaded_storage_buffer.copy_(self.main_modules[info_to_evict.index].storage, non_blocking=True)

        self.device_expert_buffers.append(self.main_modules[info_to_evict.index])
        self.main_modules[info_to_evict.index] = device_expert_buffer
        self.offloaded_storage_buffers.append(self.offloaded_storages[info_to_load.index])
        self.offloaded_storages[info_to_load.index] = offloaded_storage_buffer

        self.main_infos[info_to_evict.index] = info_to_load
        self.offloaded_infos[info_to_load.index] = info_to_evict
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_evict.index, info_to_load.index = info_to_load.index, info_to_evict.index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
        self.timers['t_swap_setup'] += time.perf_counter() - t0
        return device_expert_buffer


    def prefetch_lookahead(self, uids: List[ExpertUID]):
        """
        Actively pre-load a list of experts into the background buffers. 
        Will silently fail if buffers are full or if experts are already loaded.
        """
        if not getattr(self, 'use_prefetch', False) or not uids:
            return
            
        t0 = time.perf_counter()
        prefetch_count = 0
        
        for uid in uids:
            if uid not in self.registered_experts:
                continue
                
            info_to_load = self.registered_experts[uid]
            
            # Only prefetch if offloaded and we have free buffers
            if info_to_load.offloaded and len(self.device_expert_buffers) > 0:
                eviction_group = self.group_infos[info_to_load.eviction_group]
                try:
                    # choose_expert_to_evict automatically skips locked/active experts
                    info_to_evict = eviction_group.choose_expert_to_evict()
                    self._swap(info_to_load, info_to_evict)
                    prefetch_count += 1
                except Exception:
                     # Silently ignore if we can't evict (e.g. all experts are active)
                     pass
        
        if 't_prefetch_lookahead' not in self.timers:
            self.timers['t_prefetch_lookahead'] = 0.0
        self.timers['t_prefetch_lookahead'] += time.perf_counter() - t0

    def prefetch_experts(self, uids: List[ExpertUID]):
        """
        Initiates non-blocking loading of future experts.
        Unlike load_experts, this does NOT yield modules. It just ensures they are moved to GPU.
        """
        if self.active:
            # If cache is currently active (loading something else), we skip prefetch 
            # to avoid buffer conflict or state corruption.
            return

        # Filter out experts that are already on device (offloaded=False)
        # We need to check existence first
        uids_to_fetch = []
        for uid in uids:
            if uid in self.registered_experts:
                info = self.registered_experts[uid]
                if info.offloaded:
                    uids_to_fetch.append(uid)
        
        if not uids_to_fetch:
            return

        # We assume they are in the same eviction group (usually next layer)
        # But if predictions cross layers (unlikely for now), handling single group is safer.
        first_info = self.registered_experts[uids_to_fetch[0]]
        eviction_group = self.group_infos[first_info.eviction_group]

        try:
            self.active = True
            
            # We use the same buffers as load_experts.
            # Logic: Swap needed offloaded experts with victims.
            
            infos_to_load = deque([self.registered_experts[uid] for uid in uids_to_fetch])
            
            # We can process as many as we have buffers? 
            # Actually we can do one by one or batch.
            # limit by buffer size to be safe
            window_size = min(len(self.device_expert_buffers), len(infos_to_load))
            
            for _ in range(window_size):
                info_to_load = infos_to_load.popleft()
                
                # Check again if it's still offloaded (race condition?)
                if not info_to_load.offloaded:
                    continue
                    
                # Choose victim
                victim_info = eviction_group.choose_expert_to_evict()
                
                # If victim is the same as what we want to load? Impossible if offloaded=True
                # If victim is currently used? choose_expert_to_evict logic should handle it?
                # Base choose_expert_to_evict just picks LRU.
                # Graph one picks score.
                
                # Perform swap
                # self._swap handles the copy and metadata update
                with torch.cuda.stream(self.prefetch_stream):
                    self._swap(info_to_load, victim_info)
                
        finally:
            self.active = False
