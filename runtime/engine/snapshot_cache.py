"""Storage accounting for bounded, privately cloned MTP checkpoints.

Reservation covers retained checkpoint storage and the pending snapshot. It
does not replace the serving memory guard or account for active model states.
Ordinary tensors are cloned by object identity, whereas PagedKV pages remain
immutable shared storage and only their pointer tables are cloned.
"""
from dataclasses import dataclass, field, is_dataclass
from types import SimpleNamespace

import torch

from .ops.kv_cache import PagedKV
from .prefix_cache import Checkpoint


@dataclass
class SnapshotCheckpoint(Checkpoint):
    chunk_size: int = 0
    last_used: int = 0
    storages: dict = field(default_factory=dict)


def storage_map(value, result=None):
    """Return storage identities and sizes, without retaining any tensors."""
    result = {} if result is None else result
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        result[(value.device, storage.data_ptr())] = storage.nbytes()
    elif isinstance(value, PagedKV):
        storage_map(value._pages, result)
        storage_map(value.block_table, result)
    elif isinstance(value, (list, tuple)):
        for item in value:
            storage_map(item, result)
    elif isinstance(value, dict):
        for item in value.values():
            storage_map(item, result)
    elif is_dataclass(value) or isinstance(value, SimpleNamespace):
        storage_map(vars(value), result)
    return result


def snapshot_requirements(target, draft):
    """Model the existing separate clone_state(target)/clone_state(draft).

    Distinct tensor views require distinct copies even when their source
    storage is shared. Repeated references to the same object only copy once
    per clone call. Torch clone makes non-dense views contiguous, and preserves
    the layout of dense views; both consume numel * element_size bytes.
    """
    shared = {}

    def copied_bytes(value, seen):
        if value is None or isinstance(value, (str, int, float, bool)):
            return 0
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, PagedKV):
            storage_map(value._pages, shared)
            # clone_state invokes table.clone() directly for each PagedKV.
            return value.block_table.numel() * value.block_table.element_size()
        if isinstance(value, (list, tuple)):
            return sum(copied_bytes(item, seen) for item in value)
        if isinstance(value, dict):
            return sum(copied_bytes(item, seen) for item in value.values())
        if is_dataclass(value) or isinstance(value, SimpleNamespace):
            return copied_bytes(vars(value), seen)
        raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')

    copied = copied_bytes(target, set()) + copied_bytes(draft, set())
    return shared, copied


def reserved_bytes(storage_maps, shared, copied):
    """Retained storage union plus immutable pages and pending tensor copies."""
    union = dict(shared)
    for storages in storage_maps:
        union.update(storages)
    return sum(union.values()) + copied
