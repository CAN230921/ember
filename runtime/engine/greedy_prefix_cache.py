"""Target-only checkpoints, deliberately separate from shifted MTP checkpoints.

One bounded published chain is shared. Active sessions retain only their own
checkout, never an old chain of cache entries. Completed chunk checkpoints are
published immediately, so cancellation cannot publish an incomplete state.
"""
from dataclasses import dataclass
from .prefix_cache import clone_state, tensor_bytes


@dataclass
class TargetCheckpoint:
    key: tuple
    end: int
    target: object


class GreedyPrefixCache:
    def __init__(self, max_bytes=1024**3):
        self.max_bytes = max_bytes
        self.entries = []
        self.chunk_size = None
        self.owner = None

    def checkout(self, ids, chunk_size):
        if chunk_size != self.chunk_size:
            return None, 0
        for entry in reversed(self.entries):
            if entry.end < len(ids) and tuple(ids[:entry.end]) == entry.key:
                return clone_state(entry.target), entry.end
        return None, 0

    def checkpoint(self, ids, end, target, chunk_size, owner=None):
        if not 0 < end < len(ids) or end % chunk_size or target.tokens != list(ids[:end]):
            raise ValueError('Expected a complete target prefill chunk checkpoint')
        # One active writer prevents unrelated prompts evicting each other
        # after every 2048-token chunk. Other sessions may still check out.
        if self.owner is not None and self.owner is not owner:
            return False
        size = tensor_bytes(target)
        if size > self.max_bytes:
            return False
        self.owner = owner
        key = tuple(ids[:end])
        if self.chunk_size != chunk_size or any(
                entry.key != key[:entry.end] for entry in self.entries if entry.end <= end):
            self.clear()
        # A previously published identical boundary needs no second snapshot.
        if any(entry.key == key for entry in self.entries):
            return True
        self.entries = [e for e in self.entries if e.end < end]
        # Evict before copying, including the temporary copy in the budget.
        while self.entries and self.bytes + size > self.max_bytes:
            self.entries.pop(1 if len(self.entries) > 1 else 0)
        self.entries.append(TargetCheckpoint(key, end, clone_state(target)))
        self.chunk_size = chunk_size
        return True

    def clear(self):
        self.entries = []
        self.chunk_size = None

    def release(self, owner):
        if self.owner is owner:
            self.owner = None

    @property
    def bytes(self):
        return tensor_bytes(self.entries)
