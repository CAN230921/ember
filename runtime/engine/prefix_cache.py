"""Bounded checkpoints for the previous prompt, including shifted MTP keys.

Only completed prefill chunk boundaries are cached. A draft checkpoint at N
has consumed the prompt's token N as lookahead, so its key includes N+1 IDs.
GDN recurrence cannot be sliced back to an arbitrary common-prefix length.
Snapshots and checkout copies own their mutable tensors independently.
"""
import copy
from dataclasses import dataclass,is_dataclass
from types import SimpleNamespace
import torch
from .ops.kv_cache import PagedKV


def clone_state(value,memo=None):
    memo={} if memo is None else memo
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if id(value) in memo:return memo[id(value)]
    if isinstance(value,torch.Tensor):result=value.clone()
    elif isinstance(value,PagedKV):
        # KV pages are immutable by contract; open pages use copy-on-write.
        result=PagedKV(value._pages,value.block_table.clone(),value.length,value.block_size)
    elif isinstance(value,list):result=[clone_state(item,memo) for item in value]
    elif isinstance(value,tuple):result=tuple(clone_state(item,memo) for item in value)
    elif isinstance(value,dict):result={key:clone_state(item,memo) for key,item in value.items()}
    elif is_dataclass(value) or isinstance(value,SimpleNamespace):
        result=copy.copy(value)
        for key,item in vars(value).items():setattr(result,key,clone_state(item,memo))
    else:raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')
    memo[id(value)]=result
    return result


def tensor_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if isinstance(value,torch.Tensor):
        storage=value.untyped_storage(); key=(value.device,storage.data_ptr())
        if key in seen:return 0
        seen.add(key); return storage.nbytes()
    if isinstance(value,PagedKV):return tensor_bytes(value._pages,seen)+tensor_bytes(value.block_table,seen)
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(item,seen) for item in value)
    if isinstance(value,dict):return sum(tensor_bytes(item,seen) for item in value.values())
    if is_dataclass(value) or isinstance(value,SimpleNamespace):return tensor_bytes(vars(value),seen)
    return 0


@dataclass
class Checkpoint:
    key: tuple
    end: int
    target: object
    draft: object


class PrefixCache:
    def __init__(self,max_bytes=1024**3):
        self.max_bytes=max_bytes; self.entries=[]; self.chunk_size=None

    def checkout(self,ids,chunk_size):
        if chunk_size!=self.chunk_size:return None,None,0,[]
        matched=[]
        for entry in self.entries:
            if entry.end>=len(ids) or tuple(ids[:entry.end+1])!=entry.key:break
            matched.append(entry)
        if not matched:return None,None,0,[]
        entry=matched[-1]
        return clone_state(entry.target),clone_state(entry.draft),entry.end,list(matched)

    def checkpoint(self,entries,ids,end,target,draft):
        if end>=len(ids) or draft is None:raise ValueError('MTP checkpoint requires a known lookahead token.')
        entries.append(Checkpoint(tuple(ids[:end+1]),end,clone_state(target),clone_state(draft)))
        # Retain the earliest stable boundary plus the newest checkpoints.
        while tensor_bytes(entries)>self.max_bytes:
            if len(entries)==1:entries.clear(); break
            entries.pop(1 if len(entries)>2 else 0)

    def commit(self,entries,chunk_size):
        # A short title/tool-result request must not evict the long prompt.
        if entries:self.entries=entries;self.chunk_size=chunk_size

    def clear(self):self.entries=[];self.chunk_size=None

    @property
    def bytes(self):return tensor_bytes(self.entries)
