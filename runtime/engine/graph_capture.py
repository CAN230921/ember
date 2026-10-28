"""CUDA graph capture helper: give every captured graph its own cuBLAS workspace.

cuBLAS keeps one persistent workspace per (handle, stream). Our captures warm up eagerly on a side
stream first, so without a reset the capture bakes in a workspace allocated in the ordinary pool and
shared with whatever else later runs on that (pooled) stream; graphs of different owners then alias
or outlive that block (illegal address on GB10 once per-request DFlash captures interleaved with
multi-lane verification graphs, 2026-10-21). Clearing the cached workspaces right before capture makes
cuBLAS allocate a fresh one inside the graph's private pool during capture; it stays referenced by the
workspace cache, so later captures into the same pool cannot reuse it. Only the workspace address
changes (same size, same algorithms), so captured arithmetic is unchanged.
"""
import torch


def graph(cuda_graph, **kwargs):
    """Drop-in for torch.cuda.graph(cuda_graph, **kwargs)."""
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    return torch.cuda.graph(cuda_graph, **kwargs)


# Opt-in single private pool for every decode/verify/batch/DFlash graph (instead of one pool per shape).
# Safe only when no graph output is read after another graph of the pool has replayed: the continuous
# runners clone (or materialize) committed states right after each replay and consume pending/taps
# immediately; graph inputs and persistent caches are allocated outside capture. The borrowed
# verification mode (engine/speculative.py) keeps prefix outputs across replays and refuses this pool.
# Measured 2026-10-21: nine per-shape pools held 8.35 GB with 5.7 GB free (fragmented temporaries).
SHARED_POOL = __import__('os').environ.get('SPARK_SHARED_GRAPH_POOL', '0') == '1'
_shared = []


def shared_pool():
    """The process-wide graph pool when SPARK_SHARED_GRAPH_POOL=1, else None."""
    if not SHARED_POOL:
        return None
    if not _shared:
        _shared.append(torch.cuda.graph_pool_handle())
    return _shared[0]
