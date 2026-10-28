"""Explicit request-owned storage for the existing QSA output and reductions.

A workspace binds to one exact query/index shape on first use. Context length
is deliberately not a key: it neither grows storage nor creates cached entries.
Create a fresh owner token (for example ``object()``) for each request and pass
that same token on every use. Never share the token/workspace across requests.

The returned output is borrowed until the next use or close of this workspace.
Consume it first, or clone it if it must survive. Calls must run on one stream
or explicitly ordered streams. This is buffer ownership, not a CUDA Graph or
an asynchronous lifetime manager; close does not synchronize a device.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class QSAWorkspaceShape:
    rows: int
    index_width: int
    splits: int
    device: torch.device
    dtype: torch.dtype

    @property
    def nbytes(self):
        return self.rows * 24 * (256 * 2 + self.splits * (256 * 4 + 4))


def _tensor(value, name, shape, dtype, device):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f'{name} must be a tensor')
    if value.layout != torch.strided or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} has an invalid shape/layout')
    if value.dtype != dtype or value.device != device or not value.is_contiguous():
        raise ValueError(f'{name} has an invalid dtype/device/contiguity')


class QSAWorkspace:
    """One fixed shape, three buffers, one request; no global pool or resize.

    ``buffers`` may be called before attention to prepare storage outside graph
    capture. CPU tensors are supported for ownership/metadata tests only; the
    actual attention kernels still require the existing supported CUDA path.
    """
    def __init__(self, owner, *, max_bytes=128 * 1024**2):
        if owner is None:
            raise ValueError('A request owner token is required')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('Workspace byte limit must be a positive integer')
        self._owner = owner
        self._max_bytes = max_bytes
        self._shape = None
        self._buffers = None
        self._pointers = None
        self._closed = False

    @property
    def shape(self):
        return self._shape

    @property
    def nbytes(self):
        return self._shape.nbytes if self._buffers is not None else 0

    @property
    def max_bytes(self):
        return self._max_bytes

    def _check_owner(self, owner):
        if self._closed:
            raise RuntimeError('QSA workspace is closed')
        if owner is not self._owner:
            raise ValueError('QSA workspace belongs to a different request')

    def _check_query_indices(self, query, indices, owner):
        self._check_owner(owner)
        if not isinstance(query, torch.Tensor) or query.ndim != 3 or query.shape[0] <= 0:
            raise ValueError('Expected nonempty QSA query [rows,24,256]')
        _tensor(query, 'query', (query.shape[0], 24, 256), torch.bfloat16, query.device)
        if (not isinstance(indices, torch.Tensor) or indices.ndim != 2
                or indices.shape[0] != query.shape[0] or indices.shape[1] <= 0
                or indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Expected nonempty integer QSA indices [rows,width]')
        _tensor(indices, 'indices', indices.shape, indices.dtype, query.device)

    def validate_inputs(self, query, packed, indices, *, owner):
        """Validate metadata without reading device values or synchronizing."""
        from .kv_cache import PagedKV
        self._check_query_indices(query, indices, owner)
        if isinstance(packed, PagedKV):
            block = packed.block_size
            if type(block) is not int or block < 16 or block > 1024 or block & (block - 1):
                raise ValueError('Invalid QSA KV page size')
            if type(packed.length) is not int or packed.length <= 0:
                raise ValueError('Expected nonempty QSA KV history')
            count = (packed.length + block - 1) // block
            if len(packed._pages) != count:
                raise ValueError('QSA KV page capacity differs from history length')
            _tensor(packed.block_table, 'KV page table', (count,), torch.int64, query.device)
            for page in packed._pages:
                _tensor(page, 'KV page', (block, 2, 288), torch.uint8, query.device)
            kv_inputs = (*packed._pages, packed.block_table)
        else:
            if not isinstance(packed, torch.Tensor) or packed.ndim != 3 or packed.shape[0] <= 0:
                raise ValueError('Expected nonempty QSA NVFP4 history [tokens,2,288]')
            _tensor(packed, 'KV history', (packed.shape[0], 2, 288), torch.uint8, query.device)
            kv_inputs = (packed,)
        if self._buffers is not None:
            owned = {value.untyped_storage().data_ptr() for value in self._buffers}
            if any(value.untyped_storage().data_ptr() in owned for value in kv_inputs):
                raise ValueError('QSA KV history/table must not alias workspace storage')

    def buffers(self, query, indices, *, splits, owner):
        """Return ``(out, partial, lse)``; later calls overwrite borrowed data."""
        self._check_query_indices(query, indices, owner)
        if type(splits) is not int or not 1 <= splits <= 64 or splits & (splits - 1):
            raise ValueError('Expected a QSA power-of-two split count in [1,64]')
        shape = QSAWorkspaceShape(query.shape[0], indices.shape[1], splits, query.device, query.dtype)
        if self._shape is not None and shape != self._shape:
            raise ValueError('QSA workspace shape is fixed; create a separate workspace')
        if shape.nbytes > self._max_bytes:
            raise ValueError('QSA workspace exceeds its byte limit')
        expected = ((shape.rows, 24, 256), (shape.rows, 24, splits, 256), (shape.rows, 24, splits))
        dtypes = (torch.bfloat16, torch.float32, torch.float32)
        if self._buffers is None:
            if query.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Prepare QSA workspace before CUDA Graph capture')
            # Publish only after every allocation succeeds. No old shape or
            # partial allocation is retained on a failed preparation.
            buffers = tuple(torch.empty(size, device=query.device, dtype=dtype)
                            for size, dtype in zip(expected, dtypes))
            self._buffers = buffers
            self._shape = shape
            self._pointers = tuple(value.data_ptr() for value in buffers)
        for name, value, size, dtype, pointer in zip(
                ('output', 'partial', 'LSE'), self._buffers, expected, dtypes, self._pointers):
            _tensor(value, name, size, dtype, query.device)
            if value.storage_offset() != 0 or value.untyped_storage().nbytes() != value.numel()*value.element_size():
                raise ValueError(f'QSA {name} storage capacity was changed')
            if value.data_ptr() != pointer:
                raise ValueError(f'QSA {name} storage was replaced')
            if value.untyped_storage().data_ptr() in (query.untyped_storage().data_ptr(),
                                                     indices.untyped_storage().data_ptr()):
                raise ValueError(f'QSA {name} must not alias an input')
        return self._buffers

    def close(self, *, owner):
        """Drop owned references after the caller has finished GPU work."""
        self._check_owner(owner)
        self._buffers = self._pointers = None
        self._closed = True
