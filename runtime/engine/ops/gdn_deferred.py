"""Opt-in linear GDN verification without per-token recurrent snapshots.

The existing kernels retain their BF16 products, beta, core and inter-token
state boundaries. Verification saves convolved inputs and small convolution
prefixes; commit replays only the accepted recurrent updates. No projections,
expert calls or model layers are rerun. This module does not enable itself.
"""
from dataclasses import dataclass, field

import torch

from .gdn import GDNState
from ..limits import VERIFY_WIDTHS


def _versions(state):
    def version(tensor):
        try:
            return tensor._version
        except RuntimeError:  # inference_mode tensors have no version counter
            return None
    return tuple(version(x) for x in (state.conv, state.recurrent))


@dataclass
class DeferredGDN:
    output: torch.Tensor
    model: object
    initial: GDNState
    conv_prefixes: torch.Tensor
    convolved: torch.Tensor
    ba: torch.Tensor
    _status: str = field(default='open', init=False)
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self._versions = _versions(self.initial)

    @property
    def rows(self):
        return self.convolved.shape[0]

    @property
    def retained_bytes(self):
        # Model weights, base state and the returned output are excluded.
        return sum(x.numel()*x.element_size() for x in
                   (self.conv_prefixes, self.convolved, self.ba))

    def validate(self, count):
        if self._status != 'open':
            raise RuntimeError('Deferred GDN transaction is closed')
        if type(count) is not int or not 1 <= count <= self.rows:
            raise ValueError('Expected a nonempty accepted input prefix')
        if self._versions != _versions(self.initial):
            raise RuntimeError('Deferred GDN base state changed')

    def commit(self, count):
        self.validate(count)
        from .gdn_triton import _decode
        recurrent = torch.empty_like(self.initial.recurrent)
        # Keep output arithmetic live during replay to preserve the original
        # kernel's reduction/layout contract. Its small result is discarded.
        scratch = self.output.new_empty(count, 6144)
        b, a = self.ba.split((48, 48), -1)
        _decode[(4, 48)](self.convolved, a, b, self.model.A_log,
            self.model.dt_bias, self.initial.recurrent, recurrent, scratch,
            count, a.stride(0), b.stride(0), num_warps=1, num_stages=3)
        state = GDNState(self.conv_prefixes[count-1].clone(), recurrent)
        self._status = 'committed'
        self._release()
        return state

    def discard(self):
        if self._status != 'open':
            raise RuntimeError('Deferred GDN transaction is closed')
        self._status = 'discarded'
        self._release()

    def _release(self):
        self.model = self.initial = self.conv_prefixes = self.convolved = self.ba = None


def verify_gdn(model, qkvz, ba, state):
    """Inputs/base state are read-only until commit or discard on this stream."""
    if (not qkvz.is_cuda or qkvz.dtype != torch.bfloat16 or
            qkvz.ndim != 2 or qkvz.shape[1] != 16384 or
            qkvz.shape[0] not in VERIFY_WIDTHS or ba.shape != (len(qkvz), 96) or
            ba.dtype != qkvz.dtype or ba.device != qkvz.device):
        raise ValueError('Deferred GDN requires checkpoint CUDA BF16 verification rows')
    if (state.conv.shape != (1, 10240, 3) or
            state.recurrent.shape != (1, 48, 128, 128) or
            any(x.dtype != qkvz.dtype or x.device != qkvz.device or not x.is_contiguous()
                for x in (state.conv, state.recurrent))):
        raise ValueError('Deferred GDN requires private checkpoint BF16 state')
    from .gdn_triton import _conv, _decode, _gated_norm
    rows = len(qkvz)
    convolved = qkvz.new_empty(rows, 10240)
    conv_prefixes = qkvz.new_empty(rows, *state.conv.shape)
    _conv[(40,)](qkvz, state.conv, model.conv, convolved, conv_prefixes[-1],
        qkvz.stride(0), 10240, rows, 256, True, conv_prefixes, True)
    b, a = ba.split((48, 48), -1)
    core = qkvz.new_empty(rows, 6144)
    _decode[(4, 48)](convolved, a, b, model.A_log, model.dt_bias,
        state.recurrent, None, core, rows, a.stride(0), b.stride(0),
        STORE_FINAL=False, num_warps=1, num_stages=3)
    normalized = torch.empty_like(core)
    z = qkvz[:, 10240:]
    _gated_norm[(rows*48,)](core, z, model.norm, normalized, z.stride(0),
        model.eps, model.gate_activation == 'sigmoid', rows*48, 1, num_warps=1)
    return DeferredGDN(normalized, model, state, conv_prefixes, convolved, ba)
