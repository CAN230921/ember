"""Fixed GDN prefill arithmetic for this GB10 checkpoint.

The generic autotuner omits sequence length from several keys. Starting with
a short prompt can select different K tiles from a long first prompt, which
changes BF16 intermediates and eventually the generated text. These profiles
reproduce all seven real 12.5K prefill logit/last-hidden captures across loads.
Decode kernels and the four-warp 64x64 inverse contract remain unchanged.
"""
from functools import lru_cache
import importlib
import torch


PROFILES = {
    'cumsum.chunk_local_cumsum_scalar_kernel': ({}, 2, 3),
    'chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel': ({'BK': 128}, 2, 2),
    'wy_fast.recompute_w_u_fwd_kernel': ({}, 2, 3),
    'chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64': ({'BV': 32}, 2, 2),
    'chunk_o.chunk_fwd_kernel_o': ({'BK': 128, 'BV': 64}, 4, 3),
}


@lru_cache(maxsize=1)
def configure(device_index):
    if torch.cuda.get_device_capability(device_index) != (12, 1):
        raise ValueError('Fixed GDN prefill profiles require sm_121')
    from triton.runtime.autotuner import Autotuner
    pending = []
    for qualified, (kwargs, warps, stages) in PROFILES.items():
        module, name = qualified.rsplit('.', 1)
        kernel = getattr(importlib.import_module('engine.vendor.fla.' + module), name)
        for _ in range(5):
            if isinstance(kernel, Autotuner):
                break
            kernel = getattr(kernel, 'fn', None)
        if not isinstance(kernel, Autotuner):
            raise RuntimeError('Expected pinned-image FLA autotuner: ' + qualified)
        choices = [config for config in kernel.configs
                   if config.kwargs == kwargs and config.num_warps == warps
                   and config.num_stages == stages and config.num_ctas == 1]
        if len(choices) != 1:
            raise RuntimeError('Verified GDN profile unavailable: ' + qualified)
        pending.append((kernel, choices))
    # Validate the full profile before modifying any of the five tuners.
    for kernel, choices in pending:
        kernel.cache.clear()
        kernel.configs = choices
    print('SM121_GDN_PREFILL_PROFILE_APPLIED', len(pending), flush=True)
