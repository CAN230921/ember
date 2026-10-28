"""Opt-in NVFP4 W4A16 projections (E2M1 codes, FP8-E4M3 per-16 scales, FP32 global scale).

Weights come from the offline GPTQ files (code2 = 2*E2M1 value as int8, scale as FP8 [N, K/16],
global_scale). They are packed into FlashInfer's canonical layout (two codes per byte, low
nibble = even K, 128x4-swizzled scales) and run with mm_bf16_fp4(backend='cute-dsl'):
out = x @ (code * scale * global_scale).T with BF16 activations (no activation quantization).
This changes numerics versus the FP8 path and requires quality acceptance before use.
"""
import torch

_MAG_INDEX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 8: 6, 12: 7}  # |2*value| -> E2M1 magnitude code


def pack_codes(code2):
    """int8 [N, K] holding 2*E2M1 values -> uint8 [N, K//2] E2M1 nibbles (low nibble = even K)."""
    lut = torch.full((13,), 255, dtype=torch.uint8, device=code2.device)
    for mag, idx in _MAG_INDEX.items():
        lut[mag] = idx
    mag = lut[code2.abs().long()]
    if (mag == 255).any():
        raise ValueError('code outside the E2M1 grid')
    nib = mag | ((code2 < 0).to(torch.uint8) << 3)
    return nib[:, 0::2] | (nib[:, 1::2] << 4)


def dequantize(code2, scale, global_scale):
    n, k = code2.shape
    return (code2.float()/2).view(n, k//16, 16)*scale.float()[..., None]*global_scale


class NVFP4Linear:
    def __init__(self, code2, scale, global_scale, device='cuda', lr_L=None, lr_R=None):
        import flashinfer
        code2 = code2.to(device); scale = scale.to(device)
        n, k = code2.shape
        if scale.shape != (n, k//16) or scale.dtype != torch.float8_e4m3fn or code2.dtype != torch.int8:
            raise ValueError('Expected int8 code2 [N,K] and FP8 scales [N,K/16]')
        packed = pack_codes(code2)
        swizzled = flashinfer.nvfp4_block_scale_interleave(scale.view(torch.uint8).contiguous())
        alpha = torch.tensor([global_scale], dtype=torch.float32, device=device)
        self.b, self.sf, self.alpha = flashinfer.prepare_bf16_fp4_weights(packed, swizzled, alpha, backend='cute-dsl')
        self.shape = (n, k)
        self._mm = flashinfer.mm_bf16_fp4
        # Optional BF16 low-rank compensation: y += (x @ R.T) @ L.T with L [N,r], R [r,K].
        self.lr_L = None if lr_L is None else lr_L.to(device=device, dtype=torch.bfloat16).contiguous()
        self.lr_R = None if lr_R is None else lr_R.to(device=device, dtype=torch.bfloat16).contiguous()

    def __call__(self, x):
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.shape[1]:
            raise ValueError('NVFP4 projection expects BF16 [M, K] input')
        x = x.contiguous()
        y = self._mm(x, self.b, self.sf, self.alpha, backend='cute-dsl', out_dtype=torch.bfloat16)
        if self.lr_L is not None:
            y = y + torch.nn.functional.linear(torch.nn.functional.linear(x, self.lr_R), self.lr_L)
        return y


# Which GDN projections use NVFP4 when attached: subset of {'qkvz', 'out'}; set at runtime.
ACTIVE = set()


def attach(engine, directory, projections=('qkvz', 'out')):
    """Attach offline NVFP4 weights to every GDN layer (FP8 weights stay for A/B)."""
    from pathlib import Path
    count = 0
    for index, layer in enumerate(engine.layers):
        if layer.layer_type != 'linear_attention':
            continue
        layer.attention.nvfp4 = {}
        for name in projections:
            data = torch.load(Path(directory)/f'layer{index:02d}_{name}.pt')
            layer.attention.nvfp4[name] = NVFP4Linear(data['code2'], data['scale'], data['global_scale'],
                                                      lr_L=data.get('lr_L'), lr_R=data.get('lr_R'))
            count += 1
    return count
