"""Vectorized PLE convolution vs the per-row loop it replaces.

CPU check of shapes, state hand-off, row counts and mode switch. Bitwise
equality on CUDA (where torch's FP32 reduction order is the one that matters)
is proven by diagnostics/verify_ple_conv_vector.py on the GPU host, not here.
"""
import unittest

import torch
from torch.nn import functional as F

from engine.ops.ple import PLERef

CHANNELS, TAPS, DILATION = 640, 4, 3
STATE = (TAPS-1)*DILATION


def make_ple(seed=0):
    generator = torch.Generator().manual_seed(seed)
    ple = PLERef.__new__(PLERef)
    torch.nn.Module.__init__(ple)
    ple.dilation, ple.state_length = DILATION, STATE
    conv = torch.randn(CHANNELS, 1, TAPS, generator=generator).to(torch.bfloat16)
    ple.register_buffer('conv', conv)
    return ple


def loop_reference(ple, normalized, state, dtype=torch.bfloat16):
    """Verbatim copy of the replaced loop (rows > 3, no prefixes)."""
    outputs = []
    weight = ple.conv.squeeze(1).float()
    for row in normalized:
        history = torch.cat((state, row[None, :, None]), dim=-1)
        sampled = history[:, :, ::ple.dilation]
        conv = (sampled.float()*weight[None]).sum(-1).to(dtype)
        outputs.append(F.silu(conv))
        state = history[:, :, 1:].contiguous()
    return torch.cat(outputs), state


class VectorConvCase(unittest.TestCase):
    def inputs(self, rows, seed, zero_state=False):
        generator = torch.Generator().manual_seed(seed)
        normalized = torch.randn(rows, CHANNELS, generator=generator).to(torch.bfloat16)
        state = (torch.zeros if zero_state else torch.randn)(
            1, CHANNELS, STATE, **({} if zero_state else dict(generator=generator))).to(torch.bfloat16)
        return normalized, state

    def test_matches_loop_values_state_shape_and_layout(self):
        ple = make_ple()
        weight = ple.conv.squeeze(1).float()
        for rows in (4, 5, 8, 9, 63, 64, 130):
            for seed, zero in ((1, False), (2, True)):
                normalized, state = self.inputs(rows, seed, zero)
                expected, expected_state = loop_reference(ple, normalized, state)
                actual, actual_state = ple.conv_rows(normalized, state, weight, torch.bfloat16)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                # CPU reduction order may differ in the last FP32 bit; bf16 rounding hides it
                # except at rare ties. Require near-total agreement here, exactness on CUDA.
                mismatch = (actual != expected).float().mean().item()
                self.assertLess(mismatch, 1e-4, (rows, seed, mismatch))
                torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
                self.assertTrue(torch.equal(actual_state, expected_state))   # pure data movement
                self.assertEqual(actual_state.shape, (1, CHANNELS, STATE))
                self.assertTrue(actual_state.is_contiguous())

    def test_state_chaining_across_chunks_equals_one_pass(self):
        ple = make_ple(3)
        weight = ple.conv.squeeze(1).float()
        normalized, state = self.inputs(96, 7)
        whole, whole_state = ple.conv_rows(normalized, state, weight, torch.bfloat16)
        first, mid = ple.conv_rows(normalized[:40], state, weight, torch.bfloat16)
        second, last = ple.conv_rows(normalized[40:], mid, weight, torch.bfloat16)
        self.assertTrue(torch.equal(torch.cat((first, second)), whole))
        self.assertTrue(torch.equal(last, whole_state))

    def test_rejects_unverified_tap_count(self):
        ple = make_ple()
        normalized, state = self.inputs(8, 1)
        with self.assertRaises(ValueError):
            ple.conv_rows(normalized, state, torch.ones(CHANNELS, 3), torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
