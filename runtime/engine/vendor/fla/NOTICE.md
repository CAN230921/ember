# FLA numerical baseline

These eleven Python files come from the existing reference image `46b88ebfb9d8`, path `vllm/third_party/flash_linear_attention/ops`. Their original Apache-2.0 notices and FLA MIT copyright notices are preserved. They are third-party kernels, not authored optimization claims.

Local changes remove vLLM platform/import wrappers, use installed Triton directly, and replace diagnostic sync guards with nullcontext. The GB10 64-token triangular inverse uses a fixed four-warp, three-stage configuration: the autotuner's two-warp reduction changed rare BF16 entries and caused full-model drift. The arithmetic remains the licensed FLA baseline; this change is not an authored-kernel or all-shape performance claim. CPU Torch WY math remains in `engine/ops/gdn_prefill.py` for comparison.

The engine owns model execution, weights, sequence history, PLE I/O and state; these files only execute GDN chunk arithmetic.
