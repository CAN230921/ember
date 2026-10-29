# Validation — 0.1.2

The opt-in experimental branch has separate [tree-verification evidence](TREE-VERIFICATION.md).
It is not part of the v0.1.2 release results below.

Checks run on 2026-10-28 on a DGX Spark (SM121) and on macOS.

| Check | Result |
| --- | --- |
| Full pytest run in the reference Linux aarch64 environment, with the real tokenizer | 634 passed, 1 optional CPU timing test skipped; 1,343 subtests passed |
| macOS CPU run without Triton or model files | 622 passed, 13 tokenizer tests skipped; 1,189 subtests passed |
| Five engine CUDA libraries and the bundled deterministic QSA extension | Compiled on SM121 |
| Environment checker, CPU mode | Passed |
| `serve --check-install` in a fresh workspace | Optimizers installed, `model_loaded=false` |
| Environment checker, GPU mode (native operator load) | Passed on SM121 |
| Full conversion from the official NVFP4 checkpoint | Completed |
| Converted checkpoint vs. the checkpoint used for the benchmarks | All 296,777 tensor names, dtypes and shapes match; all 93,866,052,346 tensor-data bytes identical |
| Service start with the public launcher and the newly converted checkpoint | Loaded on SM121 in 115.86 s |
| Authentication and JSON chat | Unauthenticated request returns 401; authenticated requests pass |
| SSE and simulated Agent tool workflows at 1, 2 and 4 lanes | 12/12 scenarios passed, one run each after warm-up |

The Agent test uses `benchmarks/agent_suite.py` with in-memory tool fixtures
(patch loops, repository inspection, context retrieval, code fixes). It checks
serving and protocol behavior, not speed or model quality.

The test suite covers the runtime, native-library lookup, checkpoint conversion,
the CPU row-linear fallback, the memory guard and source-index path validation.

## Not yet tested

- Full GPU quality evaluation of this package.
- Cancellation, queueing and prefix caching under real load.
- Four lanes at the full 262,144-token context at the same time.

No prebuilt image is provided; see [ENVIRONMENT.md](ENVIRONMENT.md).

The converter is lossy relative to the official BF16/FP8 weights (see
[WEIGHTS.md](WEIGHTS.md)). The byte comparison above is against the checkpoint
used for the benchmarks. Shard file names may differ between the two.
