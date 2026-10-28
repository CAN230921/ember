# Release validation — 0.1.1 experimental

Checks performed on 2026-10-28. This is an experimental source release, not a
production deployment approval. SGLang model containers were stopped only after
the owner explicitly authorized it; unrelated relay and license containers were
preserved. The Ember smoke container was stopped after testing.

| Check | Result |
| --- | --- |
| Complete pytest collection in the reference Linux aarch64 environment, with the actual local tokenizer | 634 passed, 1 optional CPU timing test skipped; 1,343 subtests passed; exit 0 |
| Local macOS CPU collection without Triton or model assets | 622 passed, 13 tokenizer-dependent tests skipped; 1,189 subtests passed; exit 0 |
| All five engine CUDA libraries and the bundled deterministic QSA extension | Compiled on SM121, exit 0 |
| CPU environment/native-library file check | Passed, exit 0 |
| `serve --check-install` in a fresh workspace | Optimizers installed, `model_loaded=false`, exit 0 |
| GPU environment and native-operator load check | Passed on SM121, exit 0; no model weights loaded |
| Complete conversion from the original official NVFP4 checkpoint | Completed in a separate output directory, exit 0; original inputs read-only |
| Converted checkpoint versus the previously measured engine checkpoint | All 296,777 tensor names, dtypes and shapes match; all 93,866,052,346 tensor-data bytes match; zero differences; exit 0 |
| Standalone service load using the public launcher and freshly converted checkpoint | Loaded on SM121 in 115.86 seconds; launch exit 0 |
| Live authentication and JSON chat | Unauthenticated model listing returned 401; authenticated model listing and marker response passed |
| Live SSE and simulated Agent tool workflows at 1, 2 and 4 concurrent lanes | 12/12 measured scenarios passed after 13 warmups; one repeat; exit 0 |
| Test service shutdown | Own container stopped normally, exit 0; no GPU compute process remained |

The Agent smoke uses `benchmarks/agent_suite.py` with synthetic tool fixtures.
It covers patch loops, repository inspection, context retrieval and code fixes;
fixture tools operate on in-memory data. This establishes basic serving and
protocol behavior. One repeat is not a performance benchmark or full model
quality evaluation.

The 634-test result covers the runtime, native-library search fix, checkpoint
conversion fixture, CPU row-linear fallback, memory-guard signal/ownership
contracts and source-index path validation. The source index rejects filenames
that could escape the checkpoint/output directory. GPU row-linear operations
continue to use the existing Triton implementation.

An earlier service-start attempt returned exit 1 with `GPU already has a
compute workload`, before creating a container. That attempt remains failed;
the subsequent authorized, isolated SM121 smoke above is a separate run.

Earlier failures remain failures: the first environment probe tried to load a
CUDA-dependent vLLM extension in a CPU-only container and returned exit 1. The
checker was corrected to check files in CPU mode and load operators only with
`--gpu`; the subsequent CPU and GPU checks passed. A local default Python probe
without the declared dependencies also failed; it was not treated as a runtime
regression or replaced with a claimed pass. The first isolated macOS CPU suite
also exposed 11 eager Triton-import failures. Moving the CPU reference path
behind a device-aware row-linear helper resolved them; the subsequent complete
local collection produced the 622-pass result above.

## What has not been established

- A validated publicly hosted base image. The tested downstream native ABI is
  described in [ENVIRONMENT.md](ENVIRONMENT.md); the optional Dockerfile still
  requires a compatible base supplied by the user.
- Fresh performance numbers for the public package. Tables in
  [BENCHMARKS.md](BENCHMARKS.md) are historical v30 results.
- Full GPU numerical/quality non-regression on this exact package, live
  cancellation/queue/cache acceptance, and four simultaneous 262,144-token
  physical-context capacity. Configured logical limits are not capacity proof.
- Public distribution of weights, compiled libraries or the reference image.

The current converter recipe quantizes the original official checkpoint; it is
lossy relative to the original BF16 projections/FP8 PLE. The equality check above
is against the **previously converted** engine checkpoint, not against BF16
weights. Safetensors file names and shard boundaries may differ even when every
named tensor's data is identical.
