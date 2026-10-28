# Ember

An inference engine for **Qwen3.8 Flash Next** on **NVIDIA DGX Spark (GB10, SM121)**, serving an OpenAI-compatible HTTP/SSE API with tool calls and reasoning budgets.

[中文说明](README.zh-CN.md)

> **Status: 0.1.2, experimental.** This is a research release for one model on one GPU. Historical performance numbers below come from the original v30 runtime. Package-specific checks are recorded in [docs/VALIDATION.md](docs/VALIDATION.md); the full quality suite and four-lane 262K capacity gate remain open.

## Highlights

| Area | What it does |
| --- | --- |
| Speculative decoding | Native MTP head plus a DFlash2 block drafter (NVFP4 weights, fused Triton kernels), a cost-model chooser between them, and 8-row single-lane verification |
| MoE | SM121 W4A4 routed-expert MoE (NVFP4 activations × NVFP4 weights, CuTe DSL), deterministic tactic table, row-GEMV fusion for router and indexer |
| Concurrency | Four lanes with continuous batching; MTP target and draft passes are batched across lanes and time; configured limits of 262,144 tokens per lane and 1,048,576 shared (full capacity unvalidated) |
| State and cache | Accepted-only GDN state cloning, request-private state arena, CUDA graphs reading request state through indirect descriptors, multi-branch copy-on-write prefix cache |
| Kernels | CUDA (INT8 output head, FP8 quantization, shared epilogue) and Triton (GDN, QSA sparse attention, hyper-connections, KV, DFlash) |
| Serving | Reasoning budgets (low / medium / high), repetition guard, tool-schema validation, cancellation, bounded queue |

## Performance

These are **historical v30 baseline results**, measured before the public packaging changes; they are not a new benchmark of this release.

Same DGX Spark, same converted checkpoint, same synthetic Agent workload (multi-step tool use, temperature 0, reasoning off, up to 1024 output tokens per call). Throughput is total completion tokens over wall time, median of two runs, summed across lanes.

| Concurrency | v30 baseline | vLLM (dev build) | llama.cpp |
| --- | ---: | ---: | ---: |
| 1 | **40.1** tok/s | 32.8 | 23.1 |
| 2 | **46.4** | 37.2 | 22.5 |
| 4 | **56.7** | 48.2 | 37.5 |

Single-lane decode on a code-fix prompt (SSE, excluding time to first chunk): **55.7** vs 46.2 vs 26.4 tok/s.

Caveats: the three engines use different drafters, caches and context limits, and produce different output sequences; tools are simulated; two repeats per cell. vLLM is a GB10 development build, not the latest release. Treat this as a serving-configuration comparison, not a kernel ranking. Details and the benchmark script are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [benchmarks/](benchmarks/).

## Requirements

- NVIDIA DGX Spark (GB10, SM121) with CUDA 13 and an SM121-capable `nvcc`.
- A container image with PyTorch, Triton, `transformers` and `nvidia-cutlass-dsl` 4.7.1. The engine also loads three compiled operators from a vLLM build for GB10 (`_C` `cutlass_scaled_mm`, `_moe_C` `topk_softmax`, and a deterministic `persistent_topk`); point `ENGINE_NATIVE_KERNEL_DIR` at them if they are not importable from `vllm/`.
- Weights:
  - target: `nvidia/Qwen3.8-Flash-Next-NVFP4` converted to this engine's layout (ternary PLE table, FP8 projections and INT8 output head re-derived from the official BF16 weights, Top10 + shared-expert config);
  - drafter: `incoai/Qwen3.8-Flash-Next-DFlash2`, quantized to NVFP4 at load.

  Convert the original checkpoint with [`tools/convert_checkpoint.py`](tools/convert_checkpoint.py); see [docs/WEIGHTS.md](docs/WEIGHTS.md). The source directory is never overwritten.

The compatible base-image/native-operator ABI is still a requirement; no public prebuilt image is validated or distributed. Exact reference versions, the optional environment Dockerfile and environment checker are documented in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Quick start

```bash
# Use a compatible GB10 vLLM image (see docs/ENVIRONMENT.md).
export IMAGE='your-compatible-GB10-image:tag'
# Build all engine kernels and the bundled deterministic QSA extension.
docker run --rm -v "$PWD:/repo" -w /repo --entrypoint bash "$IMAGE" scripts/build_kernels.sh
# Check dependencies/files (no weights or GPU needed).
docker run --rm -v "$PWD:/repo:ro" -w /repo --entrypoint python3 "$IMAGE" scripts/check_environment.py
# Convert the official target checkpoint as described in docs/WEIGHTS.md.
# Serve with read-only weights and a private API key file.
MODEL_DIR=... DFLASH_DIR=... KEY_FILE=... bash scripts/run_docker.sh
```

The launcher refuses an occupied GPU or reused container name and starts a
host-memory guard tied to its newly created, labelled container. It requires
110 GiB initial available RAM and stops that container below 20 GiB available
or on excessive swap growth. It never stops unrelated containers.

The server listens on port 8000 in the container and requires `Authorization: Bearer <key>`. Endpoints: `/v1/models`, `/v1/chat/completions` (streaming and tools), `/v1/scheduler`.

## Layout

```
runtime/
  serve.py                 entrypoint: installs the optimizers, starts the server
  api_server.py            OpenAI-compatible HTTP/SSE server
  continuous_scheduler.py  multi-lane scheduling and admission
  context_pool.py          shared context budget
  moe_adapter.py, moe_upfirst.py   SM121 W4A4 MoE backend (derived from FlashInfer)
  state_opt.py, indirect.py        GDN state-copy optimizations
  engine/                  model, MTP / DFlash, caches, weight loading
  engine/ops/              CUDA and Triton kernels
  engine/vendor/fla/       flash-linear-attention chunk kernels
  test_*.py                unit tests
scripts/                   kernel build and container launch
benchmarks/agent_suite.py  the Agent workload used above
docs/                      benchmark details
```

## Tests

```bash
bash scripts/test.sh
```

pytest collects both unittest classes and function-style tests. Install `requirements-test.txt` into the compatible environment. Most tests run on CPU but need the image's PyTorch, Triton and tokenizer; tests that need a real tokenizer read `SPARK_TEST_TOKENIZER_DIR` and are skipped without it. CPU tests do not prove GPU numerics or end-to-end speed.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party code keeps its original license: `engine/vendor/fla/` (Apache-2.0 and MIT) and `moe_upfirst.py` (Apache-2.0, FlashInfer). Model weights are not distributed here and follow their own licenses.
