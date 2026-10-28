# Ember

A hand-written inference engine for **Qwen3.8 Flash Next** on **NVIDIA DGX Spark (GB10, SM121)**, serving an OpenAI-compatible HTTP/SSE API with tool calls and reasoning budgets.

[中文说明](README.zh-CN.md)

> **Status: 0.1.0, experimental.** This is a research release for one model on one GPU. The numbers below were measured on a single DGX Spark; the full quality suite and the four-lane 262K-context capacity test have not been re-run on this exact package.

## Highlights

| Area | What it does |
| --- | --- |
| Speculative decoding | Native MTP head plus a DFlash2 block drafter (NVFP4 weights, fused Triton kernels), a cost-model chooser between them, and 8-row single-lane verification |
| MoE | SM121 W4A4 routed-expert MoE (NVFP4 activations × NVFP4 weights, CuTe DSL), deterministic tactic table, row-GEMV fusion for router and indexer |
| Concurrency | Four lanes with continuous batching; MTP target and draft passes are batched across lanes and time; 262,144 tokens per lane, 1,048,576 shared |
| State and cache | Accepted-only GDN state cloning, request-private state arena, CUDA graphs reading request state through indirect descriptors, multi-branch copy-on-write prefix cache |
| Kernels | Hand-written CUDA (INT8 output head, FP8 quantization, shared epilogue) and Triton (GDN, QSA sparse attention, hyper-connections, KV, DFlash) |
| Serving | Reasoning budgets (low / medium / high), repetition guard, tool-schema validation, cancellation, bounded queue |

## Performance

Same DGX Spark, same converted checkpoint, same synthetic Agent workload (multi-step tool use, temperature 0, reasoning off, up to 1024 output tokens per call). Throughput is total completion tokens over wall time, median of two runs, summed across lanes.

| Concurrency | Ember | vLLM (dev build) | llama.cpp |
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

  **The conversion tools are not part of 0.1.0 yet**, so the engine cannot be run from the public checkpoint alone today.

## Quick start

```bash
# 1. Build the CUDA kernels (on the GB10)
scripts/build_kernels.sh

# 2. Check that the optimizers install (no weights loaded)
SPARK_LAB_DIR=$(mktemp -d) python3 -m serve --check-install   # run inside runtime/ in the image

# 3. Serve
IMAGE=... MODEL_DIR=... DFLASH_DIR=... KEY_FILE=... scripts/run_docker.sh
```

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
cd runtime && python3 -m unittest discover -p 'test_*.py'
```

Most tests run on CPU but need the image's PyTorch, Triton and tokenizer; tests that need a real tokenizer read `SPARK_TEST_TOKENIZER_DIR` and are skipped without it. CPU tests do not prove GPU numerics or end-to-end speed.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party code keeps its original license: `engine/vendor/fla/` (Apache-2.0 and MIT) and `moe_upfirst.py` (Apache-2.0, FlashInfer). Model weights are not distributed here and follow their own licenses.
