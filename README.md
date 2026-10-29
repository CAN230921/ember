# Ember

An inference engine for **Qwen3.8 Flash Next** on **NVIDIA DGX Spark (GB10, SM121)**, serving an OpenAI-compatible HTTP/SSE API with tool calls and reasoning budgets.

[中文说明](README.zh-CN.md)

> **Status: 0.1.2, experimental.** Built for one model on one GPU. Test results are in [docs/VALIDATION.md](docs/VALIDATION.md); the full quality suite and four lanes at full 262K context have not been run yet.

## Highlights

| Area | What it does |
| --- | --- |
| Speculative decoding | Native MTP head plus a DFlash2 block drafter (NVFP4 weights, fused Triton kernels), a cost-model chooser between them, and 8-row single-lane verification |
| MoE | SM121 W4A4 routed-expert MoE (NVFP4 activations × NVFP4 weights, CuTe DSL), deterministic tactic table, row-GEMV fusion for router and indexer |
| Concurrency | Four lanes with continuous batching; MTP target and draft passes are batched across lanes and time; configured limits of 262,144 tokens per lane and 1,048,576 shared |
| State and cache | Accepted-only GDN state cloning, request-private state arena, CUDA graphs reading request state through indirect descriptors, multi-branch copy-on-write prefix cache |
| Kernels | CUDA (INT8 output head, FP8 quantization, shared epilogue) and Triton (GDN, QSA sparse attention, hyper-connections, KV, DFlash) |
| Serving | Reasoning budgets (low / medium / high), repetition guard, tool-schema validation, cancellation, bounded queue |

## Performance

Same DGX Spark, same converted checkpoint, same synthetic Agent workload (multi-step tool use, temperature 0, reasoning off, up to 1024 output tokens per call). Throughput is total completion tokens over wall time, median of two runs, summed across lanes.

| Concurrency | Ember | vLLM (dev build) | llama.cpp |
| --- | ---: | ---: | ---: |
| 1 | **40.1** tok/s | 32.8 | 23.1 |
| 2 | **46.4** | 37.2 | 22.5 |
| 4 | **56.7** | 48.2 | 37.5 |

Single-lane decode on a code-fix prompt (SSE, excluding time to first chunk): **55.7** vs 46.2 vs 26.4 tok/s.

The three engines use different drafters, caches and context limits, so their output sequences differ; tools are simulated; two runs per cell; vLLM is a GB10 development build. Details and the benchmark script are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [benchmarks/](benchmarks/).

## Requirements

- NVIDIA DGX Spark (GB10, SM121) with CUDA 13 and an SM121-capable `nvcc`.
- A container image with PyTorch, Triton, `transformers` and `nvidia-cutlass-dsl` 4.7.1. The engine also loads three compiled operators from a vLLM build for GB10 (`_C` `cutlass_scaled_mm`, `_moe_C` `topk_softmax`, and a deterministic `persistent_topk`); point `ENGINE_NATIVE_KERNEL_DIR` at them if they are not importable from `vllm/`.
- Weights:
  - target: `nvidia/Qwen3.8-Flash-Next-NVFP4` converted to this engine's layout (ternary PLE table, FP8 projections and INT8 output head re-derived from the official BF16 weights, Top10 + shared-expert config);
  - drafter: `incoai/Qwen3.8-Flash-Next-DFlash2`, quantized to NVFP4 at load.

  Convert the original checkpoint with [`tools/convert_checkpoint.py`](tools/convert_checkpoint.py); see [docs/WEIGHTS.md](docs/WEIGHTS.md).

No prebuilt image is provided. Reference versions, an optional Dockerfile and an environment checker are in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

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

The launcher checks that the GPU is idle and the container name is free, then
starts a memory guard for its container: it needs 110 GiB free RAM to start and
stops the container if free RAM drops below 20 GiB or swap grows too fast.

The server listens on port 8000 in the container and requires `Authorization: Bearer <key>`. Endpoints: `/v1/models`, `/v1/chat/completions` (streaming and tools), `/v1/scheduler`.

### Optional tree speculation

MTP remains the default. Select the experimental best-first tree backend at startup:

```bash
SPECULATION=tree MODEL_DIR=... DFLASH_DIR=... KEY_FILE=... bash scripts/run_docker.sh
```

Use `SPECULATION=mtp` to select the original backend on the next start. Direct
entrypoint users can set `SPARK_SPECULATION=tree` or pass `--speculation tree`
with `SPARK_EXECUTION_MODE=continuous_mtp` and their usual model environment.
`/health` and `/v1/scheduler` report the selected mode.

Tree mode retains streaming, tools, reasoning controls, paired prefill-cache
reuse and cancellation. Its private lanes run cooperatively, without GPU batch
verification; four-way full-context capacity has not been established. See
[implementation and validation](docs/TREE-VERIFICATION.md).

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
docs/                      environment, weights, benchmarks, validation
```

## Tests

```bash
bash scripts/test.sh
```

pytest collects both unittest classes and function-style tests. Install `requirements-test.txt` into the compatible environment. Most tests run on CPU but need the image's PyTorch, Triton and tokenizer; tests that need a real tokenizer read `SPARK_TEST_TOKENIZER_DIR` and are skipped without it.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party code keeps its original license: `engine/vendor/fla/` (Apache-2.0 and MIT) and `moe_upfirst.py` (Apache-2.0, FlashInfer). Model weights are not distributed here and follow their own licenses.
