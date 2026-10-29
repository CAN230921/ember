# Benchmarks

All numbers were measured on one NVIDIA DGX Spark (GB10), one engine at a time with the GPU otherwise idle.

## Agent workload

Script: [`benchmarks/agent_suite.py`](../benchmarks/agent_suite.py). Four synthetic scenarios against a fixed virtual repository; tool calls are executed by deterministic fixtures (no shell, no real file edits, no real test runs):

| Scenario | Description |
| --- | --- |
| `patch_loop` | read a file → apply a patch → run simulated tests → report |
| `inspect` | read a file and analyse it |
| `context` | analyse a ~5.5K-token project description |
| `code_fix` | write a patch and tests directly |

Settings: temperature 0, reasoning off, up to 1024 output tokens per model call, up to 8 tool rounds. Concurrency 1, 2 and 4 (four lanes run the four scenarios together), two timed repeats per cell after identical warm-up. A scenario passes when it satisfies simple rule checks (for example, an AST check that the patch fixes the boundary condition). All three engines passed 24/24 timed scenarios.

```bash
python3 benchmarks/agent_suite.py --port 8000 --model <model-id> --key-file <key> --out results.json
```

### Throughput (tok/s, sum across lanes)

| Concurrency | Ember | vLLM (dev build) | llama.cpp |
| --- | ---: | ---: | ---: |
| 1 | **40.1** | 32.8 | 23.1 |
| 2 | **46.4** | 37.2 | 22.5 |
| 4 | **56.7** | 48.2 | 37.5 |

Total completion tokens divided by the summed wall time of the concurrent batches, median of two runs. Includes prefill, generation, protocol and simulated tool round-trips.

### Scenario wall time (seconds, median of two)

| Scenario | Ember ×1 | vLLM ×1 | llama ×1 | Ember ×4 | vLLM ×4 | llama ×4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `patch_loop` | **6.44** | 10.75 | 9.73 | **13.27** | 20.55 | 28.29 |
| `inspect` | **8.61** | 10.29 | 11.00 | 17.44 | 19.71 | **16.03** |
| `context` | **3.79** | 5.05 | 4.98 | **10.09** | 13.97 | 13.76 |
| `code_fix` | **9.52** | 10.14 | 20.66 | **20.03** | 22.42 | 30.31 |

The engines take different paths: single-lane `patch_loop` used 4 model calls / 201 tokens on Ember, 6 / 307 on vLLM and 4 / 211 on llama.cpp. Wall time therefore reflects both speed and path length.

### Single-lane decode

`code_fix`, `(completion_tokens - 1) / (request time - time to first visible chunk)`: **55.7** (Ember), 46.2 (vLLM), 26.4 (llama.cpp) tok/s. SSE chunks can carry several tokens, so this is approximate.

## Configurations

| | Ember | vLLM | llama.cpp |
| --- | --- | --- | --- |
| Target weights | converted NVFP4 checkpoint, Top10 | same checkpoint, with a PLE mmap adapter | same checkpoint converted to GGUF (FP8 and ternary PLE expanded to F32) |
| Drafter | MTP + DFlash2 | MTP, 2 tokens | MTP, 2 tokens |
| Context | 262,144 per lane | 32,768 per request, 6 GiB KV | 32,768 total, 4 × 8,192 slots |
| Container memory | 96 GiB | 96 GiB | 104 GiB |

The llama.cpp build is upstream commit `de5122d879d00d2895c36be7e0cee94105aeeabd` without runtime changes, run with `--load-mode none --lazy-mode on --cache-ram 2048 --ctx-checkpoints 8 --ubatch-size 256`, batch 512, flash attention on, F16 KV, all regular layers on GPU. Expanding the PLE table to F32 (about 205 GB, mapped on demand) was needed for compatibility and likely costs llama.cpp some memory bandwidth.

Each engine was loaded once and the runs were not interleaved.
