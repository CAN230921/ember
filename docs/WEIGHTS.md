# Checkpoint conversion

The engine does not load arbitrary Hugging Face or GGUF weights. It expects the
Qwen3.8 Flash Next 48-layer, 512-expert, hidden-size-2560 layout, Top10 routing,
block128 FP8 projections, a symmetric GPTQ8/group128 output head and the
`tern_g16_none` PLE table. It also expects fused BF16 MTP expert banks.

Obtain `nvidia/Qwen3.8-Flash-Next-NVFP4` and
`incoai/Qwen3.8-Flash-Next-DFlash2` under their respective terms. This repository
does not download or include weights; pass local directories, mounted read-only.

`tools/convert_checkpoint.py` builds the engine layout from the official NVFP4
checkpoint:

```bash
mkdir -p converted
docker run --rm --gpus all --memory 12g --memory-swap 12g \
  -v "$PWD:/repo:ro" -v "$OFFICIAL_DIR:/source:ro" -v "$PWD/converted:/out" \
  -w /repo --entrypoint python3 "$IMAGE" -B tools/convert_checkpoint.py \
  --source /source --output /out/checkpoint --ple-device cuda
```

Run on an idle GPU. The converter can instead use `--ple-device cpu` without
`--gpus all`, at a substantial time cost. It streams tensors and PLE row chunks
and will not write into an existing directory; if a run fails, start again in a
new one. It needs roughly 100 GB of free disk space for the output.

Transformations:

- Original NVFP4 expert bytes and their scales are copied unchanged.
- 300 BF16 main-model projections are quantized to E4M3 using one F32 scale per
  128×128 block.
- The BF16 output head is packed as symmetric GPTQ8, zero point 128, group128;
  group scales are stored in FP16 using the engine's established convention.
- FP8 MTP experts are dequantized and fused into BF16 gate/up and down banks.
- Each PLE row is rounded to BF16, then coded with L2-optimal ternary
  group16: 32 packed base-three bytes plus ten E4M3 scale bytes.

The projection, output-head and PLE steps are lossy. `ember-conversion.json`
records the source index hash and the output tensor-data hashes.

Point `MODEL_DIR` at the output checkpoint and `DFLASH_DIR` at the downloaded
drafter. DFlash weights are quantized to NVFP4 at model load. The bundled draft
vocabulary is tokenizer-hash-bound; an incompatible tokenizer is rejected.

The llama.cpp comparison in [BENCHMARKS.md](BENCHMARKS.md) used a separate GGUF
converter, which Ember does not need.
