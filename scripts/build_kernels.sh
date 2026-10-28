#!/usr/bin/env bash
# Build the hand-written SM121 CUDA kernels next to their Python loaders.
# Run on the GB10 (or any host with an SM121-capable nvcc).
set -euo pipefail
ops="$(cd "$(dirname "$0")/../runtime/engine/ops" && pwd)"
for name in fp8_quant int8_head int8_head_rows_v2 int8_head_three shared_epilogue; do
  echo "building lib${name}_sm121.so"
  nvcc -std=c++17 -O3 -arch=sm_121 --shared -Xcompiler -fPIC \
    "$ops/${name}_sm121.cu" -o "$ops/lib${name}_sm121.so"
done
