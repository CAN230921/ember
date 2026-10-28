# Runtime environment

Ember is a source release. It does not redistribute a container, NVIDIA libraries,
vLLM binaries or model weights. The GPU tests use a compatible downstream GB10
vLLM image. A generic CUDA/PyTorch image by itself does not satisfy the native
operator ABI. No publicly hosted base image has been validated by this project.

The reference environment is Linux aarch64, Python 3.12, CUDA 13.0, PyTorch
2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.17, transformers 5.15.1 and CuTe DSL
4.7.1. The original downstream vLLM version identifies itself as
`0.1.dev20073+g8e685d198`; it is not the latest upstream release. Do not replace
Torch independently of the native libraries.

`requirements-runtime.txt` pins the Python-level dependencies. CUDA 13 needs the
CuTe DSL `cu13` extra. `requirements-test.txt` pins the test dependencies.

An optional environment Dockerfile accepts an explicit compatible base:

```bash
docker build --build-arg BASE_IMAGE='your-compatible-GB10-vLLM-image:tag' -t ember-env:0.1.1 .
export IMAGE=ember-env:0.1.1
docker run --rm -v "$PWD:/repo" -w /repo --entrypoint bash "$IMAGE" scripts/build_kernels.sh
docker run --rm -v "$PWD:/repo:ro" -w /repo --entrypoint python3 "$IMAGE" scripts/check_environment.py
```

The final check verifies installed dependencies, compiled files and native library
locations without loading model weights or requiring a CUDA device. With
`--gpus all` and `scripts/check_environment.py --gpu`, it also checks SM121 and
loads the native operators.

Two vLLM extensions must expose `torch.ops._C.cutlass_scaled_mm` and
`torch.ops._moe_C.topk_softmax`. The file names are
`_C_stable_libtorch.abi3.so` and `_moe_C_stable_libtorch.abi3.so`, discoverable in
the installed `vllm` package or `ENGINE_NATIVE_KERNEL_DIR`. The bundled
`runtime/native` sources build `_C_det.so` for deterministic QSA selection;
`scripts/build_kernels.sh` builds it along with the engine CUDA libraries.
Source attribution is recorded in NOTICE. Other native ABIs need a separately
validated adapter; the checker fails rather than silently switching numerics.

To check optimizer installation in a fresh workspace:

```bash
docker run --rm -v "$PWD/runtime:/runtime:ro" -v "$(mktemp -d):/lab" \
  -w /runtime --entrypoint python3 "$IMAGE" -B -m serve --check-install
```

For an unpacked CuTe distribution, `SPARK_CUTLASS_DSL_DIR` points to its
`nvidia_cutlass_dsl/dsl_packages` directory. `run_docker.sh` also accepts
`CUTLASS_ROOT` to mount the complete distribution at `/deps`.

Tests collect both unittest and pytest functions:

```bash
docker run --rm -v "$PWD:/repo:ro" -w /repo --entrypoint python3 "$IMAGE" \
  -B -m pytest -q -p no:cacheprovider
```

Use `SPARK_TEST_TOKENIZER_DIR` with a read-only tokenizer mount to include the
real-tokenizer tests. CPU tests do not validate GPU numerics, capacity or speed.
