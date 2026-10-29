# Changelog

## 0.1.2 — 2026-10-28 (experimental)

- Fix the GitHub Actions pip cache: point setup-python at `requirements-test.txt`.
  Runtime and converter code are unchanged from 0.1.1.

## 0.1.1 — 2026-10-28 (experimental)

- Honor the native-kernel directory override for deterministic QSA selection;
  include its source and build script.
- Add a streaming converter from the official checkpoint. It refuses to overwrite
  existing output and validates shard paths in the source index.
- Collect pytest functions and unittest classes together; add pinned test/runtime
  dependencies, CPU CI and an environment checker.
- Check at startup for a busy GPU, a reused container name or too little host
  memory; the memory guard watches only the container it started.
- Document the base-image and native-operator requirements.
- Make the CPU row-linear reference path work without Triton.
- Tested on SM121: a full conversion matches the benchmark checkpoint
  byte for byte, and authentication, JSON/SSE and 12 simulated Agent scenarios
  pass. See docs/VALIDATION.md.

## 0.1.0

Initial experimental source release.
