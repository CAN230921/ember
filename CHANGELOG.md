# Changelog

## 0.1.2 — 2026-10-28 (experimental)

- Point setup-python pip caching at `requirements-test.txt`; the first GitHub
  workflow failed during cache setup before installing dependencies or tests.
- Preserve the already published 0.1.1 tag. Runtime and converter code are
  unchanged from the SM121-validated 0.1.1 snapshot.

## 0.1.1 — 2026-10-28 (experimental)

- Honor the native-kernel directory override for deterministic QSA selection;
  include its attributed source and build script.
- Provide a streaming official-checkpoint converter without a private template,
  with refusal to overwrite outputs and source-index path validation.
- Collect pytest functions and unittest classes together; add pinned test/runtime
  dependencies, CPU CI and an environment checker.
- Guard startup against an occupied GPU, reused container name or insufficient
  host memory; monitor only the newly created, labelled container.
- Document the native ABI/base-image requirement and separate historical v30
  benchmark results from public-package validation.
- Keep the CPU row-linear reference path usable without Triton; preserve the
  existing CUDA implementation.
- Verify a complete checkpoint conversion against the previously measured model
  and pass live SM121 authentication, JSON/SSE and 12 simulated Agent scenarios.
- Preserve 0.1.0 history. Full GPU quality and physical-context capacity gates
  remain open; see docs/VALIDATION.md.

## 0.1.0

Initial experimental source extraction from the measured engine baseline.
