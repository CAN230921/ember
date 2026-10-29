# Experimental tree verification

The default serving runner uses its original MTP verifier. Select tree
speculation explicitly at startup with `SPECULATION=tree` in the Docker
launcher, or `SPARK_SPECULATION=tree` / `--speculation tree` in the direct
entrypoint. The direct entrypoint requires `continuous_mtp` mode and DFlash2.

The scheduling idea was reviewed against [Uzu's source](https://github.com/trymirai/uzu/tree/f1231705cc7e26a205819da7c78e0110cfbac306).
The implementation uses Ember's existing CUDA arithmetic and current target
and DFlash2 checkpoints. It adds no model weights or draft training.

## Two independent experimental paths

`engine.verify_deferred(ids, state)` implements two-phase **linear** GDN
verification. It returns logits, hidden features and a transaction with explicit
`commit(consumed_input_count)` / `discard()` operations. Verification retains
convolved inputs and small convolution prefixes. Commit replays only the
accepted recurrent updates with the original decode kernel; projections and
experts are not repeated.

For eight verification rows, the retained GDN snapshot payload drops from
13,074,432 to 656,896 bytes per layer. This accounting excludes model weights,
base state and output tensors. It is not total model memory usage. Full-model
paired verification-plus-commit timings showed parity, not a stable speed gain.

`engine.verify_tree(tree, state, observer=None)` implements a small tree using
unique node rows. Shared ancestors are evaluated once; large projections, HC
and MoE use the unique rows together. Each GDN, QSA compressed-tail and PLE
n-gram/convolution history follows its parent. Positions follow ancestor depth.

- `CandidateTree` permits one to eight nodes including the unconsumed anchor.
- The initial model backend permits at most four leaves.
- `draft.propose_tree(ctx, anchor)` reuses the DFlash2 candidate lattice with
  two active branches and up to three draft levels (usually seven nodes).
- `draft.propose_tree(ctx, anchor, policy='best_first')`, or
  `SPARK_TREE_POLICY=best_first`, spends up to eight nodes on the highest-scoring
  available prefixes. It can extend seven draft levels while retaining at most
  four leaves. The existing beam policy remains the default. These normalized
  selector scores are proposal heuristics, not target probabilities.
- Selection follows target argmax. Proposal scores never accept a token.
- The predicting input is committed; EOS, correction and final bonus output
  remain unconsumed. Output budgets are checked at every accepted node.
- `SPARK_TREE_GDN_KERNEL=1` enables three GDN launches per tree/layer:
  convolution, recurrent updates and normalization. Parent caches are BF16.
  Selected node-bank views are copied into independent committed storage.
- Without that flag, GDN calls the original single-node cores in parent order.

PLE hashing reads only the required token-history tail (two tokens for the
current checkpoint), rather than copying the full history per candidate.
The parent convolution state retains its causal history. Tree depth is bounded
by the target checkpoint's remaining context; the final legal anchor can use
a root-only tree.

The tree path retains per-node GDN caches and does **not** yet use the linear
path's deferred memory saving. Branched verification is eager, with no merged
tree CUDA Graph. Serving reuses the original linear graph verifier for an
unbranched chain. Private lanes run cooperatively on the existing model-owner
scheduler; no multi-lane tree GPU batch is implemented.

## Numerical and ownership contracts

The current checkpoint writes its GDN recurrence to BF16 between tokens; beta,
convolution products and core outputs have their own BF16 boundaries. An FP32
closed-form or triangular solve is not an interchangeable implementation.
The tree kernel preserves the original recurrent expression/order and reads
each parent's materialized BF16 cache.

Only one verification transaction may be pending on an engine. Another model
call discards the old lease. A committed path cannot cross siblings or publish
unaccepted state. Callers must serialize access and preserve the base state
until commit/discard. These interfaces implement greedy decoding only.

`engine.verify_tree_reference(tree, state)` expands root-to-leaf paths into
existing private batch lanes and checks shared ancestor logits. It is a
correctness oracle and intentionally recomputes shared ancestors.

## Validation tools

- `tools/validate_gdn_deferred.py`: synthetic actual-shape GPU verification,
  every accepted prefix, subsequent decode, immutable base and inference mode.
- `tools/validate_gdn_tree.py`: synthetic branched/linear topologies, every
  node cache/core and subsequent decode against the original kernel.
- `tools/validate_linear_verification.py`: real current-checkpoint logits,
  hidden, complete GDN/QSA/PLE state and generation checks. Tree-only runs use
  `SPARK_VALIDATE_TREE_ONLY=1`; optional alternating generation timings use
  `SPARK_TREE_TIMING=1`.
- `tools/tree_generation.py`: isolated full-generation harness, including
  DFlash feature updates from **only the committed ancestor rows**.
- `tools/validate_tree_improvements.py`: current-weight serial-node/state/next-step
  equality and alternating baseline/beam/best-first request and Agent timings.

Run GPU tools with the reference dependencies and native libraries, exclusive
GPU ownership, a bounded memory cgroup and the memory guard. They do not open
an HTTP listener. Keep each attempt's source, exit status and results in a
separate evidence directory.

Initial completed evidence (2026-10-29): 660 CPU tests passed, 13 tokenizer
tests skipped; the linear model passed all 22 accepted-prefix checks at 25 and
2,081 context tokens. The fused-GDN unique-node model passed all ten tree
checks, including a target path through the second branch. Synthetic fused
GDN passed all 14 topology/inference-mode cases, including subsequent decode.

Four alternating-order full-request measurements per backend, after warm-up,
used the same current checkpoint and disabled the prefix cache. Every output
token matched. These are one-lane synchronous requests, including prefill and
draft, with no HTTP transport:

| Request | Output tokens | Accepted runner | Eager tree | Tree GDN kernel |
| --- | ---: | ---: | ---: | ---: |
| Short Python function | 25 | 0.5532 s / 45.19 tok/s | 0.5737 s / 43.58 tok/s | 0.5556 s / 45.00 tok/s |
| Agent tool-dispatch module | 128 | 2.8510 s / 44.90 tok/s | 2.7602 s / 46.37 tok/s | 2.6593 s / 48.13 tok/s |

The longer response improved about 7.2%; the short response was effectively
unchanged. One earlier cold smoke test took 1.04 s for the tree vs. 0.56 s for
the runner. Do not equate warmed results with cold-start latency or generalize
these two requests to Agent workflow throughput. The fused GDN microbenchmark
is a layer measurement, not an end-to-end speed ratio.

`tools/validate_tree_agent.py` additionally reuses the existing virtual
read/patch/test and inspection fixtures with real normalization and tool
parsing, deterministic fixture call IDs, and identical prompt/output-token
hashes per turn. It remains a synchronous in-process test; HTTP/SSE and
concurrent scheduling require separate acceptance.

These initial checks do not establish multi-step Agent throughput, model quality, concurrent
serving, cancellation/queue/cache behavior or four-lane 262K capacity.
`release_approved=false`.

## Bounded-history and best-first update, 2026-10-30

688 CPU tests passed, with 13 tokenizer tests skipped and 1,189 subtests passed.
On the same current target and DFlash2 weights, all 36 GPU checks matched
serial decoding bit for bit: every node's logits/hidden, committed full state,
unchanged base state and subsequent decode. Cases include actual proposals,
an eight-node chain, asymmetric branches, four/six-node boundaries and output
budgets 1/2/8 at 25 and 2,081 context tokens.

Four timed samples per cell after warm-up, alternating backend order, produced
identical output token IDs:

| Single-lane synchronous request | Tokens | Existing runner | Beam tree | Best-first tree |
| --- | ---: | ---: | ---: | ---: |
| Short Python function | 25 | 45.21 tok/s | 44.85 tok/s | 58.72 tok/s |
| Tool-dispatch code module | 128 | 44.72 tok/s | 47.72 tok/s | 51.77 tok/s |
| Virtual read/patch/test/report workflow | 201 | 28.84 tok/s | 28.87 tok/s | 32.46 tok/s |
| Virtual file-inspection workflow | 297 | 33.09 tok/s | 33.24 tok/s | 34.24 tok/s |

The Agent workflows passed 30/30 fixture runs (24 timed, six warm-up), with
identical prompt/output-token hashes at every turn. Their median workflow
times changed from 6.9699 to 6.1919 seconds and 8.9757 to 8.6734 seconds,
respectively. Throughput improved about 12.6% and 3.5% on these two fixtures.
These fixtures operate on in-memory files with simulated tests. They do not
measure real repository-repair success, HTTP/SSE, concurrent scheduling or
prefix-cache reuse. All timings include prefill and draft, with prefix caching
disabled; they are not pure decode TPS or a comparison with another engine.

The code-module response needed 30 verification rounds instead of the beam
tree's 35. The short function needed four instead of six. This is one loaded
process and a small fixed workload, so the result is experimental. No service
or release was promoted, and `release_approved=false`.

## Serving switch validation, 2026-10-30

The launcher defaults to `SPECULATION=mtp`; `SPECULATION=tree` selects
best-first tree decoding. Change modes on the next start, using a free container
name and the same model environment. Existing containers are retained by the
launcher. `/health`, `/v1/scheduler` and the startup receipt identify the mode;
request audits identify tree rounds and their node-consumption counts.

Tree prefill retains the existing paired target/MTP checkpoint cache. Decode
keeps private target/DFlash state for each request and does not enter an MTP
fallback with stale draft state. EOS, output/reasoning limits and protocol
stops use the existing request control. Cancellation discards uncommitted
verification; close releases the per-request DFlash context. Selected graph
views are detached even with lazy cloning disabled.

719 local CPU tests passed (13 tokenizer tests skipped). With the exact current
checkpoint, two separately started reference-image containers passed 14 MTP
and 16 tree HTTP checks: authentication, JSON/SSE, virtual tool workflows,
four concurrent private requests, repeated-prefix cache hits of 2,048 tokens,
reasoning-budget controls, TCP disconnect cancellation and recovery. Audits
confirmed actual tree decode and forced reasoning tokens. Both containers
stopped with exit 0; the GPU was released and the unrelated service was retained.

These HTTP runs are functional smoke tests, not a new throughput study. The
earlier throughput tables use an in-process generation harness; HTTP uses a
different scheduling path and first requests can include graph capture.
Do not use those tables as the measured speedup of this service switch.
Full model quality, broad workloads, queue saturation and four full 262K
contexts remain unverified. `release_approved=false`.
