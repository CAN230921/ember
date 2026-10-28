#!/usr/bin/env bash
# Start the engine in a container with the measured four-lane configuration.
#
# Required environment:
#   IMAGE         container image providing PyTorch, Triton, CUDA 13 and the vLLM
#                 native libraries (see README "Requirements")
#   MODEL_DIR     converted checkpoint directory (read-only)
#   DFLASH_DIR    DFlash2 drafter directory (read-only)
#   KEY_FILE      file holding the API key clients must send as a Bearer token
# Optional:
#   MODEL_CONFIG  config.json overriding the checkpoint's (e.g. Top10 + shared expert)
#   WORKSPACE     writable directory for logs, Triton cache and optimizer output
#   PORT (8000)   BIND (127.0.0.1)   NAME (ember)   ACTIVE (4)
set -euo pipefail
: "${IMAGE:?}" "${MODEL_DIR:?}" "${DFLASH_DIR:?}" "${KEY_FILE:?}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
workspace="${WORKSPACE:-$repo/workspace}"
lab="$workspace/lab-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$workspace/logs/api" "$workspace/cache/triton" "$lab"

config_mount=()
if [[ -n "${MODEL_CONFIG:-}" ]]; then config_mount=(-v "$MODEL_CONFIG:/model/config.json:ro"); fi

exec docker run -d --name "${NAME:-ember}" --restart no --gpus all --ipc=host \
  --memory 96g --memory-swap 96g --stop-timeout 30 \
  -p "${BIND:-127.0.0.1}:${PORT:-8000}:8000" \
  -e PYTHONPATH=/runtime -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=4 \
  -e TOKENIZERS_PARALLELISM=false -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/workspace/cache/triton -e SPARK_LAB_DIR=/lab \
  -e SPARK_KEY_FILE=/secrets/api.key -e SPARK_API_AUDIT=/workspace/logs/api/requests.jsonl \
  -e SPARK_EXECUTION_MODE=continuous_mtp -e SPARK_MTP_BATCH=1 \
  -e SPARK_MAX_ACTIVE="${ACTIVE:-4}" -e SPARK_MAX_QUEUED=1 \
  -e SPARK_MAX_CONTEXT=262144 -e SPARK_MAX_OUTPUT=0 -e SPARK_TOTAL_CONTEXT=1048576 \
  -e SPARK_PREPARE_WORKERS=2 -e SPARK_PREFILL_CHUNK=2048 -e SPARK_QUEUE_TIMEOUT_SECONDS=120 \
  -e SPARK_DRAFT_TOKENS=2 -e SPARK_MTP_POLICY=acceptance -e SPARK_LONG_WARMUP_TOKENS=12480 \
  -e SPARK_MOE_STRATEGY=tuned_top10_v1 -e SPARK_DRAFT_TOPK=10 \
  -e SPARK_LOW_SYNC=1 -e SPARK_BORROWED_CACHE=0 -e SPARK_QSA_PREFILL_MODE=predecoded \
  -e SPARK_HC_FUSED=1 -e SPARK_HEAD_ROWS=1 -e SPARK_MTP_PAD=1 -e SPARK_DRAFT_BATCH=1 -e SPARK_LAZY_CLONE=1 \
  -e SPARK_ROW_GEMV=1 -e SPARK_ROW_GEMV_DEEP=1 -e SPARK_DRAFT_FP8=1 -e SPARK_DRAFT_FP4=1 \
  -e SPARK_VERIFY_ROWS=8 -e SPARK_SHARED_GRAPH_POOL=1 \
  -e SPARK_DFLASH_DIR=/dflash -e SPARK_DFLASH_BLOCK=8 -e SPARK_DFLASH_BATCH=0 \
  -e SPARK_DRAFT_VOCAB_PATH=/runtime/draft-vocab-65536-v2.json \
  -e SPARK_DRAFT_VOCAB_SHA256=31261e3a17c360f86bf5b43bb087739935f63527ef9cc4e40e38ddde9dcc8a64 \
  -v "$repo/runtime:/runtime:ro" -v "$workspace:/workspace" -v "$lab:/lab" \
  -v "$MODEL_DIR:/model:ro" "${config_mount[@]}" -v "$DFLASH_DIR:/dflash:ro" \
  -v "$KEY_FILE:/secrets/api.key:ro" \
  -w /runtime --entrypoint python3 "$IMAGE" -u -m serve
