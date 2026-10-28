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
mkdir -p "$workspace/logs/api" "$workspace/cache/triton"
workspace="$(cd "$workspace" && pwd)"
lab="$(mktemp -d "$workspace/lab-XXXXXXXX")"
[[ -d "$MODEL_DIR" && -d "$DFLASH_DIR" && -s "$KEY_FILE" ]] || { echo 'Check MODEL_DIR, DFLASH_DIR and KEY_FILE' >&2; exit 1; }
if docker inspect "${NAME:-ember}" >/dev/null 2>&1; then echo 'Container name already exists; choose another NAME' >&2; exit 1; fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo 'GPU already has a compute workload' >&2; exit 1; fi
baseline_swap="$(python3 - <<'PY'
from pathlib import Path
m={k:int(v.split()[0]) for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())}
if m['MemAvailable']<110*1024**2:raise SystemExit('Insufficient initial memory headroom')
print(m['SwapTotal']-m['SwapFree'])
PY
)"

config_mount=()
if [[ -n "${MODEL_CONFIG:-}" ]]; then config_mount=(-v "$MODEL_CONFIG:/model/config.json:ro"); fi
deps_mount=()
if [[ -n "${CUTLASS_ROOT:-}" ]]; then deps_mount=(-v "$CUTLASS_ROOT:/deps:ro" -e SPARK_CUTLASS_DSL_DIR=/deps/nvidia_cutlass_dsl/dsl_packages); fi

cid="$(docker run -d --name "${NAME:-ember}" --label io.ember.engine=true --restart no --gpus all --ipc=host \
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
  "${deps_mount[@]}" \
  -v "$KEY_FILE:/secrets/api.key:ro" \
  -w /runtime --entrypoint python3 "$IMAGE" -u -m serve)"
python3 - "$repo/scripts/watch_memory.py" "$cid" "$baseline_swap" "$lab/guard.ready" "$lab/guard.log" <<'PY'
import subprocess,sys,time
from pathlib import Path
script,cid,swap,ready,log=sys.argv[1:]
try:
 with open(log,'xb') as stream:
  process=subprocess.Popen([sys.executable,script,cid,swap,ready],stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,start_new_session=True)
 for _ in range(50):
  if process.poll() is not None:raise RuntimeError('Memory guard exited during startup')
  if Path(ready).exists():break
  time.sleep(.2)
 else:raise RuntimeError('Memory guard startup timeout')
except BaseException:
 subprocess.run(['docker','stop','--timeout','30',cid],check=True,timeout=45)
 raise
print(cid)
PY
