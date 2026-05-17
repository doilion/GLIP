#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

resolve_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${path}"
  fi
}

SESSION_NAME="${SESSION_NAME:-tct_ngc_v2_base}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu}"
LOG_DIR="${LOG_DIR:-${RUN_OUTPUT_DIR}/tmux_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_${RUN_ID}.log}"
LATEST_LOG_LINK="${LATEST_LOG_LINK:-${LOG_DIR}/latest.log}"

LOG_DIR="$(resolve_path "${LOG_DIR}")"
LOG_FILE="$(resolve_path "${LOG_FILE}")"
LATEST_LOG_LINK="$(resolve_path "${LATEST_LOG_LINK}")"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "$(dirname "${LATEST_LOG_LINK}")"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

ln -sfn "${LOG_FILE}" "${LATEST_LOG_LINK}"

CMD_EXPORTS=""
for var_name in \
  OUTPUT_DIR \
  PRETRAIN_WEIGHT \
  NPROC_PER_NODE \
  GLOBAL_BATCH \
  TEST_BATCH \
  NUM_WORKERS \
  TRAIN_MIN_SIZES \
  TRAIN_MAX_SIZE \
  CUDA_HOME \
  HF_ENDPOINT \
  TOKENIZERS_PARALLELISM \
  TORCH_DISTRIBUTED_DEBUG \
  OMP_NUM_THREADS \
  PYTORCH_CUDA_ALLOC_CONF
do
  if [[ -v "${var_name}" ]]; then
    printf -v export_stmt 'export %s=%q; ' "${var_name}" "${!var_name}"
    CMD_EXPORTS+="${export_stmt}"
  fi
done

printf -v CMD 'cd %q && %sbash tools/train_tct_ngc_v2_base_8gpu.sh 2>&1 | tee %q; status=${PIPESTATUS[0]}; echo; echo "[tmux] training exited with status ${status}"; exec bash' \
  "${REPO_ROOT}" \
  "${CMD_EXPORTS}" \
  "${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" bash -lc "${CMD}"

echo "started tmux session: ${SESSION_NAME}"
echo "log file: ${LOG_FILE}"
echo "attach: tmux attach -t ${SESSION_NAME}"
echo "tail log: tail -f ${LOG_FILE}"
echo "latest log: ${LATEST_LOG_LINK}"
