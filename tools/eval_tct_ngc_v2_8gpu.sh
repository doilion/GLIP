#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

eval "$(conda shell.bash hook)"
conda activate glip

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu/ft_task_1/config.yml}"
WEIGHT_PATH="${WEIGHT_PATH:-${REPO_ROOT}/OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu/ft_task_1/model_final.pth}"
TASK_CONFIG="${TASK_CONFIG:?TASK_CONFIG is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TEST_BATCH="${TEST_BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_AMP="${USE_AMP:-True}"
SUBSET="${SUBSET:-}"
RESULT_DATASET_NAME="${RESULT_DATASET_NAME:-}"
if [[ -z "${RESULT_DATASET_NAME}" ]]; then
  # Match the dataset name that test_grounding_net.py will write under
  # inference/<dataset_name>/. That comes straight from cfg_.DATASETS.TEST[0]
  # after the YAML merge — no suffix is added because TEST_DATASETNAME_SUFFIX
  # defaults to "". Parsing here avoids a stale RESULT_DATASET_NAME silently
  # pointing runtime.json at a non-existent result_root.
  #
  # GLIP YAML configs use YACS convention: DATASETS.TEST may be a real YAML
  # list (e.g. our generated artifacts) or a tuple-shaped string like
  # `("val",)` (e.g. the fallback configs/tct_ngc/tct_ngc_v2_base.yaml). The
  # parser handles both via ast.literal_eval on the string form.
  RESULT_DATASET_NAME="$(python -c "
import ast, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
test = cfg.get('DATASETS', {}).get('TEST')
if not test:
    raise SystemExit('TASK_CONFIG missing DATASETS.TEST')
if isinstance(test, str):
    try:
        parsed = ast.literal_eval(test)
    except (ValueError, SyntaxError):
        parsed = test
    test = parsed
if isinstance(test, (list, tuple)):
    name = test[0]
elif isinstance(test, str):
    name = test
else:
    raise SystemExit(f'Unsupported DATASETS.TEST shape: {type(test).__name__}')
print(name)
" "${TASK_CONFIG}")"
fi

mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run.log}"
COMMAND_FILE="${OUTPUT_DIR}/command.sh"
RUNTIME_FILE="${OUTPUT_DIR}/runtime.json"
WEIGHT_BASENAME="$(basename "${WEIGHT_PATH}")"
WEIGHT_STEM="${WEIGHT_BASENAME%.pth}"
RESULT_ROOT="${OUTPUT_DIR}/eval/${WEIGHT_STEM}/inference/${RESULT_DATASET_NAME}"

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  tools/test_grounding_net.py
  --config-file "${CONFIG_FILE}"
  --task_config "${TASK_CONFIG}"
  --weight "${WEIGHT_PATH}"
  TEST.IMS_PER_BATCH "${TEST_BATCH}"
  DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
  TEST.EVAL_TASK detection
  OUTPUT_DIR "${OUTPUT_DIR}"
)

if [[ -n "${SUBSET}" ]]; then
  CMD+=(TEST.SUBSET "${SUBSET}")
fi

{
  printf '#!/usr/bin/env bash\n'
  printf 'cd %q\n' "${REPO_ROOT}"
  printf 'export CUDA_HOME=%q\n' "${CUDA_HOME}"
  printf 'export PATH=%q:$PATH\n' "${CUDA_HOME}/bin"
  printf 'export HF_ENDPOINT=%q\n' "${HF_ENDPOINT}"
  printf 'export TOKENIZERS_PARALLELISM=%q\n' "${TOKENIZERS_PARALLELISM}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${COMMAND_FILE}"

start_ts="$(date +%s)"
set +e
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
set -e
end_ts="$(date +%s)"
elapsed="$((end_ts - start_ts))"

python - <<'PY' "${RUNTIME_FILE}" "${COMMAND_FILE}" "${LOG_FILE}" "${TASK_CONFIG}" "${CONFIG_FILE}" "${WEIGHT_PATH}" "${RESULT_ROOT}" "${TEST_BATCH}" "${NUM_WORKERS}" "${USE_AMP}" "${NPROC_PER_NODE}" "${SUBSET}" "${status}" "${elapsed}"
import json
import sys
from pathlib import Path

(
    runtime_path,
    command_file,
    log_file,
    task_config,
    config_file,
    weight_path,
    result_root,
    test_batch,
    num_workers,
    use_amp,
    nproc_per_node,
    subset,
    status,
    elapsed,
) = sys.argv[1:]

payload = {
    "command_file": command_file,
    "log_file": log_file,
    "task_config": task_config,
    "config_file": config_file,
    "weight_path": weight_path,
    "result_root": result_root,
    "test_batch": int(test_batch),
    "num_workers": int(num_workers),
    "use_amp": use_amp.lower() == "true",
    "nproc_per_node": int(nproc_per_node),
    "subset_batches": None if subset == "" else int(subset),
    "status": int(status),
    "elapsed_seconds": int(elapsed),
}

Path(runtime_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

exit "${status}"
