#!/usr/bin/env bash

# Run the standard 8-GPU base eval against the dev30 30-class test split, then
# derive both the ``base_with_negative`` (full) and ``base_no_negative``
# (post-hoc COCOeval ``catIds`` view) summaries off the *same* bbox.json.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

eval "$(conda shell.bash hook)"
conda activate glip

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
LEGACY_MODEL_OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${LEGACY_MODEL_OUTPUT_DIR:-${REPO_ROOT}/OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu}}"
WEIGHT_PATH="${WEIGHT_PATH:-${MODEL_OUTPUT_DIR}/ft_task_1/model_final.pth}"
CONFIG_FILE="${CONFIG_FILE:-${MODEL_OUTPUT_DIR}/ft_task_1/config.yml}"

ARTIFACTS_DIR="${MODEL_OUTPUT_DIR}/eval/artifacts"
DEFAULT_TASK_CONFIG="${ARTIFACTS_DIR}/base_with_negative_eval.yaml"
if [[ -f "${DEFAULT_TASK_CONFIG}" ]]; then
  TASK_CONFIG="${TASK_CONFIG:-${DEFAULT_TASK_CONFIG}}"
else
  TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/configs/tct_ngc/tct_ngc_v2_base.yaml}"
fi
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${MODEL_OUTPUT_DIR}/eval_model_final}"
TEST_BATCH="${TEST_BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_AMP="${USE_AMP:-True}"
SUBSET="${SUBSET:-}"

# RESULT_DATASET_NAME is auto-derived inside tools/eval_tct_ngc_v2_8gpu.sh
# from TASK_CONFIG.DATASETS.TEST[0] — leave it unset here so the inner script
# stays in sync with whatever YAML we hand it.
CONFIG_FILE="${CONFIG_FILE}" \
WEIGHT_PATH="${WEIGHT_PATH}" \
TASK_CONFIG="${TASK_CONFIG}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
TEST_BATCH="${TEST_BATCH}" \
NUM_WORKERS="${NUM_WORKERS}" \
USE_AMP="${USE_AMP}" \
SUBSET="${SUBSET}" \
  bash tools/eval_tct_ngc_v2_8gpu.sh

# After eval succeeds, derive the two summary views from the same bbox.json.
RUNTIME_JSON="${EVAL_OUTPUT_DIR}/runtime.json"
if [[ ! -f "${RUNTIME_JSON}" ]]; then
  echo "[eval_tct_ngc_v2_base_8gpu.sh] runtime.json missing at ${RUNTIME_JSON}; skipping summarize" >&2
  exit 0
fi

RESULT_ROOT="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['result_root'])" "${RUNTIME_JSON}")"
PRED_JSON="${RESULT_ROOT}/bbox.json"
GT_JSON="$(python -c "import json,sys,yaml; cfg = yaml.safe_load(open(sys.argv[1])); reg = cfg['DATASETS']['REGISTER']; key = list(reg)[0]; print(reg[key]['ann_file'])" "${TASK_CONFIG}")"
EVAL_METADATA_JSON="${ARTIFACTS_DIR}/eval_metadata.json"
REPORT_DIR="${MODEL_OUTPUT_DIR}/eval/report"
mkdir -p "${REPORT_DIR}"

if [[ ! -f "${PRED_JSON}" ]]; then
  echo "[eval_tct_ngc_v2_base_8gpu.sh] predictions JSON missing at ${PRED_JSON}; skipping summarize" >&2
  exit 0
fi

# View 1: full base_with_negative.
python tools/summarize_tct_ngc_v2_eval.py \
  --name base_with_negative \
  --gt-json "${GT_JSON}" \
  --pred-json "${PRED_JSON}" \
  --runtime-json "${RUNTIME_JSON}" \
  --out-json "${REPORT_DIR}/base_with_negative_summary.json" \
  --out-csv  "${REPORT_DIR}/base_with_negative_per_class.csv"

# View 2: post-hoc base_no_negative — same bbox.json, COCOeval restricted to
# the complement of the negative-ontology classes (recorded in eval_metadata.json).
if [[ -f "${EVAL_METADATA_JSON}" ]]; then
  NO_NEG_SUBSET="$(python -c "import json,sys; m=json.load(open(sys.argv[1])); print(','.join(str(x) for x in m['no_negative_subset_cat_ids']))" "${EVAL_METADATA_JSON}")"
  if [[ -n "${NO_NEG_SUBSET}" ]]; then
    python tools/summarize_tct_ngc_v2_eval.py \
      --name base_no_negative \
      --gt-json "${GT_JSON}" \
      --pred-json "${PRED_JSON}" \
      --runtime-json "${RUNTIME_JSON}" \
      --out-json "${REPORT_DIR}/base_no_negative_summary.json" \
      --out-csv  "${REPORT_DIR}/base_no_negative_per_class.csv" \
      --cat-ids-subset "${NO_NEG_SUBSET}"
  else
    echo "[eval_tct_ngc_v2_base_8gpu.sh] no_negative_subset_cat_ids empty in ${EVAL_METADATA_JSON}; skipping no_negative view" >&2
  fi
else
  echo "[eval_tct_ngc_v2_base_8gpu.sh] ${EVAL_METADATA_JSON} missing; skipping no_negative view" >&2
fi
