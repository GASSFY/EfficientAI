#!/usr/bin/env bash
# FP16 baseline multimodal evaluation for InternVL2-8B.
#
# Suite: mmmu_val / realworldqa / ai2d (tokens=16) + ocrbench (tokens=128)
#
# Writes:
#   - outputs_internvl/<run>/...
#   - log/internvl2_8b/fp16/run_*.log, status_*.txt, summary_*.{txt,csv}
#
# Usage:
#   bash scripts/run_internvl2_8b_fp16.sh
#   SMOKE=1 bash scripts/run_internvl2_8b_fp16.sh

set -u
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL2-8B}"
NET_NAME="$(basename "${MODEL_PATH}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs_internvl}"
CACHE_DIR="${CACHE_DIR:-./cache}"
NSAMPLES="${NSAMPLES:-128}"
GPU_ID="${GPU_ID:-0}"
DATASET_TYPE="${DATASET_TYPE:-text-vision}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
SMOKE="${SMOKE:-0}"
LIMIT_MULTIMODAL="${LIMIT_MULTIMODAL:-1.0}"

if [[ "${SMOKE}" == "1" ]]; then
  if [[ "${LIMIT_MULTIMODAL}" == "1.0" ]]; then
    LIMIT_MULTIMODAL="2"
  fi
fi

export API_TYPE="${API_TYPE:-dummy}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/log/internvl2_8b/fp16"
mkdir -p "${CACHE_DIR}" "${OUTPUT_ROOT}" "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/run_${STAMP}.log"
STATUS_FILE="${LOG_DIR}/status_${STAMP}.txt"
SUMMARY_TXT="${LOG_DIR}/summary_${STAMP}.txt"
SUMMARY_CSV="${LOG_DIR}/summary_${STAMP}.csv"
RESULT_MARKERS=()

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "========================================="
echo "InternVL2-8B FP16 eval  ${STAMP}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "attn_implementation=${ATTN_IMPLEMENTATION}"
echo "SMOKE=${SMOKE}  LIMIT_MULTIMODAL=${LIMIT_MULTIMODAL}"
echo "========================================="

: > "${STATUS_FILE}"
echo "config,task,status,exit_code,output_hint,result_line" > "${SUMMARY_CSV}"

record_status() {
  local name="$1"
  local code="$2"
  local hint="${3:-}"
  if [[ "${code}" -eq 0 ]]; then
    echo "OK   ${name}  (exit=${code}) ${hint}" | tee -a "${STATUS_FILE}"
  else
    echo "FAIL ${name}  (exit=${code}) ${hint}" | tee -a "${STATUS_FILE}"
  fi
}

eval_fp16_task() {
  # FP16 baseline: wbits=abits=16 skips MAS quantization in main.py.
  local task="$1"
  local max_new_tokens="$2"
  local cfg="FP16"
  local job="${cfg}_${task}"

  echo "========================================="
  echo "[EVAL] ${job}  max_new_tokens=${max_new_tokens}"
  echo "========================================="

  local tmp_out
  tmp_out="$(mktemp)"
  python main.py \
    --model "${MODEL_PATH}" \
    --mode train \
    --epochs 0 \
    --wbits 16 \
    --abits 16 \
    --dataset-type "${DATASET_TYPE}" \
    --nsamples "${NSAMPLES}" \
    --output_dir "${OUTPUT_ROOT}" \
    --symmetric \
    --group_size 0 \
    --cache_dir "${CACHE_DIR}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --tasks_multimodal "${task}" \
    --gen_max_new_tokens "${max_new_tokens}" \
    --limit_multimodal "${LIMIT_MULTIMODAL}" 2>&1 | tee "${tmp_out}"
  local code=${PIPESTATUS[0]}

  local out_hint
  out_hint="$(grep -oE 'output_dir is: [^ ]+' "${tmp_out}" | tail -1 | sed 's/.*output_dir is: //;s#/mas_parameters.pth[[:space:]]*##')"
  local result_line
  result_line="$(grep -E "tasks_multimodal:" "${tmp_out}" | tail -1 || true)"
  if [[ -z "${result_line}" && -n "${out_hint}" && -d "${out_hint}" ]]; then
    result_line="$(grep -hE 'tasks_multimodal:|ocrbench_accuracy|mmmu_acc|exact_match' "${out_hint}"/log_rank0_*.txt 2>/dev/null | tail -1 || true)"
  fi

  if [[ -n "${out_hint}" ]]; then
    RESULT_MARKERS+=("${out_hint}")
  fi

  local status_str="OK"
  [[ "${code}" -eq 0 ]] || status_str="FAIL"
  record_status "eval_${job}" "${code}" "${out_hint:-}"
  echo "${cfg},${task},${status_str},${code},${out_hint:-},\"${result_line}\"" >> "${SUMMARY_CSV}"
  rm -f "${tmp_out}"
  return "${code}"
}

eval_fp16_suite() {
  eval_fp16_task "mmmu_val" 16
  eval_fp16_task "realworldqa" 16
  eval_fp16_task "ai2d" 16
  eval_fp16_task "ocrbench" 128
}

if [[ "${SMOKE}" == "1" ]]; then
  echo "[SMOKE] FP16 ai2d probe only."
  eval_fp16_task "ai2d" 16
else
  echo "[PLAN] FP16 baseline suite"
  eval_fp16_suite
fi

{
  echo "InternVL2-8B FP16 summary  ${STAMP}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo ""
  echo "==== Job status ===="
  cat "${STATUS_FILE}"
  echo ""
  echo "==== Extracted results ===="
  if [[ ${#RESULT_MARKERS[@]} -gt 0 ]]; then
    for d in "${RESULT_MARKERS[@]}"; do
      echo "--- ${d} ---"
      grep -hE 'tasks_multimodal:|INFO \{' "${d}"/log_rank0_*.txt 2>/dev/null | tail -5 || true
    done
  fi
  echo ""
  echo "Also see CSV: ${SUMMARY_CSV}"
  echo "Full log: ${RUN_LOG}"
} | tee "${SUMMARY_TXT}"

echo "========================================="
echo "InternVL2-8B FP16 finished."
echo "Status : ${STATUS_FILE}"
echo "Summary: ${SUMMARY_TXT}"
echo "CSV    : ${SUMMARY_CSV}"
echo "Log    : ${RUN_LOG}"
echo "========================================="

if grep -q '^FAIL' "${STATUS_FILE}"; then
  exit 1
fi
exit 0
