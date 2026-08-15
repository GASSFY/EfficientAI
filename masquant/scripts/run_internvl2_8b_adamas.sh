#!/usr/bin/env bash
# AdaMAS (modified MAS-Quant) train + multimodal eval for InternVL2-8B.
#
# Order: W8A8 -> W4A16 -> W4A8 -> W4A6  (each: train if missing, then 4-task suite)
# Suite: mmmu_val / realworldqa / ai2d (tokens=16) + ocrbench (tokens=128)
#
# New method flags (vs original MAS-Quant):
#   --auto_modal_weight
#   --adaptive_modal_scale
#
# Depends on:
#   - act_scales/<model>-<dataset-type>-<nsamples>.pt
#     python generate_act_scale_shift.py --model <MODEL> --dataset-type text-vision --nsamples 128
#
# Writes:
#   - outputs_internvl/<run>/...
#   - log/internvl2_8b/adamas/run_*.log, status_*.txt, summary_*.{txt,csv}
#
# Usage:
#   bash scripts/run_internvl2_8b_adamas.sh
#   SMOKE=1 bash scripts/run_internvl2_8b_adamas.sh

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
EPOCHS="${EPOCHS:-2}"
GPU_ID="${GPU_ID:-0}"
DATASET_TYPE="${DATASET_TYPE:-text-vision}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
OUTPUT_DIR_POSTFIX="${OUTPUT_DIR_POSTFIX:-adamas}"
SMOKE="${SMOKE:-0}"
LIMIT_MULTIMODAL="${LIMIT_MULTIMODAL:-1.0}"
SKIP_MISSING_TRAIN="${SKIP_MISSING_TRAIN:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  if [[ "${LIMIT_MULTIMODAL}" == "1.0" ]]; then
    LIMIT_MULTIMODAL="2"
  fi
  SKIP_MISSING_TRAIN=1
fi

export inference_mode="${inference_mode:-split_scales}"
export API_TYPE="${API_TYPE:-dummy}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/log/internvl2_8b/adamas"
mkdir -p "${CACHE_DIR}" "${OUTPUT_ROOT}" "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/run_${STAMP}.log"
STATUS_FILE="${LOG_DIR}/status_${STAMP}.txt"
SUMMARY_TXT="${LOG_DIR}/summary_${STAMP}.txt"
SUMMARY_CSV="${LOG_DIR}/summary_${STAMP}.csv"
RESULT_MARKERS=()

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "========================================="
echo "InternVL2-8B AdaMAS-Quant  ${STAMP}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "inference_mode=${inference_mode}"
echo "output_dir_postfix=${OUTPUT_DIR_POSTFIX}"
echo "attn_implementation=${ATTN_IMPLEMENTATION}"
echo "AdaMAS: auto_modal_weight + adaptive_modal_scale"
echo "SMOKE=${SMOKE}  LIMIT_MULTIMODAL=${LIMIT_MULTIMODAL}  SKIP_MISSING_TRAIN=${SKIP_MISSING_TRAIN}"
echo "========================================="

ACT_SCALES="./act_scales/${NET_NAME}-${DATASET_TYPE}-${NSAMPLES}.pt"
if [[ ! -f "${ACT_SCALES}" ]]; then
  echo "[FATAL] act_scales missing: ${ACT_SCALES}"
  echo "Run first:"
  echo "  python generate_act_scale_shift.py --model ${MODEL_PATH} --dataset-type ${DATASET_TYPE} --nsamples ${NSAMPLES}"
  exit 1
fi
echo "[OK] act_scales: ${ACT_SCALES}"

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

find_mas_ckpt() {
  local wbits="$1"
  local abits="$2"
  ls -dt "${OUTPUT_ROOT}/${NET_NAME}-${EPOCHS}epochs-w${wbits}a${abits}-${OUTPUT_DIR_POSTFIX}-"*-"${inference_mode}"/mas_parameters.pth 2>/dev/null | head -1
}

train_mas() {
  local wbits="$1"
  local abits="$2"
  local cfg="W${wbits}A${abits}"
  local existing
  existing="$(find_mas_ckpt "${wbits}" "${abits}" || true)"

  if [[ -n "${existing}" && -f "${existing}" ]]; then
    echo "[SKIP TRAIN] ${cfg} already has ${existing}"
    record_status "train_${cfg}" 0 "${existing}"
    return 0
  fi

  if [[ "${SKIP_MISSING_TRAIN}" == "1" ]]; then
    echo "[SKIP TRAIN] ${cfg} missing ckpt and SKIP_MISSING_TRAIN=1"
    record_status "train_${cfg}" 1 "skipped_missing"
    return 1
  fi

  echo "========================================="
  echo "[TRAIN] ${cfg}  epochs=${EPOCHS}  (AdaMAS)"
  echo "========================================="
  python main.py \
    --model "${MODEL_PATH}" \
    --mode train \
    --epochs "${EPOCHS}" \
    --wbits "${wbits}" \
    --abits "${abits}" \
    --let \
    --loss_multi_modal_mae_alpha \
    --auto_modal_weight \
    --adaptive_modal_scale \
    --dataset-type "${DATASET_TYPE}" \
    --nsamples "${NSAMPLES}" \
    --output_dir "${OUTPUT_ROOT}" \
    --output_dir_postfix "${OUTPUT_DIR_POSTFIX}" \
    --symmetric \
    --group_size 0 \
    --cache_dir "${CACHE_DIR}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}"
  local code=$?
  local ckpt
  ckpt="$(find_mas_ckpt "${wbits}" "${abits}" || true)"
  record_status "train_${cfg}" "${code}" "${ckpt:-none}"
  if [[ "${code}" -ne 0 || -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[ERROR] train ${cfg} failed or checkpoint not found"
    return 1
  fi
  echo "[OK TRAIN] ${cfg} -> ${ckpt}"
  return 0
}

eval_task() {
  local wbits="$1"
  local abits="$2"
  local resume="$3"
  local task="$4"
  local max_new_tokens="$5"
  local cfg="W${wbits}A${abits}"
  local job="${cfg}_${task}"

  if [[ -z "${resume}" || ! -f "${resume}" ]]; then
    echo "[ERROR] missing resume for ${job}: ${resume}"
    record_status "eval_${job}" 1 "missing_resume"
    echo "${cfg},${task},FAIL,1,missing_resume," >> "${SUMMARY_CSV}"
    return 1
  fi

  echo "========================================="
  echo "[EVAL] ${job}  max_new_tokens=${max_new_tokens}"
  echo "resume=${resume}"
  echo "========================================="

  local tmp_out
  tmp_out="$(mktemp)"
  python main.py \
    --model "${MODEL_PATH}" \
    --mode train \
    --epochs 0 \
    --wbits "${wbits}" \
    --abits "${abits}" \
    --let \
    --resume "${resume}" \
    --adaptive_modal_scale \
    --dataset-type "${DATASET_TYPE}" \
    --nsamples "${NSAMPLES}" \
    --output_dir "${OUTPUT_ROOT}" \
    --output_dir_postfix "${OUTPUT_DIR_POSTFIX}" \
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

eval_standard_suite() {
  local wbits="$1"
  local abits="$2"
  local resume="$3"
  eval_task "${wbits}" "${abits}" "${resume}" "mmmu_val" 16
  eval_task "${wbits}" "${abits}" "${resume}" "realworldqa" 16
  eval_task "${wbits}" "${abits}" "${resume}" "ai2d" 16
  eval_task "${wbits}" "${abits}" "${resume}" "ocrbench" 128
}

train_and_eval() {
  local wbits="$1"
  local abits="$2"
  local cfg="W${wbits}A${abits}"
  train_mas "${wbits}" "${abits}"
  local ckpt
  ckpt="$(find_mas_ckpt "${wbits}" "${abits}" || true)"
  if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
    echo "[PLAN] ${cfg} ckpt: ${ckpt}"
    eval_standard_suite "${wbits}" "${abits}" "${ckpt}"
  else
    echo "[ERROR] skip ${cfg} eval suite (no checkpoint)"
    record_status "eval_${cfg}_suite" 1 "no_ckpt"
  fi
}

if [[ "${SMOKE}" == "1" ]]; then
  echo "[SMOKE] W4A8 ocrbench probe only (needs existing AdaMAS ckpt)."
  W4A8_CKPT="${W4A8_CKPT:-$(find_mas_ckpt 4 8 || true)}"
  if [[ -z "${W4A8_CKPT}" ]]; then
    echo "[SMOKE] no W4A8 AdaMAS ckpt; skip (train with SMOKE=0 first)"
    record_status "eval_W4A8_ocrbench" 1 "missing_resume"
    echo "W4A8,ocrbench,FAIL,1,missing_resume," >> "${SUMMARY_CSV}"
  else
    echo "[PLAN] W4A8 ckpt: ${W4A8_CKPT}"
    eval_task 4 8 "${W4A8_CKPT}" "ocrbench" 128
  fi
else
  echo "[PLAN] AdaMAS configs: W8A8, W4A16, W4A8, W4A6"
  train_and_eval 8 8
  train_and_eval 4 16
  train_and_eval 4 8
  train_and_eval 4 6
fi

{
  echo "InternVL2-8B AdaMAS summary  ${STAMP}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "inference_mode=${inference_mode}"
  echo "method=AdaMAS (auto_modal_weight + adaptive_modal_scale)"
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
echo "InternVL2-8B AdaMAS finished."
echo "Status : ${STATUS_FILE}"
echo "Summary: ${SUMMARY_TXT}"
echo "CSV    : ${SUMMARY_CSV}"
echo "Log    : ${RUN_LOG}"
echo "========================================="

if grep -q '^FAIL' "${STATUS_FILE}"; then
  exit 1
fi
exit 0
