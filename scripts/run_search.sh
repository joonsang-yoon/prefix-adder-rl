#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ALGORITHM="${ALGORITHM:-tabular}"
WIDTH="${WIDTH:-8}"
EPISODES="${EPISODES:-16}"
SEED="${SEED:-1}"
LEARNING_RATE="${LEARNING_RATE:-}"
TEMPERATURE="${TEMPERATURE:-1.0}"
HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
GRADIENT_CLIP="${GRADIENT_CLIP:-5.0}"
BASELINE_MOMENTUM="${BASELINE_MOMENTUM:-0.9}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
POLICY_INIT="${POLICY_INIT:-}"
RUN_LABEL="${RUN_LABEL:-}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-true}"
SKIP_WARM_STARTS="${SKIP_WARM_STARTS:-false}"
DAG_SUMMARY_MODE="${DAG_SUMMARY_MODE:-usage-weighted}"
ACTION_CONTEXT_MODE="${ACTION_CONTEXT_MODE:-self-attention}"
OUTPUT_ROOT="${OUTPUT_ROOT:-generated/search/${ALGORITHM}}"
BACKEND="${BACKEND:-librelane}"
LIBRELANE_CMD="${LIBRELANE_CMD:-librelane}"
REGISTER_OUTPUTS="${REGISTER_OUTPUTS:-true}"
CLOCK_PERIOD="${CLOCK_PERIOD:-5.0}"
MILL_JOBS="${MILL_JOBS:-2}"

case "${ALGORITHM}" in
tabular)
  : "${LEARNING_RATE:=0.08}"
  TARGET="PrefixTabularRL.runMain"
  MAIN_CLASS="PrefixTabularRL.PrefixSearchMain"
  EXTRA_ARGS=(
    --seed "${SEED}"
    --learning-rate "${LEARNING_RATE}"
    --temperature "${TEMPERATURE}"
  )
  ;;
deep)
  : "${LEARNING_RATE:=0.02}"
  TARGET="PrefixDeepRL.runMain"
  MAIN_CLASS="PrefixDeepRL.PrefixSearchMain"
  EXTRA_ARGS=(
    --seed "${SEED}"
    --learning-rate "${LEARNING_RATE}"
    --temperature "${TEMPERATURE}"
    --hidden-size "${HIDDEN_SIZE}"
    --gradient-clip "${GRADIENT_CLIP}"
    --dag-summary-mode "${DAG_SUMMARY_MODE}"
    --action-context-mode "${ACTION_CONTEXT_MODE}"
  )
  ;;
*)
  echo "Unsupported ALGORITHM='${ALGORITHM}'. Expected 'tabular' or 'deep'." >&2
  exit 1
  ;;
esac

bash "${REPO_DIR}/mill" -i -j "${MILL_JOBS}" "${TARGET}" "${MAIN_CLASS}" \
  --width "${WIDTH}" \
  --episodes "${EPISODES}" \
  "${EXTRA_ARGS[@]}" \
  --baseline-momentum "${BASELINE_MOMENTUM}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --policy-init "${POLICY_INIT}" \
  --run-label "${RUN_LABEL}" \
  --clean-output "${CLEAN_OUTPUT}" \
  --skip-warm-starts "${SKIP_WARM_STARTS}" \
  --backend "${BACKEND}" \
  --output-root "${OUTPUT_ROOT}" \
  --repo-root "${REPO_DIR}" \
  --librelane-config "${REPO_DIR}/librelane_config.json" \
  --elaborate-script "${REPO_DIR}/scripts/elaborate_prefix_adder.sh" \
  --librelane-cmd "${LIBRELANE_CMD}" \
  --register-outputs "${REGISTER_OUTPUTS}" \
  --clock-period "${CLOCK_PERIOD}"
