#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ALGORITHM="${ALGORITHM:-deep}"
BACKEND="${BACKEND:-synthetic}"
WIDTH="${WIDTH:-8}"
EPISODES="${EPISODES:-8}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-${REPO_DIR}/generated/search/${ALGORITHM}}"
RUN_PREFIX="${RUN_PREFIX:-sweep}"
SEEDS_LIST="${SEEDS_LIST:-1,2,3}"
TEMPERATURES_LIST="${TEMPERATURES_LIST:-1.0}"
LEARNING_RATES_LIST="${LEARNING_RATES_LIST:-}"
HIDDEN_SIZES_LIST="${HIDDEN_SIZES_LIST:-16,32,48}"
DAG_SUMMARY_MODES_LIST="${DAG_SUMMARY_MODES_LIST:-usage-weighted,attention-weighted}"
ACTION_CONTEXT_MODES_LIST="${ACTION_CONTEXT_MODES_LIST:-self-attention,mean-residual}"
BASELINE_MOMENTUM="${BASELINE_MOMENTUM:-0.9}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
REGISTER_OUTPUTS="${REGISTER_OUTPUTS:-true}"
CLOCK_PERIOD="${CLOCK_PERIOD:-5.0}"
LIBRELANE_CMD="${LIBRELANE_CMD:-librelane}"
MILL_JOBS="${MILL_JOBS:-2}"

csv_to_array() {
  local raw="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<<"${raw}"
  if [[ ${#out_ref[@]} -eq 0 ]]; then
    out_ref=("")
  fi
}

csv_to_array "${SEEDS_LIST}" seeds
csv_to_array "${TEMPERATURES_LIST}" temperatures
csv_to_array "${LEARNING_RATES_LIST}" learning_rates
csv_to_array "${HIDDEN_SIZES_LIST}" hidden_sizes
csv_to_array "${DAG_SUMMARY_MODES_LIST}" dag_modes
csv_to_array "${ACTION_CONTEXT_MODES_LIST}" action_modes

if [[ ${#learning_rates[@]} -eq 1 && -z "${learning_rates[0]}" ]]; then
  learning_rates=("")
fi

run_count=0

for seed in "${seeds[@]}"; do
  for temperature in "${temperatures[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      if [[ "${ALGORITHM}" == "tabular" ]]; then
        run_label="${RUN_PREFIX}_algo-${ALGORITHM}_backend-${BACKEND}_w-${WIDTH}_seed-${seed}_temp-${temperature}"
        if [[ -n "${learning_rate}" ]]; then
          run_label+="_lr-${learning_rate}"
        fi
        echo "[sweep] ${run_label}"
        ALGORITHM="${ALGORITHM}" BACKEND="${BACKEND}" WIDTH="${WIDTH}" EPISODES="${EPISODES}" SEED="${seed}" TEMPERATURE="${temperature}" LEARNING_RATE="${learning_rate}" BASELINE_MOMENTUM="${BASELINE_MOMENTUM}" CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" OUTPUT_ROOT="${OUTPUT_ROOT_BASE}" RUN_LABEL="${run_label}" CLEAN_OUTPUT="true" REGISTER_OUTPUTS="${REGISTER_OUTPUTS}" CLOCK_PERIOD="${CLOCK_PERIOD}" LIBRELANE_CMD="${LIBRELANE_CMD}" MILL_JOBS="${MILL_JOBS}" \
          bash "${SCRIPT_DIR}/run_search.sh"
        run_count=$((run_count + 1))
      else
        for hidden_size in "${hidden_sizes[@]}"; do
          for dag_mode in "${dag_modes[@]}"; do
            for action_mode in "${action_modes[@]}"; do
              run_label="${RUN_PREFIX}_algo-${ALGORITHM}_backend-${BACKEND}_w-${WIDTH}_seed-${seed}_temp-${temperature}_hs-${hidden_size}_dag-${dag_mode}_ctx-${action_mode}"
              if [[ -n "${learning_rate}" ]]; then
                run_label+="_lr-${learning_rate}"
              fi
              echo "[sweep] ${run_label}"
              ALGORITHM="${ALGORITHM}" BACKEND="${BACKEND}" WIDTH="${WIDTH}" EPISODES="${EPISODES}" SEED="${seed}" TEMPERATURE="${temperature}" LEARNING_RATE="${learning_rate}" HIDDEN_SIZE="${hidden_size}" DAG_SUMMARY_MODE="${dag_mode}" ACTION_CONTEXT_MODE="${action_mode}" BASELINE_MOMENTUM="${BASELINE_MOMENTUM}" CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" OUTPUT_ROOT="${OUTPUT_ROOT_BASE}" RUN_LABEL="${run_label}" CLEAN_OUTPUT="true" REGISTER_OUTPUTS="${REGISTER_OUTPUTS}" CLOCK_PERIOD="${CLOCK_PERIOD}" LIBRELANE_CMD="${LIBRELANE_CMD}" MILL_JOBS="${MILL_JOBS}" \
                bash "${SCRIPT_DIR}/run_search.sh"
              run_count=$((run_count + 1))
            done
          done
        done
      fi
    done
  done
done

echo "[sweep] completed ${run_count} runs under ${OUTPUT_ROOT_BASE}"
