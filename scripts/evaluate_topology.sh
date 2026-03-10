#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WIDTH="${WIDTH:-8}"
TOPOLOGY="${TOPOLOGY:-${REPO_DIR}/example_topologies/dependent_balanced_8.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-generated/eval}"
BACKEND="${BACKEND:-librelane}"
LIBRELANE_CMD="${LIBRELANE_CMD:-librelane}"
REGISTER_OUTPUTS="${REGISTER_OUTPUTS:-true}"
CLOCK_PERIOD="${CLOCK_PERIOD:-5.0}"
MILL_JOBS="${MILL_JOBS:-2}"

bash "${REPO_DIR}/mill" -i -j "${MILL_JOBS}" PrefixRLCore.runMain PrefixRLCore.EvaluateTopologyMain \
  --width "${WIDTH}" \
  --topology "${TOPOLOGY}" \
  --backend "${BACKEND}" \
  --output-root "${OUTPUT_ROOT}" \
  --repo-root "${REPO_DIR}" \
  --librelane-config "${REPO_DIR}/librelane_config.json" \
  --elaborate-script "${REPO_DIR}/scripts/elaborate_prefix_adder.sh" \
  --librelane-cmd "${LIBRELANE_CMD}" \
  --register-outputs "${REGISTER_OUTPUTS}" \
  --clock-period "${CLOCK_PERIOD}"
