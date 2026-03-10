#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MILL="${REPO_DIR}/mill"
PROJECT="${PROJECT:-TopLevelModule}"
WIDTH=""
TOPOLOGY=""
REGISTER_OUTPUTS="true"
TARGET_DIR=""
MILL_JOBS="${MILL_JOBS:-2}"

usage() {
  cat <<USAGE
Usage:
  bash ./scripts/elaborate_prefix_adder.sh --width <bits> --topology <path> [--register-outputs true|false] [--target-dir <dir>]

Examples:
  bash ./scripts/elaborate_prefix_adder.sh --width 8 --topology example_topologies/dependent_balanced_8.json
  bash ./scripts/elaborate_prefix_adder.sh --width 16 --topology generated/search/designs/ep00010_abcd1234/topology.json --target-dir generated/verilog/PrefixAdderMacro_16
USAGE
}

need_value() {
  local opt="$1"
  if [[ $# -lt 2 || -z "${2-}" ]]; then
    echo "Missing value for ${opt}" >&2
    usage
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --width)
    need_value "$1" "$@"
    WIDTH="$2"
    shift 2
    ;;
  --topology)
    need_value "$1" "$@"
    TOPOLOGY="$2"
    shift 2
    ;;
  --register-outputs)
    need_value "$1" "$@"
    REGISTER_OUTPUTS="$2"
    shift 2
    ;;
  --target-dir)
    need_value "$1" "$@"
    TARGET_DIR="$2"
    shift 2
    ;;
  --help | -h)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage
    exit 1
    ;;
  esac
done

if [[ -z "${WIDTH}" || -z "${TOPOLOGY}" ]]; then
  echo "Both --width and --topology are required." >&2
  usage
  exit 1
fi

if [[ -z "${TARGET_DIR}" ]]; then
  TARGET_DIR="${REPO_DIR}/generated/verilog/PrefixAdderMacro_${WIDTH}"
fi

mkdir -p "${TARGET_DIR}"

ESCAPED_TOPOLOGY="${TOPOLOGY//\\/\\\\}"
ESCAPED_TOPOLOGY="${ESCAPED_TOPOLOGY//\"/\\\"}"
MODULE_SPEC="TopLevelModule.PrefixAdderMacro(${WIDTH}, ${REGISTER_OUTPUTS}, \"${ESCAPED_TOPOLOGY}\")"

bash "${MILL}" -i -j "${MILL_JOBS}" "${PROJECT}.runMain" Elaborate "${MODULE_SPEC}" --target-dir "${TARGET_DIR}"

echo "Generated SystemVerilog in: ${TARGET_DIR}"
