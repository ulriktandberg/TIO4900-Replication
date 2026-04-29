#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/hpc/run_family.sh FAMILY [-- extra args]

Example:
  bash scripts/hpc/run_family.sh smoke --n-models 5 --k-top 2
USAGE
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

FAMILY="$1"
shift

if [[ -z "${USER_WORK:-}" ]]; then
  echo "[run_family] ERROR: USER_WORK is not set. Run this on a Solstrom compute node." >&2
  exit 1
fi

BASE_DIR="${USER_WORK}/tio4900-replication"
VENV_DIR="${BASE_DIR}/.venv"
LOG_DIR="${BASE_DIR}/logs"
ARTIFACT_ROOT="${BASE_DIR}/artifacts/hpc_runs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SYNCED_REPO_DIR="${BASE_DIR}/repo"

if [[ -d "${SYNCED_REPO_DIR}" ]]; then
  REPO_DIR="${SYNCED_REPO_DIR}"
else
  REPO_DIR="${SCRIPT_REPO_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[run_family] ERROR: virtual environment not found at ${VENV_DIR}" >&2
  echo "[run_family] Run bash scripts/hpc/setup_env.sh from the synced repository first." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${ARTIFACT_ROOT}"

find_usable_python() {
  local candidate
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        command -v "${candidate}"
        return 0
      fi
    fi
  done
  return 1
}

if command -v module >/dev/null 2>&1; then
  module load foss/2023b >/dev/null 2>&1 || true
  if ! find_usable_python >/dev/null 2>&1; then
    for python_module in Python/3.11.5-GCCcore-13.2.0 Python/3.11.5 $(module -t avail Python 2>&1 | sed 's/[[:space:]]//g' | grep -E '^Python/' | sort -Vr || true); do
      [[ -z "${python_module}" ]] && continue
      if module load "${python_module}" >/dev/null 2>&1 && find_usable_python >/dev/null 2>&1; then
        echo "[run_family] Loaded ${python_module}"
        break
      fi
    done
  fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${TIO4900_RUN_ID:-${FAMILY}_${TIMESTAMP}}"
RUN_ARTIFACT_DIR="${ARTIFACT_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "${RUN_ARTIFACT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export TIO4900_ARTIFACT_DIR="${RUN_ARTIFACT_DIR}"
export TIO4900_LOG_DIR="${LOG_DIR}"
export PYTHONUNBUFFERED=1

echo "[run_family] Family: ${FAMILY}"
echo "[run_family] Repository: ${REPO_DIR}"
echo "[run_family] Virtualenv: ${VENV_DIR}"
echo "[run_family] Artifacts: ${RUN_ARTIFACT_DIR}"
echo "[run_family] Log: ${LOG_FILE}"
echo "[run_family] Thread caps: OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"

{
  echo "[run_family] Started at $(date -Is)"
  echo "[run_family] Host: $(hostname)"
  echo "[run_family] Python: $(command -v python)"
  python --version
  echo "[run_family] Command: python -m experiments.run_core_models ${FAMILY} --artifacts-root ${ARTIFACT_ROOT} --run-id ${RUN_NAME} $*"
  python -m experiments.run_core_models "${FAMILY}" \
    --artifacts-root "${ARTIFACT_ROOT}" \
    --run-id "${RUN_NAME}" \
    "$@"
  echo "[run_family] Finished at $(date -Is)"
} 2>&1 | tee "${LOG_FILE}"

echo "[run_family] Complete"
