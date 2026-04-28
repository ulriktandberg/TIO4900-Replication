#!/usr/bin/env bash
set -euo pipefail

echo "[setup_env] Setting up Solstrom Python environment"

if [[ -z "${USER_WORK:-}" ]]; then
  echo "[setup_env] ERROR: USER_WORK is not set. Run this on Solstrom." >&2
  exit 1
fi

BASE_DIR="${USER_WORK}/tio4900-replication"
VENV_DIR="${BASE_DIR}/.venv"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQ_FILE="${REPO_DIR}/requirements.txt"
CONSTRAINTS_FILE="${SCRIPT_DIR}/constraints-solstorm.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[setup_env] ERROR: requirements.txt not found at ${REQ_FILE}" >&2
  echo "[setup_env] Run this script from the synced repository, usually ${BASE_DIR}/repo." >&2
  exit 1
fi

mkdir -p "${BASE_DIR}/logs" "${BASE_DIR}/artifacts"

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

try_module_load() {
  local module_name="$1"
  if module load "${module_name}" >/dev/null 2>&1; then
    echo "[setup_env] Loaded ${module_name}"
  else
    echo "[setup_env] WARNING: ${module_name} is not available; continuing with current environment"
  fi
}

if command -v module >/dev/null 2>&1; then
  echo "[setup_env] Loading modules when available"
  try_module_load "foss/2023b"
  if ! find_usable_python >/dev/null 2>&1; then
    echo "[setup_env] Searching available Python modules"
    module avail Python 2>&1 | sed -n '1,80p' || true

    PYTHON_MODULES=()
    while IFS= read -r module_name; do
      [[ -n "${module_name}" ]] && PYTHON_MODULES+=("${module_name}")
    done < <(module -t avail Python 2>&1 | sed 's/[[:space:]]//g' | grep -E '^Python/' | sort -Vr || true)

    # Prefer the advertised version, then try whatever the module system exposes.
    PYTHON_MODULES=("Python/3.11.5" "Python/3.11.5-GCCcore-13.2.0" "${PYTHON_MODULES[@]}")

    for python_module in "${PYTHON_MODULES[@]}"; do
      [[ -z "${python_module}" ]] && continue
      echo "[setup_env] Trying ${python_module}"
      if module load "${python_module}" >/dev/null 2>&1 && find_usable_python >/dev/null 2>&1; then
        echo "[setup_env] Loaded usable ${python_module}"
        break
      fi
    done
  fi
else
  echo "[setup_env] WARNING: environment modules are unavailable; continuing with current shell"
fi

PYTHON_BIN="$(find_usable_python || true)"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[setup_env] ERROR: no usable Python >= 3.10 is available after module setup" >&2
  echo "[setup_env] Run 'module avail Python' on Solstorm and share the output if this persists." >&2
  exit 1
fi

echo "[setup_env] Python bootstrap: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "[setup_env] Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[setup_env] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[setup_env] Installing requirements from ${REQ_FILE}"
if [[ -f "${CONSTRAINTS_FILE}" ]]; then
  echo "[setup_env] Applying Solstorm constraints from ${CONSTRAINTS_FILE}"
  python -m pip install --prefer-binary -r "${REQ_FILE}" -c "${CONSTRAINTS_FILE}"
else
  python -m pip install --prefer-binary -r "${REQ_FILE}"
fi

echo "[setup_env] Verifying Python and key packages"
python - <<'PY'
import importlib
import sys

packages = [
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "xgboost",
    "lightgbm",
    "statsmodels",
]

print(f"python={sys.executable}")
print(f"version={sys.version.split()[0]}")

missing = []
for package in packages:
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        missing.append((package, repr(exc)))
        continue
    version = getattr(module, "__version__", "unknown")
    print(f"{package}={version}")

if missing:
    print("Missing or broken packages:", file=sys.stderr)
    for package, error in missing:
        print(f"  {package}: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

echo "[setup_env] Environment ready: ${VENV_DIR}"
