#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="onnymo"
REMOTE_HOST="solstorm-login.iot.ntnu.no"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_BASE='$USER_WORK/tio4900-replication'
REMOTE_REPO="${REMOTE_BASE}/repo"
SSH_OPTS=(
  -o PreferredAuthentications=keyboard-interactive,password
  -o KbdInteractiveAuthentication=yes
  -o PasswordAuthentication=yes
  -o PubkeyAuthentication=no
  -o IdentitiesOnly=yes
  -o BatchMode=no
  -o NumberOfPasswordPrompts=3
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[sync_to_solstorm] Syncing local repository to ${REMOTE}:${REMOTE_REPO}"
echo "[sync_to_solstorm] Local repository: ${REPO_DIR}"
echo "[sync_to_solstorm] Passwords and keys are handled by ssh; this script stores no credentials."

echo "[sync_to_solstorm] Using tar-over-ssh upload because password auth on rsync can be unreliable on this cluster."
echo "[sync_to_solstorm] You will be prompted for the Solstorm password once."

tar \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude '.ruff_cache' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'artifacts' \
  --exclude 'logs' \
  --exclude '*.pyc' \
  -czf - \
  -C "${REPO_DIR}" . | \
ssh -F /dev/null "${SSH_OPTS[@]}" "${REMOTE}" \
  'set -e; mkdir -p "$USER_WORK/tio4900-replication/repo" "$USER_WORK/tio4900-replication/logs" "$USER_WORK/tio4900-replication/artifacts/hpc_runs"; tar -xzf - -C "$USER_WORK/tio4900-replication/repo"'

echo "[sync_to_solstorm] Sync complete"
echo "[sync_to_solstorm] Remote repo: ${REMOTE_REPO}"
