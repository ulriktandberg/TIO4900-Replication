#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="onnymo"
REMOTE_HOST="solstorm-login.iot.ntnu.no"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_BASE='$USER_WORK/tio4900-replication'
RSYNC_SSH="ssh -F /dev/null -o PreferredAuthentications=keyboard-interactive,password -o KbdInteractiveAuthentication=yes -o PasswordAuthentication=yes -o PubkeyAuthentication=no -o IdentitiesOnly=yes -o BatchMode=no -o NumberOfPasswordPrompts=3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_ARTIFACTS="${REPO_DIR}/artifacts/hpc_runs"
LOCAL_LOGS="${REPO_DIR}/artifacts/logs"

mkdir -p "${LOCAL_ARTIFACTS}" "${LOCAL_LOGS}"

echo "[sync_from_solstorm] Pulling Solstrom outputs from ${REMOTE}:${REMOTE_BASE}"
echo "[sync_from_solstorm] Local artifacts: ${LOCAL_ARTIFACTS}"
echo "[sync_from_solstorm] Local logs: ${LOCAL_LOGS}"
echo "[sync_from_solstorm] Passwords and keys are handled by ssh; this script stores no credentials."

rsync -az \
  -e "${RSYNC_SSH}" \
  "${REMOTE}:${REMOTE_BASE}/artifacts/hpc_runs/" \
  "${LOCAL_ARTIFACTS}/"

rsync -az \
  -e "${RSYNC_SSH}" \
  "${REMOTE}:${REMOTE_BASE}/logs/" \
  "${LOCAL_LOGS}/"

echo "[sync_from_solstorm] Pull complete"
