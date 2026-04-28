#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${USER_WORK:-}" ]]; then
  echo "[parallel_status] ERROR: USER_WORK is not set. Run this from Solstrom login node." >&2
  exit 1
fi

LOG_DIR="${USER_WORK}/tio4900-replication/logs"
RUN_GROUP_ID="${1:-${TIO4900_RUN_GROUP_ID:-}}"
SSH_OPTS=(
  -o HostbasedAuthentication=no
)

if [[ -z "${RUN_GROUP_ID}" ]]; then
  MANIFEST="$(ls -t "${LOG_DIR}"/parallel_*_manifest.tsv 2>/dev/null | head -n 1 || true)"
else
  MANIFEST="${LOG_DIR}/${RUN_GROUP_ID}_manifest.tsv"
fi

if [[ -z "${MANIFEST}" || ! -f "${MANIFEST}" ]]; then
  echo "[parallel_status] ERROR: manifest not found. Pass a run group id." >&2
  exit 1
fi

echo "[parallel_status] Manifest: ${MANIFEST}"
tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r run_group lane families node pid log_file; do
  echo
  echo "=== ${lane} | ${families} | ${node} ==="
  ssh "${SSH_OPTS[@]}" "${node}" "ps -p ${pid} -o pid,stat,etime,cmd || true; echo; test -f ${log_file} && tail -n 12 ${log_file} || echo 'log not found: ${log_file}'"
done
