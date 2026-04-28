#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${USER_WORK:-}" ]]; then
  echo "[tail_parallel_logs] ERROR: USER_WORK is not set. Run this on Solstrom." >&2
  exit 1
fi

LOG_DIR="${USER_WORK}/tio4900-replication/logs"
RUN_GROUP_ID="${1:-${TIO4900_RUN_GROUP_ID:-}}"
LINES="${TIO4900_LOG_LINES:-25}"

if [[ -z "${RUN_GROUP_ID}" ]]; then
  MANIFEST="$(ls -t "${LOG_DIR}"/parallel_*_manifest.tsv 2>/dev/null | head -n 1 || true)"
else
  MANIFEST="${LOG_DIR}/${RUN_GROUP_ID}_manifest.tsv"
fi

if [[ -z "${MANIFEST}" || ! -f "${MANIFEST}" ]]; then
  echo "[tail_parallel_logs] ERROR: manifest not found. Pass a run group id." >&2
  exit 1
fi

echo "[tail_parallel_logs] Manifest: ${MANIFEST}"
echo "[tail_parallel_logs] Showing last ${LINES} lines per lane, then following. Stop with Ctrl-C."

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

while IFS=$'\t' read -r _run_group lane families _node _pid log_file; do
  label="${lane}"
  if [[ "${lane}" != "${families}" ]]; then
    label="${lane}:${families}"
  fi
  if [[ ! -f "${log_file}" ]]; then
    echo "[${label}] waiting for ${log_file}"
  fi
  tail -n "${LINES}" -F "${log_file}" 2>/dev/null | sed -u "s/^/[${label}] /" &
  pids+=("$!")
done < <(tail -n +2 "${MANIFEST}")

wait
