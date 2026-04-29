#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${USER_WORK:-}" ]]; then
  echo "[run_parallel_full] ERROR: USER_WORK is not set. Run this from Solstrom login node." >&2
  exit 1
fi

BASE_DIR="${USER_WORK}/tio4900-replication"
REPO_DIR="${BASE_DIR}/repo"
LOG_DIR="${BASE_DIR}/logs"
ARTIFACT_ROOT="${BASE_DIR}/artifacts/hpc_runs"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[run_parallel_full] ERROR: repository not found at ${REPO_DIR}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${ARTIFACT_ROOT}"

RUN_GROUP_ID="${TIO4900_RUN_GROUP_ID:-parallel_$(date +%Y%m%d_%H%M%S)}"
N_MODELS="${TIO4900_N_MODELS:-100}"
K_TOP="${TIO4900_K_TOP:-10}"
TUNING_LEVEL="${TIO4900_TUNING_LEVEL:-standard}"
NODE_LIST="${TIO4900_NODES:-compute-5-4 compute-5-5 compute-5-6 compute-5-9 compute-5-10}"
SSH_OPTS=(
  -o HostbasedAuthentication=no
)

read -r -a NODES <<< "${NODE_LIST}"
LANES=(baselines_xgboost lightgbm model_configs_0 model_configs_1 model_configs_2)
LANE_FAMILIES=("baselines xgboost" "lightgbm" "model_configs:0/3" "model_configs:1/3" "model_configs:2/3")

if (( ${#NODES[@]} < ${#LANES[@]} )); then
  echo "[run_parallel_full] ERROR: need at least ${#LANES[@]} nodes in TIO4900_NODES." >&2
  echo "[run_parallel_full] Current TIO4900_NODES: ${NODE_LIST}" >&2
  exit 1
fi

quote_args() {
  printf "%q " "$@"
}

EXTRA_ARGS="$(quote_args "$@")"
MANIFEST="${LOG_DIR}/${RUN_GROUP_ID}_manifest.tsv"

family_threads() {
  case "$1" in
    baselines_xgboost|lightgbm) echo "${TIO4900_TREE_THREADS:-8}" ;;
    *) echo "${TIO4900_ANN_THREADS:-4}" ;;
  esac
}

{
  echo -e "run_group_id\tlane\tfamilies\tnode\tpid\tlog_file"
} > "${MANIFEST}"

echo "[run_parallel_full] Run group: ${RUN_GROUP_ID}"
echo "[run_parallel_full] Nodes: ${NODE_LIST}"
echo "[run_parallel_full] n_models=${N_MODELS} k_top=${K_TOP} tuning_level=${TUNING_LEVEL}"
echo "[run_parallel_full] Manifest: ${MANIFEST}"

for idx in "${!LANES[@]}"; do
  lane="${LANES[$idx]}"
  families="${LANE_FAMILIES[$idx]}"
  node="${NODES[$idx]}"
  threads="$(family_threads "${lane}")"
  log_file="${LOG_DIR}/nohup_${RUN_GROUP_ID}_${lane}.log"

  remote_cmd=$(cat <<EOF
set -e
cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}" "${ARTIFACT_ROOT}"
nohup env \
  TIO4900_TREE_N_JOBS="${threads}" \
  OMP_NUM_THREADS="${threads}" \
  OPENBLAS_NUM_THREADS="${threads}" \
  MKL_NUM_THREADS="${threads}" \
  VECLIB_MAXIMUM_THREADS="${threads}" \
  NUMEXPR_NUM_THREADS="${threads}" \
  BLIS_NUM_THREADS="${threads}" \
  PYTHONUNBUFFERED=1 \
  bash -lc 'set -e
    cd "${REPO_DIR}"
    for family_spec in ${families}; do
      family="\${family_spec%%:*}"
      shard="\${family_spec#*:}"
      if [[ "\${family_spec}" == "\${family}" ]]; then
        shard=""
      fi
      run_suffix="\${family}"
      shard_args=()
      if [[ -n "\${shard}" ]]; then
        run_suffix="\${family}_\${shard//\//_}"
        shard_args=(--config-shard "\${shard}")
      fi
      export TIO4900_RUN_ID="${RUN_GROUP_ID}_\${run_suffix}"
      echo "[run_parallel_full:${lane}] starting \${family_spec} at \$(date -Is)"
      bash scripts/hpc/run_family.sh "\${family}" \
        --n-models "${N_MODELS}" \
        --k-top "${K_TOP}" \
        --tuning-level "${TUNING_LEVEL}" \
        "\${shard_args[@]}" \
        ${EXTRA_ARGS}
      echo "[run_parallel_full:${lane}] finished \${family_spec} at \$(date -Is)"
    done' \
  > "${log_file}" 2>&1 < /dev/null &
echo \$!
EOF
)

  echo "[run_parallel_full] Launching ${lane} (${families}) on ${node} with ${threads} threads"
  pid="$(ssh "${SSH_OPTS[@]}" "${node}" "bash -lc $(printf "%q" "${remote_cmd}")")"
  echo -e "${RUN_GROUP_ID}\t${lane}\t${families}\t${node}\t${pid}\t${log_file}" >> "${MANIFEST}"
  echo "[run_parallel_full] ${lane}: node=${node} pid=${pid} log=${log_file}"
done

echo "[run_parallel_full] Launched ${#LANES[@]} lanes."
echo "[run_parallel_full] Check status with:"
echo "  bash scripts/hpc/parallel_status.sh ${RUN_GROUP_ID}"
