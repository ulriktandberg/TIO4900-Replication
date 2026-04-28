# Solstrom HPC Helpers

These scripts sync the repository to Solstrom, create a Python environment under `$USER_WORK`, run model families on compute nodes, and pull logs/artifacts back to the local repository.

## Paths

- Remote base: `$USER_WORK/tio4900-replication`
- Remote repo: `$USER_WORK/tio4900-replication/repo`
- Remote virtualenv: `$USER_WORK/tio4900-replication/.venv`
- Remote logs: `$USER_WORK/tio4900-replication/logs`
- Remote artifacts: `$USER_WORK/tio4900-replication/artifacts/hpc_runs`
- Local pulled artifacts: `artifacts/hpc_runs`
- Local pulled logs: `artifacts/logs`

## 1. Sync Code To Solstrom

Run from the local repository:

```bash
bash scripts/hpc/sync_to_solstorm.sh
```

The script syncs to:

```text
onnymo@solstorm-login.iot.ntnu.no:$USER_WORK/tio4900-replication/repo
```

It excludes `.git`, virtual environments, Python caches, notebook checkpoints, logs, and artifacts. It does not store passwords.

## 2. Set Up The Remote Environment

SSH to Solstrom and run setup from the synced repo:

```bash
ssh -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no -o IdentitiesOnly=yes onnymo@solstorm-login.iot.ntnu.no
cd "$USER_WORK/tio4900-replication/repo"
bash scripts/hpc/setup_env.sh
```

Use this same `ssh -o ...` form for manual login if your local machine reports `Too many authentication failures`; it prevents macOS/SSH-agent from trying unrelated saved keys before the password prompt:

```bash
ssh -o PreferredAuthentications=keyboard-interactive,password -o PubkeyAuthentication=no -o IdentitiesOnly=yes onnymo@solstorm-login.iot.ntnu.no
```

`setup_env.sh` attempts to load:

```bash
module load foss/2023b
module load Python/3.11.5
```

If either module is unavailable, the script continues with the current environment and reports a warning. It creates `$USER_WORK/tio4900-replication/.venv`, installs `requirements.txt`, and verifies Python plus key packages.

## 3. Start A Compute Node Session

Use compute nodes for runs. Prefer `compute-5-*`. If those are unavailable, use `compute-4-0` through `compute-4-49`. Avoid `compute-2` except for smoke tests. Do not use `compute-4-50` through `compute-4-58` or `compute-6-*`. Use at most 5 nodes.

Solstrom does not use a Slurm queue for these jobs. From the login node, inspect load and SSH to an idle allowed node:

```bash
gstat -a
ssh compute-5-4
top
```

Fallback example:

```bash
ssh compute-4-0
top
```

## 4. Run A Model Family

From the compute node:

```bash
cd "$USER_WORK/tio4900-replication/repo"
bash scripts/hpc/run_family.sh smoke --n-models 5 --k-top 2
```

If `screen` is not installed, use `tmux` if available:

```bash
tmux new -s tio4900-smoke
cd "$USER_WORK/tio4900-replication/repo"
bash scripts/hpc/run_family.sh smoke --n-models 5 --k-top 2
```

Detach from `tmux` with `Ctrl-b` then `d`, and reattach with:

```bash
tmux attach -t tio4900-smoke
```

If neither `screen` nor `tmux` exists, use `nohup`:

```bash
cd "$USER_WORK/tio4900-replication/repo"
nohup bash scripts/hpc/run_family.sh smoke --n-models 5 --k-top 2 > "$USER_WORK/tio4900-replication/logs/nohup_smoke.log" 2>&1 &
tail -f "$USER_WORK/tio4900-replication/logs/nohup_smoke.log"
```

General form:

```bash
bash scripts/hpc/run_family.sh FAMILY [additional arguments]
```

The script activates `$USER_WORK/tio4900-replication/.venv`, sets conservative thread limits, creates a run artifact directory under `$USER_WORK/tio4900-replication/artifacts/hpc_runs`, and writes a log under `$USER_WORK/tio4900-replication/logs`.

The executed command is:

```bash
python -m experiments.run_core_models FAMILY [additional arguments]
```

The current repository must contain the `experiments.run_core_models` module.

## 5. Run Everything In Parallel On Separate Nodes

For the fastest full run, launch one model family per compute node from the Solstrom login node:

```bash
cd "$USER_WORK/tio4900-replication/repo"
bash scripts/hpc/run_parallel_full.sh
```

Default node assignment:

```text
compute-5-4   baselines, then xgboost
compute-5-5   lightgbm
compute-5-6   model_configs shard 0/3
compute-5-9   model_configs shard 1/3
compute-5-10  model_configs shard 2/3
```

Default tuning/run settings:

```text
TIO4900_N_MODELS=100
TIO4900_K_TOP=10
TIO4900_TUNING_LEVEL=standard
TIO4900_BASELINE_THREADS=4
TIO4900_TREE_THREADS=8
TIO4900_ANN_THREADS=4
```

`model_configs` runs the configurations from `models/model_configs/fred_and_realtime_models.py`.
ANN outputs report both validation-loss top-k ensembling and trailing-OOS top-k ensembling.

The launcher writes a manifest under `$USER_WORK/tio4900-replication/logs`.
Check progress with the run group id printed by the launcher:

```bash
bash scripts/hpc/parallel_status.sh parallel_YYYYMMDD_HHMMSS
```

Use different allowed nodes if one of the defaults is busy:

```bash
TIO4900_NODES="compute-5-11 compute-5-12 compute-5-13 compute-5-14 compute-5-15" \
  bash scripts/hpc/run_parallel_full.sh
```

Use a smaller test run:

```bash
TIO4900_N_MODELS=20 TIO4900_K_TOP=5 TIO4900_TUNING_LEVEL=light \
  bash scripts/hpc/run_parallel_full.sh
```

## 6. Pull Outputs Back

Run from the local repository:

```bash
bash scripts/hpc/sync_from_solstorm.sh
```

Outputs are copied into:

```text
artifacts/hpc_runs
artifacts/logs
```
