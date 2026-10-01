#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 CONFIG OUTPUT BASELINE RUN_ID [--apply-quarantine]" >&2
  exit 64
fi
config="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
mkdir -p "$2"
output="$(cd "$2" && pwd)"
baseline="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
run_id="$4"
apply=0
[[ "${5:-}" == "--apply-quarantine" ]] && apply=1
[[ -r "${config}" && -r "${baseline}" ]] || { echo "config and baseline must be readable" >&2; exit 66; }
for value in "${config}" "${output}" "${baseline}"; do
  [[ "${value}" != *$'\n'* && "${value}" != *','* ]] || { echo "paths may not contain commas or newlines" >&2; exit 65; }
done
run_dir="${output}/${run_id}"

# Read settings without mapfile so the script also runs under bash 3.2.
settings_raw="$(python3 - "${config}" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
print(",".join(config["cluster"]["nodes"]))
print(config["cluster"]["gpus_per_node"])
print(config["cluster"].get("slurm_partition", ""))
PY
)"
nodes_csv=""
gpus_per_node=""
partition=""
{
  IFS= read -r nodes_csv || true
  IFS= read -r gpus_per_node || true
  IFS= read -r partition || true
} <<< "${settings_raw}"
[[ -n "${nodes_csv}" && -n "${gpus_per_node}" ]] || { echo "could not read cluster topology from config" >&2; exit 65; }
IFS=',' read -r -a nodes <<< "${nodes_csv}"

clustercheck initialize-run --mode real --state submitting --config "${config}" --output "${output}" --run-id "${run_id}" >/dev/null
job_ids=()
committed=0
cleanup() {
  status=$?
  if (( committed == 0 )); then
    if ((${#job_ids[@]})); then
      scancel "${job_ids[@]}" >/dev/null 2>&1 || true
    fi
    clustercheck submission-state --run-dir "${run_dir}" --state failed --error "submission transaction failed (exit ${status}); accepted jobs were cancelled" >/dev/null 2>&1 || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export CLUSTERCHECK_CONFIG="${config}" CLUSTERCHECK_OUTPUT="${output}" CLUSTERCHECK_BASELINE="${baseline}"
export CLUSTERCHECK_RUN_ID="${run_id}" CLUSTERCHECK_NODES="${nodes_csv}" CLUSTERCHECK_APPLY_QUARANTINE="${apply}"
exports="CLUSTERCHECK_CONFIG,CLUSTERCHECK_OUTPUT,CLUSTERCHECK_BASELINE,CLUSTERCHECK_RUN_ID,CLUSTERCHECK_NODES,CLUSTERCHECK_APPLY_QUARANTINE"
for name in PEER_HOST IB_DEVICE IB_PORT CLUSTERCHECK_IB_BASE_PORT CLUSTERCHECK_IB_PORT CLUSTERCHECK_IB_SERVER_DELAY; do
  if [[ -n "${!name:-}" ]]; then
    [[ "${!name}" != *$'\n'* && "${!name}" != *','* ]] || { echo "${name} may not contain commas or newlines" >&2; exit 65; }
    export "${name}"
    exports+=",${name}"
  fi
done
common=(--parsable --hold --gpus-per-node="${gpus_per_node}" --export="${exports}")
[[ -n "${partition}" ]] && common+=(--partition="${partition}")
submit_id() {
  local id
  id="$(sbatch "$@")"
  [[ "${id}" =~ ^[0-9]+$ ]] || { echo "invalid sbatch job ID: ${id}" >&2; return 1; }
  printf '%s' "${id}"
}
worker_ids=()
for node in "${nodes[@]}"; do
  id="$(submit_id "${common[@]}" --nodelist="${node}" "$(dirname "$0")/qualification.sbatch")"
  worker_ids+=("${id}"); job_ids+=("${id}")
  clustercheck submission-state --run-dir "${run_dir}" --state submitting --worker-node "${node}" --worker-job-id "${id}" >/dev/null
done
worker_dependency="$(IFS=:; echo "${worker_ids[*]}")"
fabric="$(submit_id "${common[@]}" --dependency="afterany:${worker_dependency}" --nodes="${#nodes[@]}" --nodelist="${nodes_csv}" "$(dirname "$0")/fabric.sbatch")"
job_ids+=("${fabric}")
clustercheck submission-state --run-dir "${run_dir}" --state submitting --fabric-job-id "${fabric}" >/dev/null
finalizer="$(submit_id --parsable --hold --dependency="afterany:${fabric}" --export="${exports}" "$(dirname "$0")/admission.sbatch")"
job_ids+=("${finalizer}")
clustercheck submission-state --run-dir "${run_dir}" --state submitted --admission-job-id "${finalizer}" >/dev/null
scontrol release "${job_ids[@]}"
committed=1
trap - EXIT INT TERM
printf 'submitted measured node jobs %s, %s-node NCCL job %s, and admission job %s for run %s\n' "$(IFS=,; echo "${worker_ids[*]}")" "${#nodes[@]}" "${fabric}" "${finalizer}" "${run_id}"
