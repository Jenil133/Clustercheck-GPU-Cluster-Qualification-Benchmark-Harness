#!/usr/bin/env bash
#SBATCH --job-name=clustercheck-admission
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=clustercheck-admission-%j.out
#SBATCH --error=clustercheck-admission-%j.err
set -euo pipefail

: "${CLUSTERCHECK_CONFIG:?missing CLUSTERCHECK_CONFIG}"
: "${CLUSTERCHECK_OUTPUT:?missing CLUSTERCHECK_OUTPUT}"
: "${CLUSTERCHECK_RUN_ID:?missing CLUSTERCHECK_RUN_ID}"
: "${CLUSTERCHECK_BASELINE:?missing CLUSTERCHECK_BASELINE}"
run_dir="${CLUSTERCHECK_OUTPUT}/${CLUSTERCHECK_RUN_ID}"
clustercheck finalize-run --config "${CLUSTERCHECK_CONFIG}" --run-dir "${run_dir}"
args=(evaluate --config "${CLUSTERCHECK_CONFIG}" --run-dir "${run_dir}" --baseline "${CLUSTERCHECK_BASELINE}")
if [[ "${CLUSTERCHECK_APPLY_QUARANTINE:-0}" == "1" ]]; then args+=(--apply-quarantine); fi
clustercheck "${args[@]}"
