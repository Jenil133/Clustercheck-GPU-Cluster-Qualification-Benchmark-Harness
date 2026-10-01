#!/usr/bin/env bash
#SBATCH --job-name=clustercheck-fabric
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=clustercheck-fabric-%j.out
#SBATCH --error=clustercheck-fabric-%j.err
set -euo pipefail

: "${CLUSTERCHECK_CONFIG:?missing CLUSTERCHECK_CONFIG}"
: "${CLUSTERCHECK_OUTPUT:?missing CLUSTERCHECK_OUTPUT}"
: "${CLUSTERCHECK_RUN_ID:?missing CLUSTERCHECK_RUN_ID}"
: "${CLUSTERCHECK_NODES:?missing CLUSTERCHECK_NODES}"
mapfile -t allocated < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
CLUSTERCHECK_ALLOCATED_NODES="$(IFS=,; echo "${allocated[*]}")"
export CLUSTERCHECK_ALLOCATED_NODES
echo "ClusterCheck measured allocation-wide NCCL sweep: run=${CLUSTERCHECK_RUN_ID} nodes=${CLUSTERCHECK_ALLOCATED_NODES}"
clustercheck run-fabric --config "${CLUSTERCHECK_CONFIG}" --run-dir "${CLUSTERCHECK_OUTPUT}/${CLUSTERCHECK_RUN_ID}"
