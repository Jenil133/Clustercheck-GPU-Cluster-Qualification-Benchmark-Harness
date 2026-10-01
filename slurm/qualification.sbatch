#!/usr/bin/env bash
#SBATCH --job-name=clustercheck-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=clustercheck-node-%j.out
#SBATCH --error=clustercheck-node-%j.err
set -euo pipefail

: "${CLUSTERCHECK_CONFIG:?missing CLUSTERCHECK_CONFIG}"
: "${CLUSTERCHECK_OUTPUT:?missing CLUSTERCHECK_OUTPUT}"
: "${CLUSTERCHECK_RUN_ID:?missing CLUSTERCHECK_RUN_ID}"
: "${CLUSTERCHECK_NODES:?missing CLUSTERCHECK_NODES}"
: "${SLURMD_NODENAME:?missing Slurm node identity}"
node="${SLURMD_NODENAME}"
IFS=',' read -r -a configured_nodes <<< "${CLUSTERCHECK_NODES}"
found=0
for configured in "${configured_nodes[@]}"; do [[ "${configured}" == "${node}" ]] && found=1; done
(( found == 1 )) || { echo "allocated node is not in configured node set" >&2; exit 65; }
echo "ClusterCheck MEASURED hardware worker: run=${CLUSTERCHECK_RUN_ID} node=${node} job=${SLURM_JOB_ID}"
export CLUSTERCHECK_SKIP_NCCL=1
srun --nodes=1 --ntasks=1 clustercheck run-node --mode real --config "${CLUSTERCHECK_CONFIG}" --output "${CLUSTERCHECK_OUTPUT}" --run-id "${CLUSTERCHECK_RUN_ID}" --node "${node}"
