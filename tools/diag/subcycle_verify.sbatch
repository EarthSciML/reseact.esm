#!/bin/bash
#SBATCH --job-name=subverify
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# NOTE ON --export: `sbatch --export=ALL,K=V` splits the value on COMMAS and
# silently mangles anything containing one -- which every ladder spec does. So
# export the variables in the SUBMITTING SHELL and submit with a bare
# `--export=ALL` (the default), which is what the ${VAR:-default} forms below
# are for.
set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=$PWD/run-model-jl
export RESEACT_BACKEND=cpu RESEACT_ADJ_JAC=sym RESEACT_ADJ_UJITTER=0 RESEACT_ADJ_CLAMP=1
export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_SUBCYCLE_LADDER=${RESEACT_SUBCYCLE_LADDER:-8,32,128,512}
export RESEACT_SUB_NW=${RESEACT_SUB_NW:-2}
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} ladder=${RESEACT_SUBCYCLE_LADDER} ==="
echo "started $(date -Is)"
# Julia thread count follows the allocation: -t 8 inside a 4-CPU
# allocation oversubscribes and makes every compile slower.
NT=${SLURM_CPUS_PER_TASK:-8}
HH=$(( ${SLURM_MEM_PER_NODE:-49152} / 1024 * 8 / 10 ))G
julia --project="$RESEACT_RXENV" -t "$NT" --heap-size-hint="$HH" tools/diag/subcycle_verify.jl
echo "finished $(date -Is)"
