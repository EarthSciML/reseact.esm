#!/bin/bash
#SBATCH --job-name=execovl
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=$PWD/run-model-jl
export RESEACT_BACKEND=cpu RESEACT_ADJ_JAC=sym RESEACT_ADJ_UJITTER=0
export RESEACT_NLON=${RESEACT_NLON:-13} RESEACT_NLAT=${RESEACT_NLAT:-7} RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_OVERLAP_REPS=${RESEACT_SWEEP_REPS:-10}

echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 16 --heap-size-hint=64G tools/diag/exec_overlap.jl
echo "finished $(date -Is)"
