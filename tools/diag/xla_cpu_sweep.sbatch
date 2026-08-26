#!/bin/bash
#SBATCH --job-name=xlasweep
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
export RESEACT_SWEEP_REPS=${RESEACT_SWEEP_REPS:-10}
export RESEACT_SWEEP_DUMP=${RESEACT_SWEEP_DUMP:-$PWD/logs/hlodump-$SLURM_JOB_ID}
echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/xla_cpu_sweep.jl
echo "finished $(date -Is)"
