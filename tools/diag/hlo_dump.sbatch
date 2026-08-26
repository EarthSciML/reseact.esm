#!/bin/bash
#SBATCH --job-name=hlodump
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

HD_GRID=${HD_GRID:-6x6x8}
IFS=x read -r NLON NLAT NLEV <<< "$HD_GRID"

export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export RESEACT_BACKEND=${RESEACT_BACKEND:-cpu}
export RESEACT_NLON=$NLON RESEACT_NLAT=$NLAT RESEACT_NLEV=$NLEV
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_UJITTER=0

echo "=== node $(hostname) grid=$HD_GRID cpus=$SLURM_CPUS_PER_TASK ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/hlo_dump.jl
echo "finished $(date -Is)"
