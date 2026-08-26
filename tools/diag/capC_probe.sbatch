#!/bin/bash
#SBATCH --job-name=capC
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
export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/capC_probe.jl
echo "finished $(date -Is)"
