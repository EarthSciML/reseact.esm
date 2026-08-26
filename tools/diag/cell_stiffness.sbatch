#!/bin/bash
#SBATCH --job-name=cellstiff
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
export RESEACT_BACKEND=cpu RESEACT_ADJ_JAC=sym RESEACT_ADJ_UJITTER=0 RESEACT_ADJ_CLAMP=1
export RESEACT_NLON=${RESEACT_NLON:-13} RESEACT_NLAT=${RESEACT_NLAT:-7} RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_STIFF_NMACRO=${RESEACT_STIFF_NMACRO:-2}
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/cell_stiffness.jl
echo "finished $(date -Is)"
