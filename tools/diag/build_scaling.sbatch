#!/bin/bash
#SBATCH --job-name=bldscale
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export BS_GRIDS=${BS_GRIDS:-6x6x8,9x9x8,12x12x8,18x12x8}
export BS_PARTS=${BS_PARTS:-1,2}

echo "=== node $(hostname) grids=$BS_GRIDS parts=$BS_PARTS ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 4 --heap-size-hint=100G tools/diag/build_scaling.jl
echo "finished $(date -Is)"
