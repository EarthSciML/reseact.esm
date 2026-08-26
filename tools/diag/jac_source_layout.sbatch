#!/bin/bash
#SBATCH --job-name=jacsrclayout
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export JSL_GRIDS=${JSL_GRIDS:-6x6x8}

echo "=== node $(hostname) grids=$JSL_GRIDS ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 2 --heap-size-hint=28G tools/diag/jac_source_layout.jl
echo "finished $(date -Is)"
