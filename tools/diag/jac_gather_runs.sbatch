#!/bin/bash
#SBATCH --job-name=jacgru
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export JGR_GRID=${JGR_GRID:-6x6x8}
echo "=== node $(hostname) grid=$JGR_GRID ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 4 --heap-size-hint=80G tools/diag/jac_gather_runs.jl
echo "finished $(date -Is)"
