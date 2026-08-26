#!/bin/bash
#SBATCH --job-name=picensus
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export PIC_GRIDS=${PIC_GRIDS:-6x6x8,9x9x8}

echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 4 --heap-size-hint=100G tools/diag/plan_index_census.jl
echo "finished $(date -Is)"
