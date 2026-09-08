#!/bin/bash
#SBATCH --job-name=aktbl
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/envs/buildperf}
export AKT_GRIDS=${AKT_GRIDS:-6x6x8,12x12x8,18x12x8,25x13x8,6x6x72,25x13x72}
export AKT_PARTS=${AKT_PARTS:-1}
echo "=== node $(hostname) env=$RESEACT_RXENV grids=$AKT_GRIDS parts=$AKT_PARTS ==="
free -g; echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 4 --heap-size-hint=120G tools/diag/aktbl_ladder.jl 2>&1 | tee logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.txt
echo "finished $(date -Is)"
