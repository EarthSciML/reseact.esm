#!/bin/bash
#SBATCH --job-name=axisbab
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
export RESEACT_RXENV=${AXISB_ENV:?set AXISB_ENV}
export AB_GRIDS=${AB_GRIDS:-18x12x8,36x24x8,13x7x72,25x13x72}
export AB_PARTS=${AB_PARTS:-1,2}
export AB_ARMS=${AB_ARMS:-base}
echo "=== node $(hostname) env=$RESEACT_RXENV grids=$AB_GRIDS arms=$AB_ARMS ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 4 --heap-size-hint=100G tools/diag/axisb_ab.jl
echo "finished $(date -Is)"
