#!/bin/bash
#SBATCH --job-name=p8mof
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
echo "started $(date -Is) on $(hostname)"
julia --project="$RESEACT_RXENV" -t 8 tools/diag/p8_mof_probe.jl
echo "finished $(date -Is)"
