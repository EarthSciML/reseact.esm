#!/bin/bash
#SBATCH --job-name=p8ppm
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
echo "started $(date -Is) on $(hostname)"
julia --project="$RESEACT_RXENV" -t 8 tools/diag/p8_ppm_reverse.jl
python3 tools/diag/p5_fusion_census.py logs/p8-ppm-$SLURM_JOB_ID rhs_ifelse grad_ifelse rhs_barf grad_barf rhs_bara grad_bara
echo "finished $(date -Is)"
