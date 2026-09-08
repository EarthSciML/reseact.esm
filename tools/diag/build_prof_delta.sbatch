#!/bin/bash
#SBATCH --job-name=bpd
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
export BPD_GRIDS=${BPD_GRIDS:-6x6x8,25x13x72}
export BPD_PART=${BPD_PART:-1}
export BPD_TOP=${BPD_TOP:-45}
echo "=== node $(hostname) env=$RESEACT_RXENV grids=$BPD_GRIDS part=$BPD_PART ==="
free -g; echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 4 --heap-size-hint=120G tools/diag/build_prof_delta.jl 2>&1 | tee logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.txt
echo "finished $(date -Is)"
