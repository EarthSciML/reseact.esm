#!/bin/bash
# ===========================================================================
# profile_backend.sbatch -- decompose ONE chemistry step, on a chosen backend.
# ===========================================================================
# Runs tools/diag/profile_adjoint.jl (build + compile + per-step timing, no
# stages) so the per-step cost of ros_step / ros_vjp can be compared between
# XLA:CPU and XLA:GPU at the SAME grid. Every adjoint number in this repo to
# date is XLA:CPU; the GPU has never been tried.
#
#   sbatch --export=ALL,PB_BACKEND=cpu,PB_GRID=13x7x72 tools/diag/profile_backend.sbatch
#   sbatch --export=ALL,PB_BACKEND=gpu,PB_GRID=6x6x8 --gres=gpu:1 tools/diag/profile_backend.sbatch
# ===========================================================================
#SBATCH --job-name=pbench
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

PB_BACKEND=${PB_BACKEND:-cpu}
PB_GRID=${PB_GRID:-6x6x8}
IFS=x read -r NLON NLAT NLEV <<< "$PB_GRID"

export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export RESEACT_BACKEND=$PB_BACKEND
export RESEACT_NLON=$NLON RESEACT_NLAT=$NLAT RESEACT_NLEV=$NLEV
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-0}
export RESEACT_PROF_REPS=${RESEACT_PROF_REPS:-20}
export RESEACT_PROF_VJP=${RESEACT_PROF_VJP:-1}

echo "=== node $(hostname) backend=$PB_BACKEND grid=$PB_GRID cpus=$SLURM_CPUS_PER_TASK ==="
nvidia-smi -L 2>/dev/null || echo "(no GPU visible)"
echo "started $(date -Is)"

julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/profile_adjoint.jl

echo "finished $(date -Is)"
