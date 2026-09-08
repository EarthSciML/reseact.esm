#!/bin/bash
# ===========================================================================
# adjoint_devloop_conus.sbatch -- the CONUS-scale confirmation that the
# DIFFERENTIATED map runs as a static-CAP masked `@trace for` ON THE DEVICE.
# ===========================================================================
# RESEACT_ADJ_DEVLOOP=both sweeps the SAME forward pass TWICE -- host-lifted
# first, device frozen grid second -- and compares the two gradients component
# by component. Doing both in one process is what makes the comparison mean
# anything: same base point, same checkpoints, same recorded (t, dt) sequence,
# same theta, same forcing epochs. The only difference is where the composition
# of the inner steps happens.
#
# READ THE WALL TIME ACCORDINGLY. This is a PORTABILITY change, not a speed one.
# On XLA:CPU the host round-trip it removes is ~0.2% of the loop, and the masked
# (dead) iterations the frozen grid pays for are real work, so the device sweep
# is expected to be FLAT OR SLOWER here. The point is that the differentiated
# program is now one `stablehlo.while` instead of ~n_inner separate XLA calls
# driven from Julia -- which is the difference between 0.2% and a dominant cost
# on an A100/H100.
#
# MEMORY. Both loop shapes are compiled in one process: ssp_vjp + ros_vjp for
# the host arm AND the frozen-grid out/vjp programs for every cap bucket the
# window spans. That is the peak, not the sweep. 3 macro steps add nothing to it.
#
#   mkdir -p logs && sbatch tools/diag/adjoint_devloop_conus.sbatch
# ===========================================================================
#SBATCH --job-name=adj-devloop
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${RESEACT_REPO:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm}"

export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_NLON=${RESEACT_NLON:-13}
export RESEACT_NLAT=${RESEACT_NLAT:-7}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-3}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-1}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-1}
export RESEACT_ADJ_DEVLOOP=${RESEACT_ADJ_DEVLOOP:-both}
export RESEACT_LABEL=${RESEACT_LABEL:-conus-devloop-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-logs/devloop-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== devloop=$RESEACT_ADJ_DEVLOOP jac=$RESEACT_ADJ_JAC clamp=$RESEACT_ADJ_CLAMP nmacro=$RESEACT_ADJ_NMACRO ==="
echo "started $(date -Is)"

julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=96G tools/adjoint_gradient.jl

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
