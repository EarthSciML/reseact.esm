#!/bin/bash
# ===========================================================================
# adjoint_conus_48h.sbatch -- the target run: d(mean surface O3)/d(theta) over
# 48 h of CONUS, all 49 runtime scalars in one backward sweep.
# ===========================================================================
# WHAT MAKES THIS DIFFERENT FROM EVERY PREVIOUS CONUS GRADIENT. The largest one
# on record is THREE macro steps -- about fifteen simulated minutes (slurm
# 10015169). This is 576 macro steps. Two things had to change first:
#
#   * blocker 2, the FD block Jacobian, closed by RESEACT_ADJ_JAC=sym. The
#     per-step dot-product identity went 1.8e-6 -> 5.3e-16 on the same base
#     point with the race workaround on (slurm 10042219 vs 10041660).
#   * blocker 1, the clamp. See tools/diag/adjoint_clamp_retest.sbatch. Set
#     RESEACT_ADJ_CLAMP below to whatever that retest licenses -- 0 differentiates
#     a map the production runner does not use, so 1 is the goal, not the default.
#
# COST, projected from measurement rather than guessed. At CONUS, 3 macro steps:
# forward 78.61 s (170 accepted inner steps), backward 225.51 s of which replay
# 73.30 s (32.5%) and VJPs 152.21 s (67.5%) at 0.8954 s/VJP. Those were measured
# at the JITTERED base point, which runs ~189 chemistry attempts per macro step
# against ~50 on a real trajectory, so scaling them straight up over-estimates.
# Expect roughly 1-2 h forward and 8-12 h backward. 36 h of walltime is 2-3x that,
# which is the right margin for the first run of its kind.
#
# UJITTER=0 -- ON PURPOSE, and it contradicts the advice in the driver header.
# That advice is about FINITE-DIFFERENCE VALIDATION: the default IC is exactly
# spatially uniform, which sits on the switching surface of every PPM limiter and
# makes forward, backward and central differences agree with each other while none
# of them is the derivative. This run computes a gradient, it does not FD-check
# one, and it wants the REAL trajectory. Only the very first macro step is at the
# uniform IC; by step 2 the state is not uniform and the kink is behind us.
#
# THE TAPE. 681 kB per macro step -> ~390 MB of checkpoints over 48 h, plus one
# macro step's inner tape at a time (the rest is rebuilt by fixed-sequence
# replay). No Griewank scheme is warranted and none is used.
#
#   mkdir -p logs && sbatch tools/diag/adjoint_conus_48h.sbatch
# ===========================================================================
#SBATCH --job-name=adj48
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Sized from measurement: the 3-macro-step CONUS run with jac=:sym peaked at
# MaxRSS 22.96 GB (slurm 10042579), and 576 steps adds only ~390 MB of
# checkpoints on top -- the peak is the ssp_vjp COMPILE, which does not scale
# with the window. 64G is ~2.8x the measured peak and schedules; 96G sat
# PENDING on (Resources), which on these nodes is a memory reservation problem
# rather than an idle-CPU one.
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

export RESEACT_RXENV=${RESEACT_RXENV:-/u/ctessum/reseact-esast-pin/env-sym}
export RESEACT_MODEL=${RESEACT_MODEL:-/u/ctessum/reseact-esast-pin/corpus-pin/reseact.esm/reseact.esm}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
[ -f "$RESEACT_MODEL" ] || { echo "FATAL: model $RESEACT_MODEL missing"; exit 1; }

export RESEACT_NLON=${RESEACT_NLON:-13}
export RESEACT_NLAT=${RESEACT_NLAT:-7}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-576}   # 576 * 300 s = 48 h
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-1}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-0}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-0}
export RESEACT_ADJ_OBJ=${RESEACT_ADJ_OBJ:-SuperFast.O3:surf}
export RESEACT_LABEL=${RESEACT_LABEL:-conus-adj48-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-logs/adj48-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== jac=$RESEACT_ADJ_JAC clamp=$RESEACT_ADJ_CLAMP ujitter=$RESEACT_ADJ_UJITTER nmacro=$RESEACT_ADJ_NMACRO ==="
free -g | head -2
echo "started $(date -Is)"

julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=48G tools/adjoint_gradient.jl

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
