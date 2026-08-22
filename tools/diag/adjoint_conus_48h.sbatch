#!/bin/bash
# ===========================================================================
# adjoint_conus_48h.sbatch -- the target run: d(mean surface O3)/d(theta) over
# 48 h of CONUS, every runtime scalar in the model in ONE backward sweep.
# ===========================================================================
# WHAT MAKES THIS DIFFERENT FROM EVERY PREVIOUS CONUS GRADIENT. Before it first
# ran, the largest CONUS gradient on record was THREE macro steps -- about fifteen
# simulated minutes (slurm 10015169). This is 576. Two things had to change first:
#
#   * blocker 2, the FD block Jacobian, closed by RESEACT_ADJ_JAC=sym. The
#     per-step dot-product identity went 1.8e-6 -> 5.3e-16 on the same base
#     point with the race workaround on (slurm 10042219 vs 10041660).
#   * blocker 1, the clamp -- which was never about the clamp. clamp_nonneg
#     writes exact zeros, an FD Jacobian's step at u=0 is sqrt(eps)*|u| = 0, and
#     the XLA:CPU race supplied the nondeterminism that made the resulting
#     non-finite sweep irreproducible. With jac=:sym it is simply gone, at scale:
#     see the run below. CLAMP=1 is therefore the default here, not the goal.
#
# WHAT IT COST, MEASURED (slurm 10044327, esm 0.8.0, 8 h 17 m 55 s, MaxRSS 55.2 GB):
# BUILD 693.20 s, prepare_jacobian 592.0 s, compiles 140.4 + 126.4 + 572.7 + 156.0 s.
# Forward 7,774.93 s over 576 macro steps / 27,973 accepted inner steps, checkpoints
# 374.3 MB. Backward 19,273.16 s = replay 6,372.10 s (33%) + VJPs 12,901.06 s at
# 0.4612 s each. Ratio 2.48 (1.66 VJP-only) -- NOT the 5.1x quoted since Phase 4,
# which was a demonstration-scale number. J = 30.1943301698541 ppb, structural
# identity 8.611e-15, replay 0.000e+00 at all 576 checkpoints, 0 retries in 27,973
# VJP calls, clamp fired 50,973 times over 2.38e9 pairs, 19 of 49 components nonzero.
#
# IT HAS RUN TWICE. 10044327 was esm 0.8.0 out of the hand-built env-sym against
# a corpus-pin copy of the model, because run-model-jl could not then resolve
# EarthSciASTDiff. 10055533 is this file as it stands: esm 1.0.0, plain
# run-model-jl, 8 h 01 m 59 s, MaxRSS 38.4 GB, forward 7,460.23 s, backward
# 18,761.89 s at 0.4585 s/VJP, ratio 2.51 (VJP-only 1.72), J = 30.1943301698564.
#
# The pair is the only cross-version check this gradient has, and it is a strong
# one: the 576-step accept/reject ladder is BYTE-IDENTICAL, so the controller made
# all 27,973 decisions the same way. J moved 647 ulps (7.6e-14) and the 19 nonzero
# components agree to 2.8e-10 worst-case. That is reassociation, not a schema
# difference -- 1.0.0 exposes 111 setup-time geometry constants as runtime scalars
# that XLA can no longer constant-fold, and all 111 came back EXACTLY zero. The
# same comparison at 6x6x8 was bit-identical; 27,973 steps of accumulation is what
# separates them. The parameter count rising 49 -> 160 costs nothing: the reverse
# pass returns the whole theta cotangent in one sweep.
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
# The peak is the ssp_vjp COMPILE, not the window: 576 macro steps add only
# ~390 MB of checkpoints on top. 64G was the pre-run sizing, taken from a
# 3-macro-step peak of 22.96 GB; 10044327 then measured MaxRSS 55.2 GB against it,
# 86% of the request, which is not margin -- hence 80G. Under esm 1.0.0 the same
# run peaked at 38.4 GB (10055533), so 80G is now ~2x rather than 1.4x. Do not
# trim it back: 96G sat PENDING on (Resources), and these nodes are held by
# whole-node --mem=0 jobs often enough that the margin is cheaper than a requeue.
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

# run-model-jl, no overrides: since the esm 1.0.0 migration it develops all four
# packages at their release versions (EarthSciAST 0.1.1, EarthSciASTDiff 0.1.0,
# EarthSciASTSplitter 0.1.0, EarthSciIO 0.1.2), so jac=:sym needs no pinned env
# and the model is this repo's own working copy.
export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_NLON=${RESEACT_NLON:-13}
export RESEACT_NLAT=${RESEACT_NLAT:-7}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-576}   # 576 * 300 s = 48 h
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-1}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-0}
# 1, not 0: 0 differentiates a map the production runner does not use. The retest
# (10042578) and the 576-step run (10044327) both kept the sweep finite with it on.
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-1}
export RESEACT_ADJ_OBJ=${RESEACT_ADJ_OBJ:-SuperFast.O3:surf}
export RESEACT_LABEL=${RESEACT_LABEL:-conus-adj48-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-logs/adj48-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== jac=$RESEACT_ADJ_JAC clamp=$RESEACT_ADJ_CLAMP ujitter=$RESEACT_ADJ_UJITTER nmacro=$RESEACT_ADJ_NMACRO ==="
free -g | head -2
echo "started $(date -Is)"

julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/adjoint_gradient.jl

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
