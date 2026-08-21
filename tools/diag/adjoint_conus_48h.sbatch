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
# WHY IT IS BEING RUN AGAIN. That run was produced out of the hand-built env-sym
# against a corpus-pin copy of the model, because run-model-jl could not then
# resolve EarthSciASTDiff. The esm 1.0.0 migration retired both pins, and its
# equivalence evidence is at 6x6x8: J bit-identical, all 19 nonzero components
# bit-identical, worst relative 0.000e+00. This re-run is that same check AT CONUS
# AND AT 48 h -- the scale where a schema migration could plausibly move something
# a small grid cannot see. Expect J = 30.1943301698541 and a gradient that matches
# on its nonzero components; 1.0.0 exposes setup-time geometry constants as runtime
# scalars, so the parameter COUNT rises from 49 to ~160 with the additions zero
# (the reverse pass returns the whole theta cotangent at once, so that is free).
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
# 3-macro-step peak of 22.96 GB; the 576-step run then measured MaxRSS 55.2 GB
# against it, 86% of the request, which is not margin. 80G restores it without
# reaching for a whole node's memory reservation (96G sat PENDING on Resources).
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
