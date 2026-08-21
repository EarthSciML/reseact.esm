#!/bin/bash
# ===========================================================================
# traced_symjac_smoke.sbatch -- RESEACT_RXJAC=sym inside the traced while loop.
# ===========================================================================
# The adjoint driver lifts its time loop onto the host and differentiates ONE
# step at a fixed dt, so `jac=:sym` there is only ever traced as a standalone
# program. This runner does not: `adaptive_solve` is a `@trace while` and the
# Jacobian is emitted INSIDE the loop body. Whether the band model traces there
# is a separate question, and it is the one that has to be answered before the
# forward runner can be moved off finite differences.
#
# WHY MOVE IT AT ALL. The 36.2 h CONUS death was attributed to the FD chemistry
# Jacobian failing on species that dry deposition had driven to ~1e-30 -- FD's
# step there is sqrt(eps)*|u|, which is not a step. If that diagnosis is right,
# :sym is the fix, and 48 h becomes reachable in the forward runner.
#
# WHAT THIS CHECKS, at 6x6x8 over two macro steps so it costs minutes:
#   1. it compiles at all inside the while region;
#   2. compile time and module size against :fd;
#   3. the trajectories agree -- same physics, different linearization, so the
#      two solves should land in the same place to solver tolerance, and the
#      accept/reject counts say whether the better Jacobian pays for itself.
#
#   mkdir -p logs && sbatch tools/diag/traced_symjac_smoke.sbatch
# ===========================================================================
#SBATCH --job-name=rx-symjac
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

# run-model-jl resolves EarthSciASTDiff on its own since the esm 1.0.0 migration
# (2026-08-21): it develops all four packages at their release versions, so the
# symbolic Jacobian needs neither the hand-built env-sym nor the corpus-pin copy
# of the model that every symbolic run before that date went through. The model
# is this repo's own working copy. Override RESEACT_RXENV/RESEACT_MODEL to revisit.
export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_NLON=${RESEACT_NLON:-6}
export RESEACT_NLAT=${RESEACT_NLAT:-6}
export RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_SOLVE_SECS=${RESEACT_SOLVE_SECS:-600}
export RESEACT_RXFIX=1

echo "host=$(hostname)  started $(date -Is)"
for J in fd sym; do
  echo ""
  echo "############################################################"
  echo "### RXJAC=$J"
  echo "############################################################"
  RESEACT_RXJAC=$J RESEACT_LABEL="rx-symjac-$J" \
    RESEACT_CSV="logs/rx-symjac-$J-${SLURM_JOB_ID}.csv" \
    julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=24G run_reseact_reactant.jl \
    || echo "### RXJAC=$J FAILED -- continuing"
done
echo "finished $(date -Is)"
