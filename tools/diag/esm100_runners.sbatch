#!/bin/bash
# ===========================================================================
# esm100_runners.sbatch -- do the OTHER TWO runners still work on esm 1.0.0?
# ===========================================================================
# The 1.0.0 migration was verified through the ADJOINT path only: the model
# loads, splits, builds, and `tools/adjoint_gradient.jl` reproduces the 0.8.0
# gradient bit-for-bit (all 19 nonzero components, worst relative 0.000e+00).
# That exercises the host-loop driver. It does NOT exercise:
#
#   * run_reseact_reactant.jl -- the traced runner, whose Rosenbrock23 lives
#     inside a `@trace while` rather than being called one fixed step at a time;
#   * run_reseact.jl -- the native (non-Reactant) runner, which goes through
#     OrdinaryDiffEq and the operator-splitting stack rather than through
#     RxTracedIntegrator at all.
#
# Both read the same migrated .esm files and the same renamed provider keys, so
# both are exposed to the two non-mechanical parts of the migration: the
# coupling rewrite (data loaders stopped being subsystems) and the host-side
# provider keys (39 occurrences that live outside the .esm entirely and that no
# schema check can see).
#
# NOTE WHAT THIS USES: --project=run-model-jl and the REPO's own reseact.esm,
# with no RESEACT_RXENV or RESEACT_MODEL override. That is the point. Before the
# migration run-model-jl could not resolve EarthSciASTDiff at all (its
# EarthSciAST was the 0.1.x line while EarthSciASTDiff wanted 0.9.x), which is
# why every symbolic-Jacobian run so far went through a pinned env-sym against a
# corpus-pin copy of the model. If this job passes, that apparatus is retired.
#
# Three arms, smallest grid, short window -- this is a does-it-work gate, not a
# measurement, and the timings are not comparable to anything:
#   1. traced runner, RXJAC=fd   -- the default path
#   2. traced runner, RXJAC=sym  -- the symbolic Jacobian inside the while region
#   3. native runner             -- the OrdinaryDiffEq path
#
#   mkdir -p logs && sbatch tools/diag/esm100_runners.sbatch
# ===========================================================================
#SBATCH --job-name=esm100-run
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

export RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8
export RESEACT_SOLVE_SECS=600
export RESEACT_RXFIX=1

echo "host=$(hostname)  started $(date -Is)"
echo "env=run-model-jl (the repo's own), model=reseact.esm (the repo's own) -- no overrides"
julia --project=run-model-jl -e 'using Pkg; Pkg.status()' 2>&1 | grep -iE "earthsci|reactant"

rc=0
for J in fd sym; do
  echo ""; echo "############ traced runner, RXJAC=$J ############"
  RESEACT_RXJAC=$J RESEACT_LABEL="esm100-rx-$J" \
    RESEACT_CSV="logs/esm100-rx-$J-${SLURM_JOB_ID}.csv" \
    julia --project=run-model-jl -t 8 --heap-size-hint=24G run_reseact_reactant.jl \
    || { echo "### traced RXJAC=$J FAILED"; rc=1; }
done

echo ""; echo "############ native runner ############"
RESEACT_LABEL="esm100-native" RESEACT_CSV="logs/esm100-native-${SLURM_JOB_ID}.csv" \
  julia --project=run-model-jl -t 8 --heap-size-hint=24G run_reseact.jl \
  || { echo "### native FAILED"; rc=1; }

echo ""; echo "finished $(date -Is)  rc=$rc"
exit $rc
