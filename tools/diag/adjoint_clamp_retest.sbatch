#!/bin/bash
# ===========================================================================
# adjoint_clamp_retest.sbatch -- BLOCKER 1, retested on the fixed footing.
# ===========================================================================
# WHAT BLOCKER 1 IS. With the production positivity clamp `clamp_nonneg` ON, the
# backward sweep acquires non-finite lambda entries partway back through macro
# step 3 (312 entries, 24 cells, all 13 state groups) at 6x6x8. Every CONUS
# gradient ever produced has therefore run with RESEACT_ADJ_CLAMP=0, i.e. it
# differentiates a map the production runner does not use. A week -- or 48 h --
# is 576+ macro steps; something that fires at step 3 is not survivable.
#
# WHY RETEST IT NOW, AND WHY IT IS THE FIRST THING TO RETEST. The original
# observation predates BOTH of the things that have since changed underneath it:
#
#   * the XLA:CPU race workaround (blocker 4). The driver did not set
#     xla_cpu_prefer_vector_width=128 when blocker 1 was recorded. The race
#     returns non-finite values from the compiled chemistry program at ~1e-4
#     state-visible rate per call, and a sweep issues hundreds of calls. "A
#     non-finite appears partway through the third macro step" is EXACTLY what
#     the race looks like, and the file already names it as the obvious first
#     hypothesis -- untested.
#   * the block Jacobian. It was finite differences. Rosenbrock23 solves with
#     (I/(gamma*dt) - J); the clamp writes EXACT ZEROS into the state, an FD
#     Jacobian differences a stiff chemistry RHS about a zero concentration with
#     step sqrt(eps)*|u| = 0, and the resulting column is garbage. jac=:sym has
#     no step size and evaluates the analytic derivative AT zero, which is
#     finite. So the clamp and the FD Jacobian are not independent suspects --
#     the clamp is what MAKES the FD Jacobian degenerate.
#
# Both are now fixed, and neither fix has ever been combined with CLAMP=1.
# This job runs the identical configuration twice, clamp off then clamp on, in
# ONE process-independent pair of runs so the only difference is the clamp.
#
# READ: "the unclamped sweep is FINITE throughout" and, in the clamped run,
# whether "FIRST NON-FINITE" appears at all. If it does not, blocker 1 is gone
# and it was a consequence of blockers 2 and 4 rather than a fact about the
# clamp. If it does, the clamp is a real discontinuity and the next question is
# whether to differentiate a smoothed clamp instead.
#
# SIZING. 6x6x8 with jac=:sym peaked at 6.7 GB (slurm 10041660) and the sym
# Jacobian compiles 4x cheaper than fd. 32G is generous; two runs, ~1 h each.
#
#   mkdir -p logs && sbatch tools/diag/adjoint_clamp_retest.sbatch
# ===========================================================================
#SBATCH --job-name=adj-clamp
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

# The symbolic Jacobian needs EarthSciASTDiff, which run-model-jl cannot resolve
# (its EarthSciAST checkout is on the 0.1.x line; EarthSciASTDiff wants 0.9.x).
# env-sym is run-model-jl's dependency set plus EarthSciASTDiff, pinned, and the
# corpus-pinned model is the one that env can load.
export RESEACT_RXENV=${RESEACT_RXENV:-/u/ctessum/reseact-esast-pin/env-sym}
export RESEACT_MODEL=${RESEACT_MODEL:-/u/ctessum/reseact-esast-pin/corpus-pin/reseact.esm/reseact.esm}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
[ -f "$RESEACT_MODEL" ] || { echo "FATAL: model $RESEACT_MODEL missing"; exit 1; }

export RESEACT_NLON=${RESEACT_NLON:-6}
export RESEACT_NLAT=${RESEACT_NLAT:-6}
export RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-3}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-1}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-1e-1}

echo "host=$(hostname) env=$RESEACT_RXENV jac=$RESEACT_ADJ_JAC xlafix=$RESEACT_ADJ_XLAFIX"
echo "started $(date -Is)"

for CL in 0 1; do
  echo ""
  echo "############################################################"
  echo "### CLAMP=$CL   (jac=$RESEACT_ADJ_JAC, xlafix=$RESEACT_ADJ_XLAFIX)"
  echo "############################################################"
  RESEACT_ADJ_CLAMP=$CL RESEACT_LABEL="adj-clamp$CL-${SLURM_JOB_ID}" \
    julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=24G tools/adjoint_gradient.jl \
    || echo "### CLAMP=$CL RUN FAILED (exit $?) -- continuing to the next"
done

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
