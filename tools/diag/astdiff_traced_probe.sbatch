#!/bin/bash
# ===========================================================================
# astdiff_traced_probe.sbatch -- GATE B: does the symbolic Jacobian TRACE?
# ===========================================================================
# See tools/diag/astdiff_traced_probe.jl for what this decides and why. In one
# sentence: the analytical Jacobian is proven correct on the HOST, and the only
# question left before it can replace the FD block Jacobian on the traced arm
# (and, more importantly, inside the ADJOINT, where jac=:ad segfaults) is
# whether the derived band model compiles under Reactant.
#
# ENV. env-pin-seam is the ONLY env with all three pieces:
#   esast-pin-wt   = EarthSciAST pin 98e6f1b6e + the expand_flattened_refs seam.
#                    The pin is required: reseact.esm does not validate against
#                    EarthSciAST main since `examples` -> `analyses` (619 errors).
#                    The seam is required: ReSEACT's chemistry half exists only
#                    as a FlattenedSystem, and that path skipped template
#                    expansion until the seam landed.
#   astdiff-item1  = EarthSciASTDiff with the state-independence pruning.
#   Reactant       = needed here and NOT present in every pinned env.
# Do not "simplify" this to a repo-default env; it will fail at load.
#
# SIZING. Build ~600 s at 6x6x8, then prepare_jacobian, then an XLA compile of
# the band model (the chemistry RHS itself compiles in ~89 s at this size; the
# band model does ~2.9x the arithmetic, so allow several times that). 3 h is
# slack, not an estimate.
#
# 16 cpus because the XLA:CPU compile is threaded; --mem=64G because the session
# cgroup is 40 GiB and the symbolic step allocates hard.
# ===========================================================================
#SBATCH --job-name=astdiff-traced
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -u
REPO=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-/u/ctessum/reseact-esast-pin/env-pin-seam}
# CORPUS PIN. On 2026-08-20 the two sibling corpora reseact.esm imports from
# both landed on esm 1.0.0, one minute apart:
#   EarthSciDiscretizations a4fb619  15:13:47  refactor!: the library corpus ...
#   EarthSciModels          bf0526f  15:14:42  refactor!: the model corpus ...
# The pinned EarthSciAST predates the 1.0.0 schema, so it rejects the migrated
# corpus with 754 SchemaValidationErrors -- job 10037638 died on exactly this,
# 20 minutes after job 10037492 built fine from the same env. Same class of
# break as `examples` -> `analyses`, different repo.
# corpus-pin/ holds worktrees of both at their last pre-migration commits
# (7a44026 / c7047cd) plus a COPY of reseact.esm as their sibling, so the
# model's 16 relative `../EarthSciModels|EarthSciDiscretizations/...` refs
# resolve into the pin. VERIFIED: pinned loads in 78 s, live fails.
# NOTE the copy can drift from the repo's reseact.esm -- refresh it if the
# model changes. Unset RESEACT_MODEL to test against the live corpus.
export RESEACT_MODEL=${RESEACT_MODEL:-/u/ctessum/reseact-esast-pin/corpus-pin/reseact.esm/reseact.esm}
[ -f "$RESEACT_MODEL" ] || { echo "FATAL: model $RESEACT_MODEL missing"; exit 1; }
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV does not exist."; \
  echo "Pkg.activate would SILENTLY CREATE an empty one and fail seconds later."; exit 1; }
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$REPO" || exit 1
echo "host=$(hostname) env=$RESEACT_RXENV threads=$JULIA_NUM_THREADS"
julia --project="$RESEACT_RXENV" tools/diag/astdiff_traced_probe.jl
echo "=== exit $? at $(date -Is) ==="
