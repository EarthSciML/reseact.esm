#!/bin/bash
# ===========================================================================
# rx_adjoint_sym.sbatch -- can the ADJOINT cross the SYMBOLIC Jacobian?
# ===========================================================================
# Gate B established that the symbolic (EarthSciASTDiff) Jacobian COMPILES under
# Reactant and matches the host to 2.1e-16 at both 6x6x8 and full CONUS. What it
# did NOT establish is the thing the adjoint actually needs: that a REVERSE pass
# can cross it, and what that does to accuracy. This job asks exactly that, via
# the `ros_sym` stage added to tools/rx_adjoint_check.jl.
#
# WHY IT SHOULD WORK WHERE jac=:ad DOES NOT. `:ad` builds the Jacobian by
# coloured forward-mode JVPs, so it emits an inner `enzyme.fwddiff`; reverse
# over that is reverse-over-forward, which segfaults inside Enzyme-MLIR on this
# model (tools/diag/UPSTREAM_reverse_over_forward.md). The symbolic Jacobian
# contains NO nested AD -- it is a gather into a straight-line band model -- so
# there is no fwddiff for the reverse pass to trip over. Verified in miniature
# first: a padded gather with repeated indices reverse-differentiates exactly,
# accumulating rather than copying.
#
# WHAT WOULD MAKE IT A WIN. jac=:fd degrades the dot-product identity
# <lam, Jv> == <J'lam, v> from 1.8e-16 to 1.1e-6 on ReSEACT -- right at the
# 1e-6 accuracy the week-long gradient is supposed to deliver. An exact
# Jacobian the reverse pass can cross removes that entirely. SPEED is a
# SEPARATE question and this job does not answer it.
#
# STAGES, in the order the harness runs them:
#   census   -- the module handed to Enzyme must contain no stablehlo.while,
#               including the jac=:sym primal (reverse cannot cross one)
#   jac      -- the symbolic Jacobian against BOTH the AD and FD ones. AD and
#               symbolic answer to different authorities (emitted code vs the
#               AST), so agreement is a real cross-check, and the structural-
#               zero test catches a coupling the band model simply lacks --
#               which no norm would show.
#   ros_sym  -- the full one-step VJP/JVP/FD battery.
# `jac` and `census` print BEFORE `ros_sym` compiles anything, so if the reverse
# pass does segfault after all, the cross-check survives in the log.
#
# SIZING. 6x6x8 (the stage algebra is model-independent). Build ~600 s,
# prepare_jacobian ~430 s, then several XLA compiles of which the VJP is the
# unknown. 6 h is slack, not an estimate.
# ===========================================================================
#SBATCH --job-name=rx-adj-sym
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -u
REPO=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
# run-model-jl, plain. This job used to need env-sym (a hand-built environment
# with EarthSciASTDiff pinned against an older EarthSciAST) and a corpus-pin copy
# of the model, because the sibling corpora went esm 1.0.0 on 2026-08-20 and the
# pinned EarthSciAST rejected the migrated form with 754 SchemaValidationErrors.
# The 2026-08-21 migration moved this repo to 1.0.0 too, so both pins are retired.
export RESEACT_RXENV=${RESEACT_RXENV:-$REPO/run-model-jl}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-census,jac,ros_sym}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-0}
export RESEACT_NLON=${RESEACT_NLON:-6}
export RESEACT_NLAT=${RESEACT_NLAT:-6}
export RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_RXFIX=${RESEACT_RXFIX:-1}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV does not exist."; \
  echo "Pkg.activate would SILENTLY CREATE an empty one and fail seconds later."; exit 1; }
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$REPO" || exit 1
echo "host=$(hostname) env=$RESEACT_RXENV stages=$RESEACT_ADJ_STAGES threads=$JULIA_NUM_THREADS"
julia --project="$RESEACT_RXENV" tools/rx_adjoint_check.jl
echo "=== exit $? at $(date -Is) ==="
