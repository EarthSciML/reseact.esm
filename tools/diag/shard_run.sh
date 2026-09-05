#!/bin/bash
# ===========================================================================
# shard_run.sh -- run tools/adjoint_gradient.jl with the chemistry SHARDS on.
# ===========================================================================
# Thin launcher used by the sharding gate and timing jobs (tools/diag/shard_*.sbatch)
# and interactively at 6x6x8. Every RESEACT_* variable already in the
# environment WINS; the defaults below are the 6x6x8 gate configuration.
#
#   SHARD_LOG   log path (default logs/shard-<label>.out)
#   JULIA_ARGS  driver julia flags (default "-t 4 --heap-size-hint=12G")
# ===========================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export RESEACT_RXENV="${RESEACT_RXENV:-$REPO/run-model-jl}"
# the model's ../EarthSciModels refs resolve against the DOCUMENT's directory,
# which a git worktree does not have as a sibling; the file is tracked and
# identical, so point at the main checkout's copy
export RESEACT_MODEL="${RESEACT_MODEL:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/reseact.esm}"
# same reason: prototypes/reseact_3d_chem/blockdiag_local.jl includes two files
# of the sibling EarthSciMLBase checkout by relative path unless told otherwise
export EARTHSCIMLBASE_SRC="${EARTHSCIMLBASE_SRC:-/projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciMLBase/src}"
export RESEACT_NLON="${RESEACT_NLON:-6}" RESEACT_NLAT="${RESEACT_NLAT:-6}" RESEACT_NLEV="${RESEACT_NLEV:-8}"
export RESEACT_ADJ_NMACRO="${RESEACT_ADJ_NMACRO:-3}"
export RESEACT_ADJ_UJITTER="${RESEACT_ADJ_UJITTER:-0}"
export RESEACT_ADJ_CLAMP="${RESEACT_ADJ_CLAMP:-1}"
export RESEACT_ADJ_STAGES="${RESEACT_ADJ_STAGES:-fwd,adj}"
export RESEACT_ADJ_SHARDS="${RESEACT_ADJ_SHARDS:-2}"
export RESEACT_LABEL="${RESEACT_LABEL:-shard-s${RESEACT_ADJ_SHARDS}}"
export RESEACT_ADJ_CSV="${RESEACT_ADJ_CSV:-logs/${RESEACT_LABEL}.csv}"
JULIA_ARGS="${JULIA_ARGS:--t 4 --heap-size-hint=12G}"
mkdir -p logs
SHARD_LOG="${SHARD_LOG:-logs/${RESEACT_LABEL}.out}"
echo "=== $(hostname) label=$RESEACT_LABEL grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} shards=$RESEACT_ADJ_SHARDS stages=$RESEACT_ADJ_STAGES nmacro=$RESEACT_ADJ_NMACRO $(date -Is) ===" | tee "$SHARD_LOG"
julia --project="$RESEACT_RXENV" $JULIA_ARGS tools/adjoint_gradient.jl >> "$SHARD_LOG" 2>&1
rc=$?
echo "=== finished rc=$rc $(date -Is) ===" | tee -a "$SHARD_LOG"
exit $rc
