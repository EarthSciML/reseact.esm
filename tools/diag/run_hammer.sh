#!/bin/bash
# Standard environment for the nondeterminism hammer. See nondet_hammer.jl.
export JULIA_DEPOT_PATH="/tmp/claude-1111265/depot:/u/ctessum/.julia"
export RESEACT_RXENV="${RESEACT_RXENV:-/tmp/claude-1111265/nanhunt-env}"
export RESEACT_LABEL="${RESEACT_LABEL:-hammer}"
export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-0}
cd "$(dirname "$0")/../.."
exec julia "${JLFLAGS[@]}" tools/diag/nondet_hammer.jl
