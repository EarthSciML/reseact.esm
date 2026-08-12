#!/bin/bash
# One build, many experiments -- see nondet_server.jl. Drop jobs in tools/diag/jobs/.
export JULIA_DEPOT_PATH="/tmp/claude-1111265/depot:/u/ctessum/.julia"
export RESEACT_RXENV="${RESEACT_RXENV:-/tmp/claude-1111265/nanhunt-env}"
export RESEACT_LABEL="${RESEACT_LABEL:-nondet}"
export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-0}
cd "$(dirname "$0")/../.."
exec julia tools/diag/nondet_server.jl
