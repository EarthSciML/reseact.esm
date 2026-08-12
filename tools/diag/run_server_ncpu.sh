#!/bin/bash
# The same server, but XLA:CPU gets exactly ONE intra-op thread. XLA sizes its
# CPU thread pool from tsl::port::MaxParallelism(), which reads the process's
# CPU affinity mask, so `taskset -c N` is a reliable way to force a 1-thread
# pool; XLA_FLAGS says the same thing the other way in case the client does not.
export JULIA_DEPOT_PATH="/tmp/claude-1111265/depot:/u/ctessum/.julia"
export RESEACT_RXENV="${RESEACT_RXENV:-/tmp/claude-1111265/nanhunt-env}"
export RESEACT_LABEL="${RESEACT_LABEL:-nondet4t}"
export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-0}
export RESEACT_JOBDIR="${RESEACT_JOBDIR:-/tmp/claude-1111265/wt-nanhunt/tools/diag/jobs4t}"
# XLA_FLAGS must start with "--"; the bare intra_op_parallelism_threads token is
# rejected by this build, so the thread count is forced by the affinity mask alone
# (XLA sizes its CPU pool from tsl::port::MaxParallelism -> sched_getaffinity).
cd "$(dirname "$0")/../.."
exec taskset -c "${NANHUNT_CPU:-8-11}" julia tools/diag/nondet_server.jl
