#!/bin/bash
# ===========================================================================
# forcing_size_ladder.sbatch -- one PROCESS per rung, each under `perf stat`.
# ===========================================================================
# One process per rung for the same reason build_res_window_split.sbatch has
# one per arm: a second arm in the same process runs JIT-warm and its time is
# not comparable (measured at 4x5: build_evaluator1 92 s -> 40 s purely from
# warm-up).
#
# `perf stat` wraps the WHOLE process, which is exactly what makes it readable
# here: every rung decodes the SAME 2x2.5 files and builds the SAME equations,
# so everything except the size of the forcing array is common-mode and the
# DIFFERENCE between two rungs' counters is attributable to the array. That is
# the discriminator between a locality mechanism (cache-misses / dTLB /
# page-faults grow while instructions stay flat) and a work mechanism
# (instructions grow too).
#
#   sbatch tools/diag/forcing_size_ladder.sbatch
#   sbatch --export=ALL,FSL_GRID=25x13x72,FSL_STAGES=build,loop tools/diag/forcing_size_ladder.sbatch
# ===========================================================================
#SBATCH --job-name=fsl
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/run-model-jl}
export EARTHSCIIO_HTTP_TIMEOUT=${EARTHSCIIO_HTTP_TIMEOUT:-3600}
export EARTHSCIIO_HTTP_RETRIES=${EARTHSCIIO_HTTP_RETRIES:-8}
export FSL_GRID=${FSL_GRID:-25x13x72}
export FSL_STAGES=${FSL_STAGES:-build}
export FSL_PARTS=${FSL_PARTS:-1}
export FSL_REPS=${FSL_REPS:-30}
RUNGS=${FSL_RUNGS:-win,r1,r2,r4,r8}
EV=task-clock,cycles,instructions,cache-references,cache-misses,LLC-load-misses,dTLB-load-misses,page-faults,major-faults

echo "=== node $(hostname) grid=$FSL_GRID stages=$FSL_STAGES rungs=$RUNGS (one process per rung) ==="
free -g
echo "started $(date -Is)"
for rung in ${RUNGS//,/ }; do
  echo
  echo "################ process: rung=$rung ################"
  FSL_RUNGS="$rung" perf stat -e "$EV" -- \
    julia --project="$RESEACT_RXENV" -t 8 tools/diag/forcing_size_ladder.jl || true
done
echo "finished $(date -Is)"
