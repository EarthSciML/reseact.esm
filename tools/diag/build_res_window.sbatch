#!/bin/bash
# ===========================================================================
# build_res_window.sbatch -- the resolution axis of the build cost, and the fix.
# ===========================================================================
# Holds the MODEL grid fixed and varies the GEOS-FP native resolution, then
# repeats each resolution with the forcing arrays pre-sliced to the halo window
# the model actually reads. Answers, by measurement rather than arithmetic,
# where the 5.85x build tax between 2x2.5 and 0.25x0.3125 at an IDENTICAL cell
# count (slurm 10386111 vs 10395008) actually lands.
#
# The 0.25 arm holds ~7.8 GB of whole-globe forcing and decodes several times
# that transiently, so this is an sbatch and not a login-node run.
#
#   sbatch tools/diag/build_res_window.sbatch
#   sbatch --export=ALL,BRW_RES=0.25x0.3125,BRW_ARMS=win tools/diag/build_res_window.sbatch
# ===========================================================================
#SBATCH --job-name=brwindow
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/run-model-jl}
# 0.25 deg A3dyn is 3.80 GB/day and EarthSciIO caps ONE request at 90 s, so a
# cold fetch dies at ~3.5 GB. The 2016-01-01 files are already cached, but keep
# the raise so a cache miss is a slow fetch and not a failed run.
export EARTHSCIIO_HTTP_TIMEOUT=${EARTHSCIIO_HTTP_TIMEOUT:-3600}
export EARTHSCIIO_HTTP_RETRIES=${EARTHSCIIO_HTTP_RETRIES:-8}
export BRW_RES=${BRW_RES:-2x2.5,0.25x0.3125}
export BRW_ARMS=${BRW_ARMS:-full,win}
export BRW_GRID=${BRW_GRID:-6x6x8}
export BRW_PARTS=${BRW_PARTS:-1,2}
echo "=== node $(hostname) res=$BRW_RES arms=$BRW_ARMS grid=$BRW_GRID ==="
free -g
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 tools/diag/build_res_window.jl
echo "finished $(date -Is)"
