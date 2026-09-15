#!/bin/bash
# ===========================================================================
# decode_cost.sbatch -- decode cost per forcing refresh, at a chosen resolution.
# ===========================================================================
# The 0.25x0.3125 arm decodes GLOBAL arrays (~1.9 GB per 3-D variable), which is
# why this exists instead of running decode_cost.jl on the login node.
#
#   sbatch --export=ALL,RESEACT_DECODE_RES=0.25x0.3125 tools/diag/decode_cost.sbatch
# ===========================================================================
#SBATCH --job-name=decodecost
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/run-model-jl}
export EARTHSCIIO_HTTP_TIMEOUT=${EARTHSCIIO_HTTP_TIMEOUT:-3600}
export RESEACT_DECODE_RES=${RESEACT_DECODE_RES:-4x5,2x2.5}
echo "=== node $(hostname) res=$RESEACT_DECODE_RES ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 tools/diag/decode_cost.jl
echo "finished $(date -Is)"
