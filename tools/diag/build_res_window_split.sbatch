#!/bin/bash
# ===========================================================================
# build_res_window_split.sbatch -- one PROCESS per (res, arm).
# ===========================================================================
# `build_res_window.sbatch` runs every arm in one julia process, so each arm
# after the first inherits the previous one's JIT warm-up. Measured at 4x5
# (logs/brw-smoke3.out): `doc load+split` 62.49 s -> 40.41 s and
# `build_evaluator1` 92.38 s -> 40.15 s between the `full` and `win` arms of the
# SAME resolution -- i.e. more than half of the apparent windowing speedup was
# the second arm running warm. The RSS, kept-MB and RHS-hash columns are
# order-independent and unaffected; the TIMES are not.
#
# So: one process per arm, each paying the same cold start. Same script, one arm
# at a time.
#
#   sbatch tools/diag/build_res_window_split.sbatch
#   sbatch --export=ALL,BRW_RES=0.25x0.3125 tools/diag/build_res_window_split.sbatch
# ===========================================================================
#SBATCH --job-name=brwsplit
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/run-model-jl}
export EARTHSCIIO_HTTP_TIMEOUT=${EARTHSCIIO_HTTP_TIMEOUT:-3600}
export EARTHSCIIO_HTTP_RETRIES=${EARTHSCIIO_HTTP_RETRIES:-8}
RESL=${BRW_RES:-2x2.5,0.25x0.3125}
ARMS=${BRW_ARMS:-full,win}
export BRW_GRID=${BRW_GRID:-6x6x8}
export BRW_PARTS=${BRW_PARTS:-1,2}
echo "=== node $(hostname) res=$RESL arms=$ARMS grid=$BRW_GRID (one process per arm) ==="
free -g
echo "started $(date -Is)"
for res in ${RESL//,/ }; do
  for arm in ${ARMS//,/ }; do
    echo
    echo "################ process: res=$res arm=$arm ################"
    BRW_RES="$res" BRW_ARMS="$arm" \
      julia --project="$RESEACT_RXENV" -t 8 tools/diag/build_res_window.jl
  done
done
echo "finished $(date -Is)"
