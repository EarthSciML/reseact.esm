#!/bin/bash
# ===========================================================================
# build_driver_decomp.sbatch -- the driver's BUILD, decomposed, one process per
# (res, ndays) arm.
# ===========================================================================
#   sbatch tools/diag/build_driver_decomp.sbatch
#   sbatch --export=ALL,BDD_ARMS='2x2.5:1,2x2.5:4' tools/diag/build_driver_decomp.sbatch
#
# BDD_ARMS is a comma list of `res:ndays` pairs. Each runs in its OWN process,
# for the same reason build_res_window_split.sbatch does.
# ===========================================================================
#SBATCH --job-name=bdd
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=175G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/run-model-jl}
export EARTHSCIIO_HTTP_TIMEOUT=${EARTHSCIIO_HTTP_TIMEOUT:-3600}
export EARTHSCIIO_HTTP_RETRIES=${EARTHSCIIO_HTTP_RETRIES:-8}
export BDD_GRID=${BDD_GRID:-25x13x72}
export BDD_INSPECT=${BDD_INSPECT:-1}
# Each arm is `res:ndays[:ballast_gb]`. The third field holds that many GB
# resident for the whole process, which is how the driver's own residency (a
# 62-88 GB process with eight 5.5 GB shard workers) is turned into a controlled
# variable against this probe's ~16 GB.
ARMS=${BDD_ARMS:-2x2.5:1,2x2.5:4,0.25x0.3125:1,0.25x0.3125:4}

echo "=== node $(hostname) grid=$BDD_GRID inspect=$BDD_INSPECT arms=$ARMS ==="
free -g
echo "started $(date -Is)"
for arm in ${ARMS//,/ }; do
  IFS=: read -r res nd bal <<< "$arm"
  bal=${bal:-0}
  echo
  echo "################ process: res=$res ndays=$nd ballast=${bal}GB ################"
  BDD_RES="$res" BDD_NDAYS="$nd" BDD_BALLAST_GB="$bal" \
    julia --project="$RESEACT_RXENV" -t 8 tools/diag/build_driver_decomp.jl || true
done
echo "finished $(date -Is)"
