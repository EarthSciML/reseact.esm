#!/bin/bash
# ===========================================================================
# steptime_shard_run.sh -- orchestrate steptime_shard.jl over N lon shards
# ===========================================================================
# Runs each shard's Julia process with a staggered build (at most one process
# in its memory-heavy build phase at a time), a solo timing, a file barrier,
# and one concurrent timing across all shards. The decisive comparison is
#   max over shards of (concurrent step time)   vs   full-domain solo time:
# it banks BOTH the smaller working set and the extra execution streams, and
# needs no Reactant multi-device machinery -- plain separate processes.
#
#   SHARDS      space-separated "label:lon0:nlon" (default the P1 halves:
#               "west:11:7 east:18:6"); run "full:11:13" alone for a baseline
#   SHARD_DIR   scratch dir for barrier files + logs (default ./logs/shardrun)
#   JULIA_ARGS  extra julia flags (default "-t 5 --heap-size-hint=6G")
# Callers must provide RESEACT_RXENV / JULIA_DEPOT_PATH (see the local-run
# recipe in the campaign notes: snapshot the dev packages, sed the Manifest
# `path =` entries -- Pkg.develop cannot resolve through an offline depot).
#
# MEASURED 2026-08-24/25 on ccc0232 (load ~17-26), 60 reps, vs the same
# session's full-CONUS solo baseline 375.64 ms med / 212.72 ms min:
#   N=2 halves   west NC=3528 solo 133.56/115.84 -> conc 141.06/127.52
#                east NC=3024 solo 111.58/ 93.10 -> conc 120.61/104.59
#                domain step max(conc) = 141.1 ms -> 2.66x med (1.67x min)
#   The split: ~1.5x/cell working-set effect x a near-free second stream
#   (6-8% concurrency penalty -- one stream leaves the machine nearly idle).
#   N=4 quarters: see the K4 MEASURED block appended below after that run.
# ===========================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SHARDS="${SHARDS:-west:11:7 east:18:6}"
SHARD_DIR="${SHARD_DIR:-$REPO/logs/shardrun}"
JULIA_ARGS="${JULIA_ARGS:--t 5 --heap-size-hint=6G}"
mkdir -p "$SHARD_DIR"; rm -f "$SHARD_DIR"/GO*
export RESEACT_BACKEND=cpu RESEACT_ADJ_JAC="${RESEACT_ADJ_JAC:-sym}" RESEACT_ADJ_CLAMP=1
export RESEACT_NLAT="${RESEACT_NLAT:-7}" RESEACT_NLEV="${RESEACT_NLEV:-72}"
export RESEACT_STEPTIME_REPS="${RESEACT_STEPTIME_REPS:-60}"
cd "$REPO"
echo "=== shard run on $(hostname), load $(cut -d' ' -f1 /proc/loadavg), $(date -Is) ==="
PIDS=(); LABELS=()
for spec in $SHARDS; do
  IFS=: read -r LBL LON0 NLON <<< "$spec"
  echo "--- launching $LBL (lon0=$LON0 nlon=$NLON), waiting for its solo ---"
  RESEACT_LABEL=$LBL RESEACT_LON0=$LON0 RESEACT_NLON=$NLON \
    RESEACT_STEPTIME_BARRIER="$SHARD_DIR/GO" \
    julia --project="$RESEACT_RXENV" $JULIA_ARGS tools/diag/steptime_shard.jl \
    > "$SHARD_DIR/$LBL.log" 2>&1 &
  PID=$!; PIDS+=("$PID"); LABELS+=("$LBL")
  while [ ! -f "$SHARD_DIR/GO.$LBL.solo.done" ]; do
    kill -0 "$PID" 2>/dev/null || { echo "$LBL DIED before solo done"; tail -20 "$SHARD_DIR/$LBL.log"; exit 1; }
    sleep 5
  done
  grep STEPTIME "$SHARD_DIR/$LBL.log" || true
done
echo "--- all solos done, releasing barrier ($(date -Is), load $(cut -d' ' -f1 /proc/loadavg)) ---"
touch "$SHARD_DIR/GO"
wait "${PIDS[@]}"
for LBL in "${LABELS[@]}"; do echo "--- $LBL ---"; grep STEPTIME "$SHARD_DIR/$LBL.log"; done
echo "SHARDRUN_DONE $(date -Is)"
