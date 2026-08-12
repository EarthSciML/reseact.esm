#!/usr/bin/env bash
# Drive tools/diag/rof_repro.jl ONE CONFIGURATION PER PROCESS and record the
# verdict from OUTSIDE the process -- a segfault cannot report its own death.
#
#   tools/diag/rof_sweep.sh <tag> [KEY=VAL ...]
#
# Appends one line to tools/diag/rof_results.tsv and leaves the full log in
# tools/diag/logs/<tag>.log.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-/tmp/claude-1111265/depot:/u/ctessum/.julia}"
export RESEACT_RXENV="${RESEACT_RXENV:-/tmp/claude-1111265/conus-run-env}"
TAG="$1"; shift
mkdir -p "$REPO/tools/diag/logs"
LOG="$REPO/tools/diag/logs/$TAG.log"
RES="$REPO/tools/diag/rof_results.tsv"
env "$@" timeout "${ROF_TIMEOUT:-3600}" julia --project="$RESEACT_RXENV" \
    "$REPO/tools/diag/rof_repro.jl" > "$LOG" 2>&1
rc=$?
case $rc in
  0)   verdict=OK ;;
  124) verdict=TIMEOUT ;;
  139) verdict=SEGV ;;
  134) verdict=ABORT ;;
  *)   verdict="EXIT$rc" ;;
esac
printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$TAG" "$verdict" "$*" >> "$RES"
echo "$TAG -> $verdict (rc=$rc)"
