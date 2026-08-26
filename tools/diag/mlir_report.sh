#!/bin/bash
# Summarize every .mlir in a dump dir: raw op count, GVN-removable count, and the
# residual -- i.e. how much of the emitter's output is redundancy XLA has to
# rediscover pairwise. Pass the dump directory.
D=${1:?usage: mlir_report.sh <dumpdir> [--dups N]}
shift
HERE=$(dirname "$0")
printf "%-34s %10s %10s %10s %8s\n" file ops gvn_dupes residual factor
for f in "$D"/*.mlir; do
  "$HERE/mlir_gvn.py" "$f" --top 0 2>/dev/null | awk -v n="$(basename "$f" .mlir)" '
    /ops in dump/            {t=$4}
    /removable by pure GVN/  {d=$5}
    /residual after GVN/     {r=$4; f=$NF}
    END {printf "%-34s %10d %10d %10d %8s\n", n, t, d, r, f}'
done
