#!/usr/bin/env bash
# One configuration per process; the verdict comes from the PARENT because a
# SIGSEGV cannot report its own death. 139=SIGSEGV, 134=SIGABRT, 124=timeout.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export RESEACT_RXENV="${RESEACT_RXENV:-$REPO/run-model-jl}"
LOGD="$REPO/tools/diag/logs/mwe_binomial_segv"; mkdir -p "$LOGD"
RES="$REPO/tools/diag/mwe_binomial_segv_results.tsv"
: > "$RES"
printf 'variant\tbudget\tnstep\traise\trc\tverdict\tstatus\trel\n' >> "$RES"
run1() {  # variant budget nstep raise
  local v=$1 n=$2 s=$3 r=$4 tag="${1}-n${2}-s${3}-raise${4}"
  MWE_VARIANT=$v MWE_BUDGET=$n MWE_NSTEP=$s MWE_RAISE=$r \
    timeout "${MWE_TIMEOUT:-900}" julia --project="$RESEACT_RXENV" \
    "$REPO/tools/diag/mwe_binomial_segv.jl" > "$LOGD/$tag.log" 2>&1
  local rc=$? verdict line status rel
  case $rc in 0) verdict=RAN ;; 124) verdict=TIMEOUT ;; 139) verdict=SEGV ;;
              134) verdict=ABORT ;; *) verdict="EXIT$rc" ;; esac
  line=$(grep -m1 '^RESULT ' "$LOGD/$tag.log" || true)
  status=$(sed -n 's/.*status=\([^ ]*\).*/\1/p' <<<"$line"); status=${status:-none}
  rel=$(sed -n 's/.*rel=\([^ ]*\).*/\1/p' <<<"$line"); rel=${rel:-none}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$v" "$n" "$s" "$r" "$rc" "$verdict" "$status" "$rel" >> "$RES"
  printf '%-30s rc=%-4s %-8s %-13s rel=%s\n' "$tag" "$rc" "$verdict" "$status" "$rel"
}
echo "=== the ORIGINAL spelling, verbatim (raise=1, NSTEP=8, Binomial(3)) ==="
run1 none            0 8 1
run1 true            0 8 1
run1 periodic        3 8 1
run1 binomial        3 8 1
run1 binomial_mincut 3 8 1
echo "=== the same without the raising pipeline ==="
run1 binomial        3 8 0
run1 binomial_mincut 3 8 0
echo "=== budget / trip-count sweep, raise on ==="
for b in 2 4 5 8; do run1 binomial "$b" 8 1; done
for s in 4 16 20 64; do run1 binomial 3 "$s" 1; done
echo; echo "=== $RES ==="; column -t "$RES"
