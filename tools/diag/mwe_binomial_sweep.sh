#!/usr/bin/env bash
# Drive tools/diag/mwe_binomial_ckpt.jl ONE CONFIGURATION PER PROCESS and take
# the verdict from OUTSIDE the process -- a segfault cannot report its own
# death (claim B2). Exit 139 = SIGSEGV, 134 = SIGABRT, 124 = timeout.
#
#   tools/diag/mwe_binomial_sweep.sh
#
# Writes tools/diag/mwe_binomial_results.tsv and per-config logs under
# tools/diag/logs/mwe_binomial/.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export RESEACT_RXENV="${RESEACT_RXENV:-$REPO/run-model-jl}"
LOGD="$REPO/tools/diag/logs/mwe_binomial"; mkdir -p "$LOGD"
RES="$REPO/tools/diag/mwe_binomial_results.tsv"
: > "$RES"
printf 'bound\tckpt\tbudget\ttrips\trc\tverdict\tstatus\trel\n' >> "$RES"

run1() {  # bound ckpt budget trips
  local b=$1 c=$2 n=$3 t=$4
  local tag="${b}-${c}-n${n}-t${t}"
  MWE_BOUND=$b MWE_CKPT=$c MWE_BUDGET=$n MWE_TRIPS=$t MWE_HLO=${MWE_HLO:-1} \
    timeout "${MWE_TIMEOUT:-900}" julia --project="$RESEACT_RXENV" \
    "$REPO/tools/diag/mwe_binomial_ckpt.jl" > "$LOGD/$tag.log" 2>&1
  local rc=$?
  local verdict
  case $rc in
    0) verdict=RAN ;; 124) verdict=TIMEOUT ;; 139) verdict=SEGV ;;
    134) verdict=ABORT ;; *) verdict="EXIT$rc" ;;
  esac
  local line status rel
  line=$(grep -m1 '^RESULT ' "$LOGD/$tag.log" || true)
  status=$(sed -n 's/.*status=\([^ ]*\).*/\1/p' <<<"$line"); status=${status:-none}
  rel=$(sed -n 's/.*rel=\([^ ]*\).*/\1/p' <<<"$line"); rel=${rel:-none}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$b" "$c" "$n" "$t" "$rc" "$verdict" "$status" "$rel" >> "$RES"
  printf '%-28s rc=%-4s %-8s %-13s rel=%s\n' "$tag" "$rc" "$verdict" "$status" "$rel"
}

echo "=== controls: no checkpointing ==="
run1 static none     0 20
run1 traced none     0 20

echo "=== B1: TRACED bound + Binomial(n), sweeping the budget ==="
for n in 2 3 4 5 6 8 10 12 16; do run1 traced binomial "$n" 20; done

echo "=== B1 control: TRACED bound + Periodic(n) ==="
for n in 2 5 10; do run1 traced periodic "$n" 20; done

echo "=== B2: STATIC bound + Binomial(n)  (expect SIGSEGV per the claim) ==="
for n in 2 5 10; do run1 static binomial "$n" 20; done

echo "=== B2 controls: STATIC bound + Periodic / auto ==="
for n in 5; do run1 static periodic "$n" 20; done
run1 static auto 0 20
run1 traced auto 0 20

echo
echo "=== $RES ==="
column -t "$RES"
