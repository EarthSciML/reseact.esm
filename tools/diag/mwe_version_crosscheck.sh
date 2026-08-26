#!/usr/bin/env bash
# ===========================================================================
# mwe_version_crosscheck.sh -- run the Target A and Target B probes against a
# PINNED OLD Reactant, so a "no longer reproduces" verdict is backed by a
# positive control rather than by absence of evidence.
# ===========================================================================
# Two of the three claims investigated here were recorded on Reactant 0.2.274 /
# Reactant_jll 0.0.395+1 and do NOT fire on 0.2.280. That is only credible if
# the same probe DOES fire on 0.2.274 -- otherwise it might just be a probe
# that never tested anything. This builds a throwaway environment pinned to
# 0.2.274 in a scratch depot STACKED on top of the shared one (so every package
# resolves from the shared depot but nothing is written into it) and runs the
# probes there.
#
#   tools/diag/mwe_version_crosscheck.sh [RXVERSION]     # default 0.2.274
#
# Measured 2026-08-24, Julia 1.12.6, CPU/PJRT:
#
#   probe                              0.2.274 / jll 0.0.395+1   0.2.280 / jll 0.0.405+0
#   mwe_case_reverse   Ops.case rev    FAIL "could not compute   FAIL (identical)
#                                            the adjoint"
#   mwe_binomial_ckpt  traced+Binom(5) WRONG rel 1.05e-3         EXACT rel 2.1e-15
#   mwe_binomial_segv  static+Binom(3) SIGSEGV (rc 139) in       EXACT rel 3.8e-16
#                                      BinomialProgressConstProp
# ===========================================================================
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RXV="${1:-0.2.274}"
BASE="${MWE_XCHECK_DIR:-${TMPDIR:-/tmp}/mwe-xcheck-$RXV}"
ENVD="$BASE/env"; DEPOT="$BASE/depot"
SHARED_DEPOT="${MWE_SHARED_DEPOT:-/projects/illinois/eng/cee/ctessum/ctessum/.julia}"
mkdir -p "$ENVD" "$DEPOT"
export JULIA_DEPOT_PATH="$DEPOT:$SHARED_DEPOT"
export RESEACT_RXENV="$ENVD"
LOGD="$REPO/tools/diag/logs/mwe_xcheck_$RXV"; mkdir -p "$LOGD"

echo "=== building a pinned Reactant $RXV env in $ENVD (scratch depot $DEPOT) ==="
julia --project="$ENVD" -e "using Pkg; Pkg.add(name=\"Reactant\", version=\"$RXV\"); Pkg.status()" \
      > "$LOGD/build.log" 2>&1 || { echo "env build FAILED, see $LOGD/build.log"; exit 1; }
grep -E '^\s+\[' "$LOGD/build.log" | head -5

for probe in mwe_case_reverse mwe_binomial_ckpt mwe_binomial_segv; do
  echo; echo "########## $RXV :: $probe ##########"
  case $probe in
    mwe_binomial_ckpt) env MWE_BOUND=traced MWE_CKPT=binomial MWE_BUDGET=5 MWE_TRIPS=20 MWE_HLO=0 \
        timeout "${MWE_TIMEOUT:-1800}" julia --project="$ENVD" "$REPO/tools/diag/$probe.jl" \
        > "$LOGD/$probe.log" 2>&1 ;;
    mwe_binomial_segv) env MWE_VARIANT=binomial MWE_BUDGET=3 MWE_NSTEP=8 MWE_RAISE=1 \
        timeout "${MWE_TIMEOUT:-1800}" julia --project="$ENVD" "$REPO/tools/diag/$probe.jl" \
        > "$LOGD/$probe.log" 2>&1 ;;
    *) timeout "${MWE_TIMEOUT:-1800}" julia --project="$ENVD" "$REPO/tools/diag/$probe.jl" \
        > "$LOGD/$probe.log" 2>&1 ;;
  esac
  rc=$?
  case $rc in 0) v=RAN ;; 124) v=TIMEOUT ;; 139) v="SEGV (SIGSEGV)" ;; 134) v=ABORT ;; *) v="EXIT$rc" ;; esac
  echo "  verdict: $v   (log: $LOGD/$probe.log)"
  grep -E '^RESULT |^\s+\[(PASS|FAIL)\]' "$LOGD/$probe.log" | head -20
  [ "$rc" = 139 ] && grep -m1 -i 'BinomialProgress' "$LOGD/$probe.log" | cut -c1-160
done
echo; echo "logs in $LOGD ; throwaway env in $BASE (delete when done)"
