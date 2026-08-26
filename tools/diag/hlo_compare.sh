#!/bin/bash
# Before/after table for one hlo_dump output directory produced with the
# ess-oop-* variant sweep. All variants share ONE build, so the rows are
# directly comparable. Op counts use the same census as mlir_gvn.py (every
# `%x = dialect.op` line, all dialects), which is what hlo_dump.jl reports.
D=${1:?usage: hlo_compare.sh <dumpdir>}
count() { [ -f "$1" ] && grep -cE '=[[:space:]]+"?[a-zA-Z_][a-zA-Z_0-9]*\.[a-zA-Z_0-9.]+' "$1" || echo 0; }
ident() { [ -f "$1" ] || { echo 0; return; }; python3 - "$1" <<'PY'
import re,sys
n=0
for l in open(sys.argv[1]):
    if 'broadcast_in_dim' not in l and 'stablehlo.transpose' not in l: continue
    m=re.search(r'\(tensor<([^>]+)>\) -> tensor<([^>]+)>', l)
    if m and m.group(1)==m.group(2): n+=1
print(n)
PY
}
printf "%-44s %10s %10s %12s\n" program unopt opt identity-ops
for prog in rhs jac step; do
  for v in "" "ESS_OOP_NATIVE0." "ESS_OOP_GVN0." "ESS_OOP_GVN0_ESS_OOP_NATIVE0."; do
    f="$D/${v}${prog}.unopt.mlir"; o="$D/${v}${prog}.opt.mlir"
    [ -f "$f" ] || continue
    printf "%-44s %10s %10s %12s\n" "${v:-default.}$prog" "$(count "$f")" "$(count "$o")" "$(ident "$f")"
  done
done
