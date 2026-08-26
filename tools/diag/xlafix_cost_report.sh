#!/bin/bash
# ===========================================================================
# xlafix_cost_report.sh -- reduce a xlafix_cost_conus.sbatch paired log to the
# numbers the keep-or-flip decision turns on.
#
#   ./tools/diag/xlafix_cost_report.sh logs/xfixcost-<jobid>.out
# ===========================================================================
set -uo pipefail
LOG=${1:?usage: xlafix_cost_report.sh <paired log>}
JOB=$(basename "$LOG" .out | sed 's/^xfixcost-//')
DIR=$(dirname "$LOG")

split_leg () {   # $1 = 1 or 0
    awk -v fix="$1" '
        /^### LEG XLAFIX=/ { inleg = ($0 ~ ("XLAFIX=" fix "  ")) }
        inleg { print }
    ' "$LOG"
}

for fix in 1 0; do
    echo "================ LEG XLAFIX=$fix ================"
    split_leg "$fix" | grep -E "^BUILD |prepare_jacobian |@compile |forward pass |accepted inner steps|^  J = |COST RATIO|structural identity|Maximum resident set size|Elapsed \(wall clock\)|LEG XLAFIX|src hash"
    echo
done

echo "================ accept/reject ladder ================"
for fix in 1 0; do
    split_leg "$fix" | grep -A1 "accept/reject per macro step" | tail -1 > "$DIR/.ladder-$fix-$JOB"
done
if cmp -s "$DIR/.ladder-1-$JOB" "$DIR/.ladder-0-$JOB"; then
    echo "IDENTICAL -- the controller made every accept/reject decision the same way"
else
    echo "DIFFER -- the two legs integrated different trajectories:"
    diff <(tr ' ' '\n' < "$DIR/.ladder-1-$JOB") <(tr ' ' '\n' < "$DIR/.ladder-0-$JOB") | head -20
fi

echo
echo "================ J ================"
J1=$(split_leg 1 | grep -m1 "^  J = " | awk '{print $3}')
J0=$(split_leg 0 | grep -m1 "^  J = " | awk '{print $3}')
python3 - "$J1" "$J0" <<'PY'
import sys
try:
    a, b = float(sys.argv[1]), float(sys.argv[2])
except ValueError:
    print("J not found in one or both legs:", sys.argv[1:]); raise SystemExit
print(f"  XLAFIX=1  J = {a!r}")
print(f"  XLAFIX=0  J = {b!r}")
d = (a - b) / b
print(f"  relative (J_on - J_off)/J_off = {d:.3e}")
print(f"  recorded 2026-08-19 CONUS figure, for scale:  +4.0e-05")
import struct
u = lambda x: struct.unpack('<q', struct.pack('<d', x))[0]
print(f"  ulps apart = {abs(u(a) - u(b))}")
PY

echo
echo "================ gradient ================"
C1="$DIR/xfix1-$JOB.csv"; C0="$DIR/xfix0-$JOB.csv"
if [ -f "$C1" ] && [ -f "$C0" ]; then
python3 - "$C1" "$C0" <<'PY'
import sys, csv
def load(p):
    d = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            k = row.get("param") or row.get("parameter") or list(row.values())[0]
            for c in ("dJ_dtheta", "grad", "dJdtheta"):
                if c in row:
                    d[k] = float(row[c]); break
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = [k for k in a if k in b]
worst = 0.0; wk = None
for k in keys:
    den = max(abs(a[k]), abs(b[k]))
    if den == 0: continue
    r = abs(a[k] - b[k]) / den
    if r > worst: worst, wk = r, k
print(f"  {len(keys)} shared components; worst relative difference {worst:.3e} at {wk}")
print(f"  nonzero components: XLAFIX=1 {sum(1 for k in a if a[k] != 0)}, XLAFIX=0 {sum(1 for k in b if b[k] != 0)}")
PY
else
    echo "  (csv not found: $C1 / $C0)"
fi

echo
echo "================ memory ================"
grep -E "memory.max|memory.peak|sampled memory.current" "$LOG"
awk -v lg="$LOG" '
  BEGIN { while ((getline l < lg) > 0) {
            if (l ~ /^### LEG XLAFIX=1 .*start/)  { split(l, f, "epoch "); s1 = f[2]+0 }
            if (l ~ /^### LEG XLAFIX=1 .*end/)    { split(l, f, "epoch "); e1 = f[2]+0 }
            if (l ~ /^### LEG XLAFIX=0 .*start/)  { split(l, f, "epoch "); s0 = f[2]+0 }
            if (l ~ /^### LEG XLAFIX=0 .*end/)    { split(l, f, "epoch "); e0 = f[2]+0 } } }
  { t = $1+0; m = $2+0
    if (t >= s1 && t <= e1 && m > p1) p1 = m
    if (t >= s0 && t <= e0 && m > p0) p0 = m }
  END { printf "  cgroup memory.current peak within LEG XLAFIX=1: %.2f GiB\n", p1/1073741824
        printf "  cgroup memory.current peak within LEG XLAFIX=0: %.2f GiB\n", p0/1073741824
        if (p0 > 0) printf "  ratio on/off = %.2fx\n", p1/p0 }
' "$DIR/xfixcost-$JOB.memtrace"
