#!/usr/bin/env python3
"""p11_csv_diff.py -- compare two tools/adjoint_gradient.jl CSVs component by
component, and say plainly whether the gradient moved.

The CSV is one row per runtime scalar (`param,theta0,dJ_dtheta,objective,...`)
with the objective value `J` repeated on every row. 111 of the 160 components
are structural zeros (see the driver header), so a RELATIVE metric over all of
them is dominated by 0-vs-0; this reports the nonzero set separately and prints
every component that moved by more than the tolerance.

    python3 tools/diag/p11_csv_diff.py A.csv B.csv [reltol]
"""
import csv, sys, math

def load(path):
    rows = {}
    J = None
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["param"]] = float(r["dJ_dtheta"])
            J = float(r["J"])
    return rows, J

def main(a, b, reltol=1e-12):
    ra, Ja = load(a)
    rb, Jb = load(b)
    ka, kb = set(ra), set(rb)
    if ka != kb:
        print(f"  PARAM SETS DIFFER: only-A {sorted(ka - kb)[:5]} only-B {sorted(kb - ka)[:5]}")
    keys = sorted(ka & kb)
    dJ = abs(Ja - Jb) / max(abs(Ja), abs(Jb), 1e-300)
    print(f"  J  A={Ja!r}  B={Jb!r}   rel {dJ:.3e}   {'IDENTICAL' if Ja == Jb else 'differs'}")
    nz = [k for k in keys if ra[k] != 0.0 or rb[k] != 0.0]
    bits = sum(1 for k in keys if ra[k] == rb[k])
    worst = ("", 0.0)
    ulps = ("", 0)
    for k in nz:
        x, y = ra[k], rb[k]
        rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
        if rel > worst[1]:
            worst = (k, rel)
        if x != y and math.isfinite(x) and math.isfinite(y):
            # ulp distance, for a "how many last bits" statement
            import struct
            ix = struct.unpack("<q", struct.pack("<d", x))[0]
            iy = struct.unpack("<q", struct.pack("<d", y))[0]
            d = abs(ix - iy)
            if d > ulps[1]:
                ulps = (k, d)
    print(f"  components {len(keys)} ({len(nz)} nonzero in either arm); "
          f"bit-for-bit identical {bits}/{len(keys)}")
    print(f"  worst relative over the nonzero set: {worst[1]:.3e}  ({worst[0]})")
    print(f"  worst ulp distance: {ulps[1]}  ({ulps[0]})")
    over = [(k, abs(ra[k] - rb[k]) / max(abs(ra[k]), abs(rb[k]), 1e-300)) for k in nz]
    over = [t for t in over if t[1] > reltol]
    if over:
        print(f"  components over reltol={reltol:g}:")
        for k, rel in sorted(over, key=lambda t: -t[1])[:20]:
            print(f"    {k:<40s} {ra[k]!r:>26s} {rb[k]!r:>26s}  rel {rel:.3e}")
    else:
        print(f"  every nonzero component agrees to better than reltol={reltol:g}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12)
