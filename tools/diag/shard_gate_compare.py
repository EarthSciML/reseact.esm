#!/usr/bin/env python3
"""shard_gate_compare.py REF.csv OTHER.csv [OTHER2.csv ...]

Compare the adjoint driver's gradient CSVs component by component: exact
string equality of the printed dJ/dtheta (the "identical to every printed
digit" gate), the worst relative difference over nonzero components, and any
component that is nonzero in one file and zero in the other (a structural
change, which no tolerance should forgive)."""
import csv, sys

def load(p):
    rows = list(csv.DictReader(open(p)))
    return {r['param']: (r['dJ_dtheta'], float(r['dJ_dtheta'])) for r in rows}, rows[0]['J']

ref, Jref = load(sys.argv[1])
print(f"reference {sys.argv[1]}: {len(ref)} components, J={Jref}")
for other in sys.argv[2:]:
    o, Jo = load(other)
    same = sum(1 for k in ref if k in o and o[k][0] == ref[k][0])
    worst = 0.0; wk = ''; structural = []
    for k, (s, v) in ref.items():
        if k not in o:
            structural.append((k, 'MISSING')); continue
        w = o[k][1]
        if (v == 0) != (w == 0):
            structural.append((k, f'{v} vs {w}')); continue
        if v != 0:
            d = abs(v - w) / max(abs(v), abs(w))
            if d > worst: worst, wk = d, k
    nz = sum(1 for k in ref if ref[k][1] != 0)
    print(f"{other}: J={Jo} ({'same digits' if Jo == Jref else 'DIFFERS'}); "
          f"{same}/{len(ref)} components identical to every printed digit; "
          f"{nz} nonzero; worst relative diff {worst:.3e} at {wk or '-'}; "
          f"structural changes: {structural if structural else 'none'}")
