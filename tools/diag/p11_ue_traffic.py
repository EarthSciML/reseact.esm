#!/usr/bin/env python3
"""p11_ue_traffic.py -- how much of a dumped XLA:CPU program is whole-EXTENDED-
BUFFER traffic, and how much of that is the observed prelude's adjoint?

p5_fusion_census.py classifies every ENTRY fusion by the root opcode of its
fused computation. This one asks the narrower question the ess-oop-levelbase
change is about: of the fusions whose RESULT is one whole extended value vector
(`ue`, the flat state-plus-materialized-observeds buffer), how many are there,
what are they rooted at, and how many `copy` instructions does copy-insertion
add alongside them? Those three numbers are the before/after census of the
change -- a zeroing `broadcast`-rooted DUS over the whole buffer costs
`n_total` element writes to zero a few slots, and there was one per fill.

    python3 tools/diag/p11_ue_traffic.py <dumpdir> [<dumpdir> ...]

`<dumpdir>` is one program directory of a p5_vjp_dump.jl / p7_rhs_vjp.jl run
(it reads the largest `*after_optimizations.txt` inside).
"""
import re, sys, os, glob, collections


def elems(ty):
    m = re.search(r'\[([0-9,]*)\]', ty)
    if not m:
        return 1
    c = m.group(1)
    if c == '':
        return 1
    p = 1
    for x in c.split(','):
        p *= int(x)
    return p


INSTR = re.compile(r'^\s*(?:ROOT\s+)?(%[\w.\-]+)\s*=\s*(\S+)\s+([a-z\-]+)\((.*)$')


def report(d):
    fs = glob.glob(os.path.join(d, "*after_optimizations.txt"))
    if not fs:
        print(f"== {d}: no dump")
        return
    f = max(fs, key=os.path.getsize)
    lines = open(f).read().split('\n')
    # root opcode of every fused computation
    comps, cur = {}, None
    for l in lines:
        m = re.match(r'^(%[\w.\-]+)\s+\(.*\)\s*->\s*.*\{\s*$', l)
        if m:
            cur = m.group(1)
            comps[cur] = None
            continue
        if cur and l.strip() == '}':
            cur = None
            continue
        if cur and l.lstrip().startswith("ROOT"):
            mi = INSTR.match(l)
            if mi:
                comps[cur] = mi.group(3)
    i0 = next(i for i, l in enumerate(lines) if l.startswith("ENTRY"))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == "")
    fus, copies, shapes = [], [], collections.Counter()
    for l in lines[i0 + 1:i1]:
        mi = INSTR.match(l)
        if not mi:
            continue
        name, ty, op, rest = mi.groups()
        n = elems(ty)
        if op == "fusion":
            mc = re.search(r'calls=(%[\w.\-]+)', rest)
            mo = re.match(r'^\s*(%[\w.\-]+)', rest)
            fus.append((name, n, comps.get(mc.group(1)) if mc else "?",
                        mo.group(1)[1:] if mo else None))
            shapes[n] += 1
        elif op == "copy":
            copies.append((name, n))
    # the extended buffer is the largest whole-vector shape that appears often
    big = max((n for n, c in shapes.items() if c >= 3), default=0)
    tot = sum(t[1] for t in fus)
    at = [t for t in fus if t[1] == big]
    print(f"\n== {os.path.basename(d)}  ({os.path.basename(f)})")
    print(f"   ENTRY fusions {len(fus)}   result elements {tot/1e6:.2f} M")
    print(f"   extended-buffer length taken as {big} ({big*8e-6:.2f} MB)")
    print(f"   whole-buffer fusions {len(at)}   {sum(t[1] for t in at)/1e6:.2f} M "
          f"({100*sum(t[1] for t in at)/max(tot,1):.1f}% of element writes)")
    by = collections.Counter()
    for t in at:
        by[t[2]] += 1
    for r, c in by.most_common():
        print(f"      root {str(r):<24s} n {c:4d}   {c*big/1e6:8.2f} M")
    cb = [n for _, n in copies if n == big]
    print(f"   copy instructions {len(copies)} ({sum(n for _, n in copies)*8e-6:.1f} MB), "
          f"of which whole-buffer {len(cb)} ({len(cb)*big*8e-6:.1f} MB)")
    # IN PLACE OR NOT, which is what the element counts above hinge on: a
    # DUS-rooted fusion whose result shares its operand's slot writes only its
    # own update; one that does not rewrites the whole buffer. XLA:CPU can only
    # take the in-place path when the operand is dead after the write, so this
    # is the direct measure of whether the write chain is single-use.
    ba = glob.glob(os.path.join(d, "*buffer-assignment.txt"))
    if not ba:
        return
    slot = {}
    alloc = None
    for l in open(max(ba, key=os.path.getsize)):
        m = re.match(r'^allocation (\d+):', l)
        if m:
            alloc = m.group(1)
            continue
        m = re.match(r'^\s*value: <\d+ ([\w.\-]+) @0> \(size=(\d+),offset=(\d+)\)', l)
        if m and alloc is not None:
            slot[m.group(1)] = (alloc, m.group(3))
    ip = collections.Counter()
    mat = collections.Counter()
    unknown = 0
    for nm, n, r, op0 in at:
        a = slot.get(nm.lstrip('%'))
        b = slot.get(op0) if op0 else None
        if a is None or b is None:
            unknown += 1
        elif a == b:
            ip[r] += 1
        else:
            mat[r] += 1
    print(f"   of those, by whether the result ALIASES its operand 0 (XLA:CPU then "
          f"writes only the update; unresolved {unknown}):")
    for r in sorted(set(ip) | set(mat)):
        print(f"      root {str(r):<24s} aliased {ip[r]:4d}   materialising {mat[r]:4d}"
              f"   {mat[r]*big*8e-6:8.1f} MB")
    nmat = sum(mat.values())
    real = (tot - sum(t[1] for t in at) + nmat * big
            + sum(n for _, n in copies))
    print(f"   REAL element writes (aliased whole-buffer fusions counted at ~0): "
          f"{real/1e6:.2f} M ({real*8e-6:.1f} MB), of which extended-buffer "
          f"{(nmat*big + len(cb)*big)/1e6:.2f} M "
          f"({100*(nmat*big + len(cb)*big)/max(real,1):.1f}%)")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
