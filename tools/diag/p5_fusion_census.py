#!/usr/bin/env python3
"""p5_fusion_census.py -- what does each XLA:CPU fusion of a dumped program DO,
and how much does it read and write?

Reads the largest `*after_optimizations.txt` under each program directory of a
p5_vjp_dump.jl output (logs/p5-dump-<grid>-<job>/<prog>/) and, for the ENTRY
computation, classifies every fusion by the ROOT opcode of its fused computation
and its result shape, summing result elements (bytes written) and operand
elements (bytes read, upper bound: a DUS-root fusion writes only its update).
The question it answers (2026-09-05): is the transport VJP's 22x-over-primal
cost a few thousand whole-state accumulation passes, and if so which ones?

    python3 tools/diag/p5_fusion_census.py logs/p5-dump-6x6x8-10356482 [prog ...]
"""
import re, sys, os, glob, collections

def elems(ty):
    m = re.search(r'\b[a-z0-9]+\[([0-9,]*)\]', ty)
    if not m: return 0
    c = m.group(1)
    if c == '': return 1
    p = 1
    for x in c.split(','): p *= int(x)
    return p

INSTR = re.compile(r'^\s*(?:ROOT\s+)?(%[\w.\-]+)\s*=\s*(\S+(?:\s+\S+)?)\s+([a-z\-]+)\((.*)$')

def parse(txt):
    lines = txt.split('\n')
    # fused computations: name -> root opcode, body op count
    comps = {}
    cur = None
    for l in lines:
        m = re.match(r'^(%[\w.\-]+)\s+\(.*\)\s*->\s*.*\{\s*$', l)
        if m:
            cur = m.group(1); comps[cur] = {"root": None, "n": 0}; continue
        if cur and l.strip() == '}':
            cur = None; continue
        if cur:
            mi = INSTR.match(l)
            if mi:
                comps[cur]["n"] += 1
                if l.lstrip().startswith("ROOT"): comps[cur]["root"] = mi.group(3)
    i0 = next(i for i, l in enumerate(lines) if l.startswith("ENTRY"))
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == "")
    shape = {}
    entry = []
    for l in lines[i0 + 1:i1]:
        mi = INSTR.match(l)
        if not mi: continue
        name, ty, op, rest = mi.groups()
        shape[name] = elems(ty)
        entry.append((name, ty, op, rest))
    return comps, shape, entry

def main(root, progs):
    for prog in progs:
        d = os.path.join(root, prog)
        fs = glob.glob(os.path.join(d, "*after_optimizations.txt"))
        if not fs: print(f"== {prog}: no dump"); continue
        f = max(fs, key=os.path.getsize)
        comps, shape, entry = parse(open(f).read())
        cls = collections.defaultdict(lambda: [0, 0, 0])  # count, write elems, read elems
        nfus = 0; wtot = 0; rtot = 0
        other = collections.defaultdict(lambda: [0, 0])
        for name, ty, op, rest in entry:
            # operands: the leading parenthesised list of %refs
            ops_txt = rest.split(')')[0]
            rd = sum(shape.get(r, 0) for r in re.findall(r'%[\w.\-]+', ops_txt))
            if op == "fusion":
                mc = re.search(r'calls=(%[\w.\-]+)', rest)
                c = comps.get(mc.group(1), {"root": "?", "n": 0}) if mc else {"root": "?", "n": 0}
                key = (c["root"], ty.split('{')[0])
                w = shape[name]
                cls[key][0] += 1; cls[key][1] += w; cls[key][2] += rd
                nfus += 1; wtot += w; rtot += rd
            elif op not in ("parameter", "constant", "get-tuple-element", "tuple", "bitcast"):
                other[op][0] += 1; other[op][1] += shape[name]
        print(f"\n== {prog}: {os.path.basename(f)}")
        print(f"   fusions {nfus}: write {wtot*8e-6:.1f} MB/call, read(upper) {rtot*8e-6:.1f} MB/call")
        print(f"   {'root op':<22s} {'result shape':<22s} {'n':>5s} {'write MB':>9s} {'read MB':>9s}")
        for (r, s), (n, w, rd) in sorted(cls.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:18]:
            print(f"   {str(r):<22s} {s:<22s} {n:5d} {w*8e-6:9.2f} {rd*8e-6:9.2f}")
        print("   non-fused ops (n, result MB):")
        for op, (n, w) in sorted(other.items(), key=lambda kv: -kv[1][1])[:8]:
            print(f"      {op:<22s} {n:5d} {w*8e-6:9.2f}")

if __name__ == "__main__":
    root = sys.argv[1]
    progs = sys.argv[2:] or ["ssp_step", "ssp_vjp", "ros_step", "ros_vjp"]
    main(root, progs)
