#!/usr/bin/env python3
"""Portable, (NLON,NLAT)-parameterized driver for the full ReSEACT model.

Chains the three committed generators that build the full transport+chemistry model
(prototypes/transport_3d/gen_t3d.py -> prototypes/reseact_3d/gen_r3d.py ->
prototypes/reseact_3d_chem/gen_c1.py), but:
  * derives all paths from this repo's location (no hard-coded absolute paths),
  * parameterizes the horizontal grid (NLON, NLAT) via env, keeping NLEV=72,
  * chains SRC/OUT through a private staging dir (never touches the committed .esm),
  * re-relativizes the final top-level refs (ESD rules, geosfp.esm, superfast.esm)
    so they resolve relative to the OUTPUT file's own directory.

The per-generator transformation logic is left byte-identical; only the module-level
path/size constants are rewritten. At NLON=NLAT=7, NLEV=72 the output is bit-identical
(ref-normalized) to prototypes/reseact_3d_chem/reseact_3d_chem.esm.

Sibling repos EarthSciDiscretizations and EarthSciModels must be checked out next to
this one (i.e. under the same parent directory as reseact.esm).

Grid limits (native GEOS-FP 4x5 window at LON_OFF=14, LAT_OFF=29): NLON <= 57, NLAT <= 17.

Usage:
  NLON=14 NLAT=14 OUT=/abs/path/reseact.esm python3 tools/build_full_model.py
"""
import os, re, json, subprocess, sys, tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(TOOLS)                 # .../reseact.esm
CODE  = os.path.dirname(REPO)                  # parent holding the sibling repos
PROTO = os.path.join(REPO, "prototypes")
NLON  = os.environ.get("NLON", "7")
NLAT  = os.environ.get("NLAT", "7")
NLEV  = os.environ.get("NLEV", "72")
OUT   = os.environ["OUT"]

STAGE = os.environ.get("STAGE") or tempfile.mkdtemp(prefix="reseact_build_")
os.makedirs(STAGE, exist_ok=True)

def portable(src_path, subs):
    """Read a generator, apply (pattern, repl) line substitutions, write to STAGE."""
    txt = open(src_path).read()
    for pat, repl in subs:
        new, n = re.subn(pat, repl, txt, flags=re.M)
        if n == 0:
            raise SystemExit(f"substitution failed in {src_path}: {pat!r}")
        txt = new
    dst = os.path.join(STAGE, os.path.basename(src_path))
    open(dst, "w").write(txt)
    return dst

def run(script, cwd):
    print(f"--- running {os.path.basename(script)} (NLON={NLON} NLAT={NLAT} NLEV={NLEV}) ---")
    subprocess.run([sys.executable, script], cwd=cwd, check=True)

# 1. gen_t3d.py : base transport, horizontal grid bindings = NLON/NLAT
t3d = portable(f"{PROTO}/transport_3d/gen_t3d.py", [
    (r'^SRC = ".*"$',  f'SRC = "{CODE}/EarthSciDiscretizations/problems/latlon3d_transport_cwc_regional_inflow.esm"'),
    (r'^OUT = ".*"$',  f'OUT = "{STAGE}/transport_3d.esm"'),
    (r'^NLON=NLAT=NLEV=7$', f'NLON={NLON}; NLAT={NLAT}; NLEV=7  # base NLEV; gen_r3d rebinds to {NLEV}'),
])
run(t3d, STAGE)   # cwd=STAGE so hybrid_coefs.json lands there, not in the repo

# 2. gen_r3d.py : GEOS-FP native slice forcing, rebinds NLEV=72, uses NLON/NLAT offsets
r3d = portable(f"{PROTO}/reseact_3d/gen_r3d.py", [
    (r'^SRC = ".*"$', f'SRC = "{STAGE}/transport_3d.esm"'),
    (r'^OUT = ".*"$', f'OUT = "{STAGE}/reseact_3d.esm"'),
    (r'^NLON = NLAT = 7$', f'NLON = {NLON}; NLAT = {NLAT}'),
    (r'^NLEV = 72\b', f'NLEV = {NLEV}'),
])
run(r3d, STAGE)

# 3. gen_c1.py : SuperFast chemistry lift (grid-agnostic; inherits grid from r3d)
c1 = portable(f"{PROTO}/reseact_3d_chem/gen_c1.py", [
    (r'^SRC = ".*"$',    f'SRC = "{STAGE}/reseact_3d.esm"'),
    (r'^OUT = ".*"$',    f'OUT = "{OUT}"'),
    (r'^SF_ABS = ".*"$', f'SF_ABS = "{CODE}/EarthSciModels/components/gaschem/superfast.esm"'),
])
run(c1, f"{PROTO}/reseact_3d_chem")

# 4. re-relativize the top-level refs so they resolve from OUT's directory.
#    The generators embed "../../../<Repo>/..." (correct for a 3-deep prototype dir).
#    Resolve each against that 3-deep base, then re-relativize against OUT's dir.
STAGE_BASE = f"{PROTO}/reseact_3d_chem"          # the dir the embedded "../../../" assume
out_dir = os.path.dirname(os.path.abspath(OUT))
doc = json.load(open(OUT))
n_fixed = 0
def fix_ref(r):
    global n_fixed
    if not isinstance(r, str) or r.startswith(("http://", "https://")) or os.path.isabs(r):
        return r
    absolute = os.path.normpath(os.path.join(STAGE_BASE, r))
    rel = os.path.relpath(absolute, out_dir)
    if rel != r:
        n_fixed += 1
    return rel
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "ref" and isinstance(v, str):
                o[k] = fix_ref(v)
            else:
                walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(doc)
with open(OUT, "w") as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"wrote {OUT}  (re-relativized {n_fixed} refs against {out_dir})")
