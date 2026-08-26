#!/usr/bin/env python3
# ===========================================================================
# bucket_replay_segment.py -- price the bucket policies on ONE window segment
# of the recorded per-cell demand matrix, for reading bucket_conus.sbatch
# results against the right ceiling.
#
# WHY. The headline replay figures (stiffness_policy_replay.py: trend 2.91x
# at K=16) are 288-window WHOLE-DAY numbers. bucket_conus.jl measures one
# diurnal segment chosen to contain the sunrise sweep t=42300-52200 -- the
# predictor's known worst case -- and on that segment the same cost model
# predicts materially less (the terminator's regime-change windows spend
# their prediction on penalty). Comparing a sunrise-segment measurement to
# the whole-day ceiling overstates the scheduler's shortfall; this script
# prints the segment's own ceiling, per policy and K, plus the per-window
# prediction so measured spikes can be attributed (the measured K=16 spike
# windows coincide with the replay report's worst prev-vs-oracle penalty
# windows, i.e. they are prediction penalty, not scheduler defects).
#
# The cost model is stiffness_policy_replay.py's: a bucket of B cells pays
# B * max(true demand of members); it charges NOTHING for per-bucket
# controller slop (PI restart each window, rejects), so it is a CEILING for
# a real per-bucket controller, not an estimate of it.
#
# Usage: bucket_replay_segment.py [outdir] [t0] [t1]
#        (defaults: the recorded CONUS matrix, 41400, 53400)
# ===========================================================================
import os
import sys

import numpy as np

outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "stiffdiurnal")
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 41400.0
t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 53400.0

meta = dict(l.strip().split("=", 1) for l in open(os.path.join(outdir, "meta.txt"))
            if "=" in l and not l.startswith("record"))
NC = int(meta["NC"])
rec = 6 + NC
raw = np.fromfile(os.path.join(outdir, "percell_steps.bin"), dtype=np.float64)
M = raw.size // rec
raw = raw[: M * rec].reshape(M, rec)
hdr, S = raw[:, :6], raw[:, 6:]
tcur, tnext, _, _, naC, _ = hdr.T
S = np.maximum(S, 1.0)

seg = np.where((tcur >= t0 - 1e-6) & (tcur < t1 - 1e-6))[0]
if seg.size == 0:
    sys.exit(f"no recorded windows in [{t0}, {t1}]")
print(f"segment: {len(seg)} windows (m={seg[0]}..{seg[-1]}), "
      f"t=[{tcur[seg[0]]:.0f},{tnext[seg[-1]]:.0f}]")
glob_seg = naC[seg].sum() * NC
print(f"recorded actual global cell-steps in segment: {glob_seg:.4g} "
      f"({naC[seg].sum():.0f} accepted steps)\n")

ident = np.arange(NC)


def order(name, m):
    if name == "trend":
        if m > 1:
            return np.argsort(S[m - 1] ** 2 / S[m - 2])
        return np.argsort(S[m - 1]) if m > 0 else ident
    if name == "prev":
        return np.argsort(S[m - 1]) if m > 0 else ident
    return np.argsort(S[m])  # oracle


def bucket_cost(s_true, o, K):
    srt = s_true[o]
    return sum(len(b) * srt[b].max() for b in np.array_split(np.arange(NC), K))


for K in (4, 16, 64):
    for pol in ("prev", "trend", "oracle"):
        tot = sum(bucket_cost(S[m], order(pol, m), K) for m in seg)
        print(f"  K={K:3d} {pol:6s}: predicted cost {tot:.4g}  "
              f"=> {glob_seg / tot:.2f}x vs actual global (segment ceiling)")
print("\nper-window trend K=16 ceiling vs recorded actual:")
for m in seg:
    c = bucket_cost(S[m], order("trend", m), 16)
    print(f"  t=[{tcur[m]:.0f},{tnext[m]:.0f}]  ceiling {naC[m] * NC / c:.2f}x  "
          f"(pred {c:.3e}, actual {naC[m] * NC:.3e})")
