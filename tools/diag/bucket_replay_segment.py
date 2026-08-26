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

# ===========================================================================
# WALL PRICER (Phase 1.5). Cell-step ratios above ignore what the CONUS
# measurement showed actually kills the wall: small batches cost more PER
# LANE than the full-grid program, and padding to a rung multiplies that.
# This section prices partitions in SECONDS using the measured per-call cost
# of the three compiled programs from the Phase-1 CONUS job (fitted below as
# a log-linear per-lane curve through the three (C, us/lane) points), and
# answers, BEFORE any node time is spent: can ANY partition beat lockstep's
# wall on this segment under the ceiling model (zero controller slop)?
#
#   partitions priced:
#     equal-K        the Phase-1 scheme (model validation against the run)
#     lazy+bands     one lazy bucket (predicted demand <= the q quantile) at
#                    a near-full-grid rung, stiff tail in equal-LOG-width
#                    demand bands, bands merged upward rather than padded
#                    below the C=128 rung floor
#     DP-optimal     best contiguous partition of the prediction-sorted
#                    cells under the cost model (stride-quantized break-
#                    points) -- the model's own ceiling over ALL such schemes
#
# Charged per bucket per window: steps = max TRUE demand over members
# (ceiling: zero restart slop, zero rejects), cost = steps x callcost(rung).
# The lockstep baseline is the MEASURED segment wall. Slop multipliers show
# how the verdict degrades if Lever 1 only partly removes the measured
# ~1.8x step slop.
# ===========================================================================
print("\n" + "=" * 74)
print("WALL PRICER (measured per-lane cost curve, ceiling step counts)")

# (C, us per lane) measured on slurm 10155020 (one node, 8 threads):
#   6552 lanes: 270.65 s / 3208 calls   512: 408.37 s / 44204   128: 408.35 s / 115752
_pts_C = np.array([128.0, 512.0, 6552.0])
_pts_us = np.array([408.35e6 / 115752 / 128, 408.37e6 / 44204 / 512,
                    270.65e6 / 3208 / 6552])
LOCK_WALL_MEAS = 270.65      # s, same segment, same node


def perlane_us(C):
    return float(np.interp(np.log(C), np.log(_pts_C), _pts_us))


def callcost_s(C):           # one step of a C-lane program, seconds
    return C * perlane_us(C) * 1e-6


RUNGS = np.array([128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
                  4096, 5120, NC])


def rung_of(B):
    return int(RUNGS[np.searchsorted(RUNGS, max(B, 1))])


print(f"  per-lane cost: " + "  ".join(f"C={int(c)}: {perlane_us(c):.1f}us"
                                       for c in [128, 256, 512, 1024, 2048, 4096, NC]))
# The baseline is MODELED with the same zero-slop convention as the schemes
# (accepted steps x the full-grid call cost) so the ratios compare like with
# like; the measured Phase-1 wall is printed alongside when this is the
# measured segment (the difference is lockstep's own rejects, ~7%).
LOCK_WALL = naC[seg].sum() * callcost_s(NC)
print(f"  modeled lockstep wall on segment: {LOCK_WALL:.1f} s "
      f"({naC[seg].sum():.0f} accepted steps x {1e3 * callcost_s(NC):.1f} ms)"
      + (f"; measured (slurm 10155020, rejects included): {LOCK_WALL_MEAS:.1f} s"
         if abs(t0 - 41400.0) < 1 and abs(t1 - 53400.0) < 1 else ""))


def wall_equalK(s_true, o, K):
    srt = s_true[o]
    return sum(srt[b].max() * callcost_s(rung_of(len(b)))
               for b in np.array_split(np.arange(NC), K))


def wall_lazy(s_true, o, s_pred, q, nbands, minband=96):
    """One lazy bucket (pred demand <= quantile q) + log-width stiff bands.
    Bands smaller than `minband` merge into their stiffer neighbor rather
    than pad a floor rung. Returns (seconds, bucket sizes)."""
    sp = s_pred[o]; st = s_true[o]
    edge = np.quantile(sp, q)
    lazy = np.where(sp <= edge)[0]
    tail = np.where(sp > edge)[0]
    buckets = [lazy] if lazy.size else []
    if tail.size:
        lo, hi = max(edge, 1.0), max(sp[tail].max(), edge * (1 + 1e-9))
        edges = np.geomspace(lo, hi, nbands + 1)
        bands = [tail[(sp[tail] > edges[i]) & (sp[tail] <= edges[i + 1])]
                 for i in range(nbands)]
        bands[0] = np.concatenate([tail[sp[tail] <= edges[0]], bands[0]])
        merged = []
        for b in bands:                       # merge small bands UPWARD
            if merged and (len(merged[-1]) < minband or len(b) < minband):
                merged[-1] = np.concatenate([merged[-1], b])
            elif b.size:
                merged.append(b)
        buckets += merged
    w = sum(st[b].max() * callcost_s(rung_of(len(b))) for b in buckets if b.size)
    return w, [len(b) for b in buckets]


def _dp_cuts(svals, stride):
    """Cuts minimizing the cost model against `svals` (contiguous groups of
    the given order), breakpoints every `stride` cells. Returns cut list.
    Vectorized: per endpoint j, the candidate costs over all i are a reverse
    cummax of the per-chunk maxes times a size-indexed call-cost table."""
    cuts = np.arange(0, NC, stride)
    cuts = np.append(cuts, NC)
    n = len(cuts) - 1
    gm = np.array([svals[cuts[i]:cuts[i + 1]].max() for i in range(n)])
    # call cost for a group of k chunks (size cuts[i+k]-cuts[i]; the last
    # chunk may be short -- costs are cheap to overestimate by one stride)
    cc = np.array([callcost_s(rung_of(min(k * stride, NC))) for k in range(1, n + 1)])
    best = np.full(n + 1, np.inf); best[0] = 0.0
    arg = np.zeros(n + 1, dtype=int)
    for j in range(1, n + 1):
        gmax = np.maximum.accumulate(gm[j - 1::-1])[::-1]     # max(gm[i:j]) for i=0..j-1
        cand = best[:j] + gmax * cc[j - 1 - np.arange(j)]     # size (j-i) chunks
        i = int(np.argmin(cand))
        best[j] = cand[i]; arg[j] = i
    out = []; j = n
    while j > 0:
        out.append((int(cuts[arg[j]]), int(cuts[j]))); j = arg[j]
    return out[::-1]


def wall_dp(s_true, o, stride=16):
    """CLAIRVOYANT bound: cuts chosen against the TRUE demand (only the
    ordering comes from the policy). Not realizable; a model ceiling."""
    st = s_true[o]
    return sum(st[a:b].max() * callcost_s(rung_of(b - a))
               for a, b in _dp_cuts(st, stride))


def wall_dp_real(s_true, s_pred, o, stride=16):
    """REALIZABLE: order and cuts both from the PREDICTED demand; the price
    is what the TRUE demand then costs on that partition."""
    st = s_true[o]; sp = s_pred[o]
    return sum(st[a:b].max() * callcost_s(rung_of(b - a))
               for a, b in _dp_cuts(sp, stride))


for label, fn in (
        ("equal-K16 (phase-1)", lambda m: wall_equalK(S[m], order("trend", m), 16)),
        ("equal-K64 (phase-1)", lambda m: wall_equalK(S[m], order("trend", m), 64)),
        ("lazy q=.75 nb=4", lambda m: wall_lazy(S[m], order("trend", m),
                                                S[m - 1] ** 2 / S[m - 2] if m > 1 else S[max(m - 1, 0)],
                                                0.75, 4)[0]),
        ("lazy q=.85 nb=5", lambda m: wall_lazy(S[m], order("trend", m),
                                                S[m - 1] ** 2 / S[m - 2] if m > 1 else S[max(m - 1, 0)],
                                                0.85, 5)[0]),
        ("lazy q=.60 nb=6", lambda m: wall_lazy(S[m], order("trend", m),
                                                S[m - 1] ** 2 / S[m - 2] if m > 1 else S[max(m - 1, 0)],
                                                0.60, 6)[0]),
        ("DP-clairvoyant(trend)", lambda m: wall_dp(S[m], order("trend", m))),
        ("DP-clairvoyant(oracle)", lambda m: wall_dp(S[m], order("oracle", m))),
        ("DP-REALIZABLE (trend)", lambda m: wall_dp_real(
            S[m], S[m - 1] ** 2 / S[m - 2] if m > 1 else S[max(m - 1, 0)],
            order("trend", m))),
        ("DP-REALIZABLE (prev)", lambda m: wall_dp_real(
            S[m], S[max(m - 1, 0)], order("prev", m)))):
    w = sum(fn(m) for m in seg)
    r = LOCK_WALL / w
    print(f"  {label:22s} predicted wall {w:7.1f} s  ceiling {r:5.2f}x"
          f"   with slop 1.25x: {r / 1.25:5.2f}x   1.5x: {r / 1.5:5.2f}x   1.8x: {r / 1.8:5.2f}x")

# a typical window's lazy partition shape, for the implementation to mirror
m0 = seg[len(seg) // 2]
_, sizes = wall_lazy(S[m0], order("trend", m0),
                     S[m0 - 1] ** 2 / S[m0 - 2], 0.75, 4)
print(f"  lazy q=.75 nb=4 sizes at t={tcur[m0]:.0f}: {sizes} "
      f"(rungs {[rung_of(s) for s in sizes]})")
