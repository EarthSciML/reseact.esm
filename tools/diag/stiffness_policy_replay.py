#!/usr/bin/env python3
# ===========================================================================
# stiffness_policy_replay.py -- price bucketing policies against the recorded
# per-cell step-demand matrix from stiffness_diurnal.jl.
#
# The question this answers: how much of the stiffest-cell-dictatorship waste
# does a REALIZABLE re-sorting policy capture, given that which cells are
# stiff changes as the sun moves? Policies use ONLY realized controller state
# (previous windows' per-cell demand) -- no model variables, by design.
#
#   global   : what one global controller costs (both the actual accepted
#              count and the max-cell estimate, as a consistency check)
#   spatial  : K equal contiguous shards in state layout order, never re-sorted
#   prev     : re-sort every window by the previous window's demand
#   prev2max : re-sort by max of the previous two windows (1-window headroom
#              against regime changes, e.g. the terminator band)
#   oracle   : re-sort by the current window's own demand (upper bound)
#
# Bucket cost model: a bucket of B cells stepping together pays
# B * max(demand of its members); a policy's window cost is the sum over its
# K buckets; "ideal" is sum(demand) (a per-cell controller).
#
# Usage: stiffness_policy_replay.py [outdir]   (default logs/stiffdiurnal)
# ===========================================================================
import sys, os
import numpy as np

outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "stiffdiurnal")
meta = dict(l.strip().split("=", 1) for l in open(os.path.join(outdir, "meta.txt"))
            if "=" in l and not l.startswith("record"))
NC = int(meta["NC"])
rec = 6 + NC
raw = np.fromfile(os.path.join(outdir, "percell_steps.bin"), dtype=np.float64)
M = raw.size // rec
raw = raw[: M * rec].reshape(M, rec)
hdr, S = raw[:, :6], raw[:, 6:]
tcur, tnext, naT, nrT, naC, nrC = hdr.T
S = np.maximum(S, 1.0)          # a cell never takes less than one step
print(f"loaded {M} windows x {NC} cells from {outdir}")
print(f"  global controller accepted steps: total {naC.sum():.0f}, "
      f"per-window med {np.median(naC):.0f} max {naC.max():.0f}")
smax = S.max(axis=1)
print(f"  consistency: max-cell demand vs accepted  med ratio "
      f"{np.median(smax / naC):.2f}  (should be ~1)\n")

def bucket_cost(s_true, order, K):
    """Cost of stepping K equal-size buckets (formed by `order`) through one
    window, each at its own worst member's demand under the TRUE demand."""
    srt = s_true[order]
    cost = 0.0
    for b in np.array_split(np.arange(NC), K):
        cost += len(b) * srt[b].max()
    return cost

Ks = [1, 2, 4, 8, 16, 32, 64, 128]
ident = np.arange(NC)
policies = {"spatial": lambda m: ident,
            "prev":    lambda m: np.argsort(S[m - 1]) if m > 0 else ident,
            "prev2max": lambda m: np.argsort(np.maximum(S[m - 1], S[m - 2]))
                                  if m > 1 else (np.argsort(S[m - 1]) if m > 0 else ident),
            # trend: extrapolate each cell's demand geometrically from its
            # last two windows -- rising cells (sunrise band) sort stiffer
            "trend":   lambda m: np.argsort(S[m - 1] ** 2 / S[m - 2])
                                  if m > 1 else (np.argsort(S[m - 1]) if m > 0 else ident),
            "oracle":  lambda m: np.argsort(S[m])}

glob_actual = float((naC + 0.0).sum() * NC)   # what was actually paid
glob_est = float(sum(NC * smax[m] for m in range(M)))
ideal = float(S.sum())

lines = []
lines.append(f"{'policy':10s}" + "".join(f"{('K=' + str(K)):>12s}" for K in Ks))
totals = {}
for name, fn in policies.items():
    row = []
    for K in Ks:
        tot = sum(bucket_cost(S[m], fn(m), K) for m in range(M))
        row.append(tot)
    totals[name] = row
    lines.append(f"{name:10s}" + "".join(f"{glob_actual / t:11.2f}x" for t in row))
lines.append("")
lines.append(f"speedups are cell-step reductions vs the ACTUAL global controller "
             f"({glob_actual:.3g} cell-steps)")
lines.append(f"  max-cell global estimate {glob_est:.3g} ({glob_actual / glob_est:.2f}x of actual; "
             f"the gap is controller slop: rejects + PI ramping)")
lines.append(f"  per-cell ideal (infinite buckets, perfect prediction): "
             f"{glob_actual / ideal:.2f}x")
lines.append("")

# how much does misprediction cost, and when?
K = 16
oracle_w = np.array([bucket_cost(S[m], policies['oracle'](m), K) for m in range(M)])
prev_w = np.array([bucket_cost(S[m], policies['prev'](m), K) for m in range(M)])
pen = prev_w / oracle_w
worst = np.argsort(pen)[::-1][:8]
lines.append(f"K={K}: prev-vs-oracle penalty  med {np.median(pen):.3f}  "
             f"p90 {np.quantile(pen, .9):.3f}  max {pen.max():.3f}")
lines.append("  worst windows (regime changes -- expect terminator crossings):")
for m in worst:
    lines.append(f"    w{m:03d} t=[{tcur[m]:.0f},{tnext[m]:.0f}] penalty {pen[m]:.3f} "
                 f"(chem {naC[m]:.0f} acc)")
# window-to-window predictability of the demand field itself
rho = [np.corrcoef(np.log(S[m - 1]), np.log(S[m]))[0, 1] for m in range(1, M)]
lines.append(f"\nlog-demand autocorrelation window-to-window: "
             f"med {np.median(rho):.3f}  min {np.min(rho):.3f} "
             f"(at w{int(np.argmin(rho)) + 1:03d})")

report = "\n".join(lines)
print(report)
with open(os.path.join(outdir, "replay_report.txt"), "w") as f:
    f.write(report + "\n")
print(f"\nwritten {os.path.join(outdir, 'replay_report.txt')}")
