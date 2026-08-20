#!/usr/bin/env python3
"""Compare the traced arm's 24 h CONUS output against the native arm's.

WHY THIS EXISTS. `run_reseact_reactant.jl`'s header records agreement with the
native arm over 24 h (289 aligned records) at max relative difference 6.3e-5 on
O3_mean, 1.6e-3 on O3_min, 1.1e-3 on OH_max. That was measured with the XLA:CPU
race ACTIVE.

The race is now known to shift the answer by integrating a DIFFERENT trajectory
(it moved the adjoint objective 4.0e-5 relative), and that biasing -- not speed
-- is the whole justification for defaulting RESEACT_RXFIX on, since it COSTS
~6% of wall clock on a real trajectory (slurm 10017939). This script tests that
justification: with the race fixed, agreement should IMPROVE, plausibly by about
the size of the bias. If it does not move, the correctness case for the default
is weak on this runner.

BOTH CSVs MUST COME FROM THE SAME EarthSciAST. The traced CSV was produced on
pre-rename EarthSciAST (before PR #167's examples -> analyses landed at 21:16:07
on 2026-08-19); the native reference must be pinned to match, or a schema change
confounds the race question. See tools/diag/native_24h_reference.sbatch.

USAGE:
    python3 tools/diag/compare_traced_native.py [native.csv] [traced.csv]
defaulting to logs/native24.csv and logs/rx24-racefix.csv.
"""
import csv
import sys

# The native arm is the reference; relative differences are taken against it.
COLUMNS = ("o3_mean", "o3_min", "o3_max", "oh_max", "no2_mean", "m_min")

# What the documented run got, with the race ACTIVE. Anything materially below
# these is an improvement attributable to the fix; anything at or above them
# means the race was not what limited agreement.
DOCUMENTED = {"o3_mean": 6.3e-5, "o3_min": 1.6e-3, "oh_max": 1.1e-3}


def load(path):
    """Index rows by simulated hour, which is the join key both runners share."""
    with open(path) as fh:
        return {round(float(r["hours"]), 4): r for r in csv.DictReader(fh)}


def main():
    native_path = sys.argv[1] if len(sys.argv) > 1 else "logs/native24.csv"
    traced_path = sys.argv[2] if len(sys.argv) > 2 else "logs/rx24-racefix.csv"

    native, traced = load(native_path), load(traced_path)
    keys = sorted(set(native) & set(traced))
    if not keys:
        sys.exit(f"no aligned records between {native_path} and {traced_path}")

    print(f"native  {native_path}  ({len(native)} records)")
    print(f"traced  {traced_path}  ({len(traced)} records)")
    print(f"aligned {len(keys)} records, h={keys[0]:g}..{keys[-1]:g}\n")
    print(f"{'column':10s} {'max rel diff':>12s}  {'at hour':>8s}  "
          f"{'documented':>11s}  verdict")

    for col in COLUMNS:
        # Seed the hour, not None: when the two runs agree EXACTLY (rel == 0.0
        # everywhere, as in a self-comparison) `rel > worst` never fires and a
        # None hour would crash the formatter on the one input we most want to
        # work -- the identity check that proves this script is measuring at all.
        worst, worst_h = 0.0, keys[0]
        for k in keys:
            a, b = float(native[k][col]), float(traced[k][col])
            # Guard the zero denominator: several species start at exactly 0.0,
            # where a relative difference is undefined rather than infinite.
            denom = max(abs(a), 1e-30)
            rel = abs(b - a) / denom
            if rel > worst:
                worst, worst_h = rel, k

        doc = DOCUMENTED.get(col)
        if doc is None:
            verdict, doc_s = "", "--"
        elif worst < doc / 2:
            verdict, doc_s = "IMPROVED", f"{doc:.1e}"
        elif worst <= doc * 1.5:
            verdict, doc_s = "unchanged", f"{doc:.1e}"
        else:
            verdict, doc_s = "WORSE", f"{doc:.1e}"

        print(f"{col:10s} {worst:12.3e}  {worst_h:8g}  {doc_s:>11s}  {verdict}")

    print("\nIMPROVED on o3_mean supports keeping RESEACT_RXFIX on by default;")
    print("'unchanged' means the ~6% it costs buys no accuracy on this runner.")


if __name__ == "__main__":
    main()
