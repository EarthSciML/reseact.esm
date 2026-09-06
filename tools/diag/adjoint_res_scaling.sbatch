#!/bin/bash
# ===========================================================================
# adjoint_res_scaling.sbatch -- the SAME CONUS gradient at two GEOS-FP
# resolutions, so that "how does the adjoint scale with resolution" is a
# measurement rather than an estimate.
# ===========================================================================
# WHAT IT VARIES, and what it deliberately does not. Exactly two knobs move:
# RESEACT_RES (the GEOS-FP grid row) and RESEACT_ADJ_NMACRO (the window). The
# resolution row carries the CONUS index box with it -- 13x7 columns at 4x5,
# 25x13 at 2x2.5 -- so both arms simulate the SAME geography (lon -125..-65,
# lat 26..50) over the same days, with the same 72 hybrid levels, the same
# objective, the same 8 chemistry shards, the same solver tolerances and the
# same node shape. The only physical difference is cell size.
#
# MEASURED 2026-09-06 (slurm 10386109 / 10386110 / 10386111 / 10386112, one
# whole node each, 40 cpus / 180 GB, 8 chemistry shards, jac=:sym, clamp on):
#
#   arm            fwd s/win   bwd s/win     loop      all in   J (ppb)
#   4x5   48 h        1.160       3.657    46 m 15 s   1 h 11 m  30.1943301698523
#   4x5    5 d        1.147       3.647   1 h 55 m     2 h 21 m  27.0756235696808
#   2x2.5 48 h        4.088      12.419   2 h 38 m     3 h 21 m  30.5139290179448
#   2x2.5  5 d        4.185      12.509   6 h 41 m     7 h 25 m  27.8190596524603
#
# ALL FOUR: replay 0.000e+00 at every checkpoint, 0 flaky-reverse retries
# (70,401 and 71,472 VJP calls in the five-day arms). Driver MaxRSS 42.9 /
# 45.5 / 78.6 / 83.4 GB in the order above.
#
# THE HEADLINE: cost is LINEAR IN CELLS, and slightly sublinear at that. 2x2.5
# is 3.57x the columns (91 -> 325) and costs 3.52x the forward, 3.40x the
# backward, 3.43x the 48 h loop and 3.48x the five-day loop. Both windows scale
# the same way, and the five-day arm is 2.53x its own 48 h -- i.e. the window
# is linear too, with no drift or degradation over 120 h. The pre-run sizing above guessed ~7x for transport
# on the theory that half the cell width would force ~2x the substeps; it does
# not. The adaptive controller took 28,384 inner steps at 2x2.5 against 27,973
# at 4x5 -- +1.5%, of which transport is 1,977 vs 1,859 -- because at 300 s
# macro steps and ~220 km cells the substep count is set by ACCURACY, not by
# CFL. Everything else follows the cell count:
#
#   per window, 48 h      4x5     2x2.5    ratio
#   C.vjp                1.42 s   4.55 s    3.2
#   T.vjp                1.07 s   3.68 s    3.4
#   C.replay             0.61 s   2.07 s    3.4
#   T.replay             0.06 s   0.31 s    5.2
#   refresh              0.42 s   1.50 s    3.6   (forcing decode; see
#                                                  INDEX_RANGE_READS.md)
#   driver MaxRSS       42.9 GB  78.6 GB    1.8
#
# SETUP scales SUBLINEARLY and is a smaller share at 2x2.5: build 226 -> 330 s,
# prepare_jacobian 64 -> 142 s, ssp_step compile 129 -> 228 s, ssp_vjp compile
# 315 -> 933 s, shards ready 577 -> 711 s.
#
# The five-day 4x5 arm is the FIRST end-to-end 120 h gradient: 1,440 macro
# steps, 70,401 accepted inner steps, replay 0.0 at every checkpoint, 0 retries.
#
# DOES THE GRADIENT ITSELF SURVIVE THE REFINEMENT? Mostly, and that is the
# scientific point of running two grids. Over the same 48 h, the dominant
# physical components move a few percent between 4x5 and 2x2.5:
#
#   NEIRegrid.scale        -2.1144 -> -2.2073   +4.4%
#   DryDepositionGas.kappa -1.6165 -> -1.7114   +5.9%
#   DryDepositionGas.g_const -0.5129 -> -0.4896  -4.5%
#   NEIRegrid.g0           -0.2156 -> -0.2251   +4.4%
#   Transport3D.tau_pblmix -4.13e-4 -> -4.35e-4 +5.3%
#
# and over five days they agree BETTER, not worse -- NEIRegrid.scale -2.4%,
# kappa +4.2%, g_const -5.4%, g0 -2.4% -- so the emissions and dry-deposition
# sensitivities an inversion would actually use are converged to within ~6% at
# 4x5, and refining the grid is not where the next digit comes from. The outliers are the two GEOMETRY
# derivatives (dlon_deg 2.8x, dlat_deg 2.2x), which is what they should do --
# they measure the discretisation, and the discretisation is exactly what
# changed. J itself moves 30.194 -> 30.514 ppb (+1.1%) over 48 h and
# 27.076 -> 27.819 (+2.7%) over five days.
#
# TWO NEW NONZERO GRADIENT COMPONENTS, and they are a side effect of this work
# rather than new physics: dlon_deg/dlat_deg were literal equations and are now
# parameters, so the sweep returns dJ/d(cell size) (-1.33 ppb/deg for dlat over
# five days). Read it as the derivative of the DISCRETISATION -- bigger cells,
# same meteorology -- not of anything in the atmosphere.
#
#   mkdir -p logs
#   sbatch --job-name=adjres-4x5-48h  --time=06:00:00 \
#          --export=ALL,RESEACT_RES=4x5,RESEACT_ADJ_NMACRO=576 \
#          tools/diag/adjoint_res_scaling.sbatch
#   sbatch --job-name=adjres-4x5-5d   --time=10:00:00 \
#          --export=ALL,RESEACT_RES=4x5,RESEACT_ADJ_NMACRO=1440 \
#          tools/diag/adjoint_res_scaling.sbatch
#   sbatch --job-name=adjres-2x25-48h --time=24:00:00 \
#          --export=ALL,RESEACT_RES=2x2.5,RESEACT_ADJ_NMACRO=576 \
#          tools/diag/adjoint_res_scaling.sbatch
#   sbatch --job-name=adjres-2x25-5d  --time=48:00:00 \
#          --export=ALL,RESEACT_RES=2x2.5,RESEACT_ADJ_NMACRO=1440 \
#          tools/diag/adjoint_res_scaling.sbatch
#
# The runner (run_reseact_adjoint.jl) sets every other default with `get!`, so
# anything NOT exported here comes from its preset block -- clamp on, jac=:sym,
# the XLA:CPU race workaround, 8 shards, stages fwd,adj -- and is identical
# across the arms by construction rather than by being copied here.
# ===========================================================================
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
# 8 shard workers plus the driver. At 4x5 the workers peaked ~4 GB each against
# a 160G request; 2x2.5 is 3.6x the cells, so ask for the node (192 GB physical,
# ~188 usable) and let the smaller arm over-request rather than run the two arms
# at different memory pressure.
#SBATCH --mem=180G
# Overridden per submission (see the commands above); this is only the fallback.
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_RES=${RESEACT_RES:-4x5}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-576}
# The label and the CSV carry the arm, so four runs cannot overwrite each other
# and a CSV can be traced to its grid without opening the log.
ARM="${RESEACT_RES//./}-${RESEACT_ADJ_NMACRO}"
export RESEACT_LABEL=${RESEACT_LABEL:-conus-adjres-${ARM}-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-logs/adjres-${ARM}-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== res=$RESEACT_RES nmacro=$RESEACT_ADJ_NMACRO label=$RESEACT_LABEL ==="
free -g | head -2
echo "started $(date -Is)"

julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G run_reseact_adjoint.jl

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
