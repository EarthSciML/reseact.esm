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
# WHAT TO EXPECT (pre-run sizing, to be replaced by measurements):
#   columns   4x5 91 -> 2x2.5 325, i.e. 3.57x the cells;
#   transport substeps ~2x more at half the cell width (CFL), so the transport
#   half should cost ~7x and the chemistry half ~3.6x;
#   build and compile scale with the grid too (235 s build at 4x5 CONUS).
# A 48 h window took 2,799 s of loop at 4x5 on 2026-09-05 (slurm 10372969), so
# the 2x2.5 48 h arm is expected somewhere near 4-6x that, and the five-day arm
# 2.5x its own 48 h.
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
