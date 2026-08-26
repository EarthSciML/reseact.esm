#!/bin/bash
# ===========================================================================
# obs_inline_ab.sbatch -- A/B the ARRAY-OBSERVED MATERIALISATION on one ROS23 step
# ===========================================================================
# The CONUS HLO dump (logs/hlodump-10105946) shows the ros23 step spends 45% of
# its element-ops moving a f64[1310641] "extended state" buffer around:
#   47 whole-buffer concatenate + 35 scatter + 21 slice+concat + 4 pad at ENTRY
#   level = 1868 MB WRITTEN per step, against a real state of 85176 doubles.
# That buffer is the materialised array-observed block built by
# `_oop_prefix_copy`/`_oop_fill_levels` (EarthSciAST src/tree_walk/oop.jl:2797).
#
# `ESS_ARRAY_OBS_INLINE=1` makes `_collect_materialized_array_obs` return an
# empty set (EarthSciAST src/tree_walk/build.jl:371,378), so `mat_vars` is empty,
# `ue === u`, and the ENTIRE prelude folds away -- the observeds are inlined into
# their readers instead. Same answers, no buffer.
#
# PREDICTION if the buffer traffic is the cost: the inline arm's step time drops
# well beyond the ~12% run-to-run drift seen in this session.
# FALSIFIED if the two arms are within ~15% of each other.
#
#   sbatch --export=ALL,AB_INLINE=0,AB_GRID=13x7x24 tools/diag/obs_inline_ab.sbatch
#   sbatch --export=ALL,AB_INLINE=1,AB_GRID=13x7x24 tools/diag/obs_inline_ab.sbatch
# ===========================================================================
#SBATCH --job-name=c2dab
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

AB_INLINE=${AB_INLINE:-0}
AB_GRID=${AB_GRID:-13x7x24}
IFS=x read -r NLON NLAT NLEV <<< "$AB_GRID"

export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export RESEACT_BACKEND=cpu
export RESEACT_NLON=$NLON RESEACT_NLAT=$NLAT RESEACT_NLEV=$NLEV
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_UJITTER=0
export RESEACT_PROF_REPS=${RESEACT_PROF_REPS:-20}
export RESEACT_PROF_VJP=0
export ESS_ARRAY_OBS_INLINE=0
export AB_C2D=${AB_C2D:-0}

echo "=== node $(hostname) grid=$AB_GRID cpus=$SLURM_CPUS_PER_TASK AB_C2D=$AB_C2D ==="
echo "started $(date -Is)"

julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/c2d_driver.jl

echo "finished $(date -Is)"
