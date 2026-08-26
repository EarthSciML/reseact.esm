#!/bin/bash
# ===========================================================================
# c2d_conus.sbatch -- CONUS A/B of Reactant's CONCATS_TO_DUS on one ROS23 step.
# ===========================================================================
# At NC=2184 the flag was worth 1.12x (45.765 -> 40.878 ms). The buffer-traffic
# mechanism predicts the ratio must GROW with grid, because the extended
# observed buffer is 3.5 MB at NC=2184 (L3-resident) and 10.5 MB at CONUS (not).
# If CONUS also comes back ~1.12x, the mechanism claim fails.
#
# BOTH arms are run fresh: the historical 253.40 ms CONUS control predates the
# count-locate change (measured 1.27x), so it is NOT a valid control any more.
#
#   sbatch --export=ALL,AB_C2D=0 tools/diag/c2d_conus.sbatch
#   sbatch --export=ALL,AB_C2D=1 tools/diag/c2d_conus.sbatch
# ===========================================================================
#SBATCH --job-name=c2dstruct
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

AB_C2D=${AB_C2D:-0}
AB_GRID=${AB_GRID:-13x7x24}
IFS=x read -r NLON NLAT NLEV <<< "$AB_GRID"

export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export RESEACT_BACKEND=cpu
export RESEACT_NLON=$NLON RESEACT_NLAT=$NLAT RESEACT_NLEV=$NLEV
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_UJITTER=0
export RESEACT_PROF_REPS=${RESEACT_PROF_REPS:-40}
export RESEACT_PROF_VJP=0
export ESS_ARRAY_OBS_INLINE=0
export AB_C2D=$AB_C2D

echo "=== node $(hostname) grid=$AB_GRID cpus=$SLURM_CPUS_PER_TASK AB_C2D=$AB_C2D ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/c2d_structural.jl
echo "finished $(date -Is)"
