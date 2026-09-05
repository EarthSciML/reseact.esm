#!/bin/bash
# ===========================================================================
# p5_bufgrad_pair.sbatch -- the 917b754 recipe for a driver-default change:
# the SAME 6x6x8 fwd,adj configuration twice, RESEACT_ADJ_BUFGRAD=1 (the old
# VJP, forcing buffers active) then =0 (p only, the new default), and the two
# gradient CSVs must agree to every printed digit.
#   sbatch tools/diag/p5_bufgrad_pair.sbatch
# ===========================================================================
#SBATCH --job-name=p6pair
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl-stencil}
export RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8 RESEACT_ADJ_NMACRO=3
export RESEACT_ADJ_CLAMP=1 RESEACT_ADJ_UJITTER=0 RESEACT_ADJ_STAGES=fwd,adj RESEACT_ADJ_JAC=sym
for arm in 0 1; do
  echo "=== arm ESS_OOP_SHIFT_SLICE=$arm  $(date -Is) ==="
  ESS_OOP_SHIFT_SLICE=$arm RESEACT_LABEL=p6pair-slice$arm RESEACT_ADJ_CSV=logs/p6pair-${SLURM_JOB_ID}-slice$arm.csv \
    julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=32G tools/adjoint_gradient.jl \
    | grep -E "BUILD|@compile|forward pass|backward sweep|of which|J = |structural|replay lands|retries|RESULT" | cut -c1-200
done
echo "=== diff of the two gradient CSVs (empty = identical to every printed digit) ==="
diff <(cut -d, -f1- logs/p6pair-${SLURM_JOB_ID}-slice0.csv | sed 's/p6pair-slice0//') \
     <(cut -d, -f1- logs/p6pair-${SLURM_JOB_ID}-slice1.csv | sed 's/p6pair-slice1//') && echo "P6PAIR_IDENTICAL" || echo "P6PAIR_DIFFERS"
echo "finished $(date -Is)"
