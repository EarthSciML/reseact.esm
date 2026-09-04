#!/bin/bash
# ===========================================================================
# p5_bufgrad_pair.sbatch -- the 917b754 recipe for a driver-default change:
# the SAME 6x6x8 fwd,adj configuration twice, RESEACT_ADJ_BUFGRAD=1 (the old
# VJP, forcing buffers active) then =0 (p only, the new default), and the two
# gradient CSVs must agree to every printed digit.
#   sbatch tools/diag/p5_bufgrad_pair.sbatch
# ===========================================================================
#SBATCH --job-name=p5pair
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
export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8 RESEACT_ADJ_NMACRO=3
export RESEACT_ADJ_CLAMP=1 RESEACT_ADJ_UJITTER=0 RESEACT_ADJ_STAGES=fwd,adj RESEACT_ADJ_JAC=sym
for arm in 1 0; do
  echo "=== arm RESEACT_ADJ_BUFGRAD=$arm  $(date -Is) ==="
  RESEACT_ADJ_BUFGRAD=$arm RESEACT_LABEL=p5pair-bufgrad$arm RESEACT_ADJ_CSV=logs/p5pair-${SLURM_JOB_ID}-bufgrad$arm.csv \
    julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=32G tools/adjoint_gradient.jl \
    | grep -E "BUILD|@compile|forward pass|backward sweep|of which|J = |structural|replay lands|retries|RESULT" | cut -c1-200
done
echo "=== diff of the two gradient CSVs (empty = identical to every printed digit) ==="
diff <(cut -d, -f1- logs/p5pair-${SLURM_JOB_ID}-bufgrad1.csv | sed 's/p5pair-bufgrad1//') \
     <(cut -d, -f1- logs/p5pair-${SLURM_JOB_ID}-bufgrad0.csv | sed 's/p5pair-bufgrad0//') && echo "P5PAIR_IDENTICAL" || echo "P5PAIR_DIFFERS"
echo "finished $(date -Is)"
