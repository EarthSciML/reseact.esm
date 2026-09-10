#!/bin/bash
# ===========================================================================
# p13_census.sbatch -- oop_ssa_stats coverage + the per-blocker attribution for
# each ess-oop-ssa redirect ARM, on both ReSEACT split parts.
# ===========================================================================
# tools/diag/p12_ssa_blockers.jl builds ONE split part and prints
# `oop_ssa_stats` (edges / sub-kernel edges / ghost edges redirected, producer
# scatters skipped, and which read surface blocks each surviving one) plus a
# per-descriptor-kind census. One build per arm, so it is a job and not an
# interactive run: the session's 40 GiB cgroup is shared and these OOM-kill each
# other when run alongside anything else.
#
#   mkdir -p logs && sbatch --job-name=p13cen tools/diag/p13_census.sbatch
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
CODE=/projects/illinois/eng/cee/ctessum/ctessum/code
export RESEACT_REPO=${RESEACT_REPO:-$CODE/reseact.esm}
cd "$RESEACT_REPO"
export RESEACT_MODEL=${RESEACT_MODEL:-$CODE/reseact.esm/reseact.esm}
export EARTHSCIMLBASE_SRC=${EARTHSCIMLBASE_SRC:-$CODE/EarthSciMLBase/src}
export RESEACT_RXENV=${RESEACT_RXENV:-$CODE/.ssaenv/run-model-jl}
export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
export ESS_OOP_SSA=1
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} ==="
for part in 1 2; do
  for arms in 000 111 100 010 001; do
    g=${arms:0:1}; s=${arms:1:1}; p=${arms:2:1}
    echo "##### part=$part ghost=$g sub=$s pgather=$p  $(date -Is)"
    ESS_OOP_SSA_GHOST=$g ESS_OOP_SSA_SUB=$s ESS_OOP_SSA_PGATHER=$p P10_PART=$part \
      julia --project="$RESEACT_RXENV" tools/diag/p12_ssa_blockers.jl 2>&1 |
      grep -E "^(BUILD|P10_|P12_)"
  done
done
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
