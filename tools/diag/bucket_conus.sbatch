#!/bin/bash
#SBATCH --job-name=bucketconus
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact-capsub/logs/bucket/%x-%j.out
# The CONUS speed measurement for the bucketed chemistry scheduler: lockstep
# vs K=16 vs K=64 over one diurnal segment containing the sunrise sweep
# t=42300-52200 (see tools/diag/bucket_conus.jl). Forward only.
# NOTE ON --export: `sbatch --export=ALL,K=V` splits on COMMAS; export in the
# submitting shell and use the default --export=ALL.
set -euo pipefail
B=/projects/illinois/eng/cee/ctessum/ctessum/code/.p34sbatch
export JULIA_DEPOT_PATH="$B/depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia"
export RESEACT_RXENV="$B/env5"
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
export RESEACT_BACKEND=cpu RESEACT_ADJ_JAC=sym RESEACT_ADJ_UJITTER=0 RESEACT_ADJ_CLAMP=1
export RESEACT_NLON=${RESEACT_NLON:-13} RESEACT_NLAT=${RESEACT_NLAT:-7} RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_T0=${RESEACT_T0:-41400}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-40}
export RESEACT_BUCKET=${RESEACT_BUCKET:-16}
export RESEACT_BUCKET_KS=${RESEACT_BUCKET_KS:-16,64}
export RESEACT_ADJ_STAGES=none
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact-capsub
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} T0=${RESEACT_T0} windows=${RESEACT_ADJ_NMACRO} Ks=${RESEACT_BUCKET_KS} ==="
echo "started $(date -Is)"
NT=${SLURM_CPUS_PER_TASK:-8}
HH=$(( ${SLURM_MEM_PER_NODE:-102400} / 1024 * 6 / 10 ))G
# tee, never a bare grep: a filtered pipe eats stack traces.
julia --project="$RESEACT_RXENV" -t "$NT" --heap-size-hint="$HH" \
    tools/diag/bucket_conus.jl 2>&1 | tee "logs/bucket/bucketconus-${SLURM_JOB_ID}.full.log"
rc=${PIPESTATUS[0]}
echo "finished $(date -Is) rc=$rc"
exit "$rc"
