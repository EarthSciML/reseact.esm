#!/bin/bash
#SBATCH --job-name=bucketverify
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact-capsub/logs/bucket/%x-%j.out
# The 6x6x8 merge gate for tools/bucket_chem.jl -- see bucket_verify.jl.
# NOTE ON --export: `sbatch --export=ALL,K=V` splits on COMMAS and mangles any
# ladder spec; export in the submitting shell and use the default --export=ALL.
set -euo pipefail
B=/projects/illinois/eng/cee/ctessum/ctessum/code/.p34sbatch
export JULIA_DEPOT_PATH="$B/depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia"
export RESEACT_RXENV="$B/env5"
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
export RESEACT_BACKEND=cpu RESEACT_ADJ_JAC=sym RESEACT_ADJ_UJITTER=0 RESEACT_ADJ_CLAMP=1
export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_BUCKET=${RESEACT_BUCKET:-4}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-6}
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact-capsub
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} K=${RESEACT_BUCKET} windows=${RESEACT_ADJ_NMACRO} ==="
echo "started $(date -Is)"
NT=${SLURM_CPUS_PER_TASK:-8}
HH=$(( ${SLURM_MEM_PER_NODE:-49152} / 1024 * 8 / 10 ))G
# tee, never a bare grep: a filtered pipe eats stack traces.
julia --project="$RESEACT_RXENV" -t "$NT" --heap-size-hint="$HH" \
    tools/diag/bucket_verify.jl 2>&1 | tee "logs/bucket/bucketverify-${SLURM_JOB_ID}.full.log"
rc=${PIPESTATUS[0]}
echo "finished $(date -Is) rc=$rc"
exit "$rc"
