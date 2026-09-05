#!/bin/bash
# ===========================================================================
# p5.sbatch -- run one of the p5_* VJP probes at CONUS (or any grid) on the
#              ctessum partition. Never run these at CONUS in a session: the
#              interactive cgroup is 40 GiB and the VJP compile alone exceeds it.
#
#   sbatch --export=ALL,P5_SCRIPT=p5_vjp_census.jl,RESEACT_NLON=13,RESEACT_NLAT=7,RESEACT_NLEV=72 tools/diag/p5.sbatch
#   sbatch --export=ALL,P5_SCRIPT=p5_vjp_time.jl,P5_EXTRA=<names>,RESEACT_NLON=13,... tools/diag/p5.sbatch
# ===========================================================================
#SBATCH --job-name=p5vjp
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl}
P5_SCRIPT=${P5_SCRIPT:-p5_vjp_census.jl}
echo "=== node $(hostname) script=$P5_SCRIPT grid=${RESEACT_NLON:-6}x${RESEACT_NLAT:-6}x${RESEACT_NLEV:-8} cpus=$SLURM_CPUS_PER_TASK ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G "tools/diag/$P5_SCRIPT"
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
