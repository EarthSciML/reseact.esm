#!/bin/bash
# p7_rhs_conus.sbatch -- transport RHS vs its reverse at CONUS, timed and
# dumped, then the fusion census.   mkdir -p logs && sbatch tools/diag/p7_rhs_conus.sbatch
#SBATCH --job-name=p7rhsC
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export RESEACT_NLON=13 RESEACT_NLAT=7 RESEACT_NLEV=72
echo "started $(date -Is) on $(hostname)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=48G tools/diag/p7_rhs_vjp.jl
python3 tools/diag/p5_fusion_census.py logs/p7-rhs-13x7x72-$SLURM_JOB_ID rhs rhs_vjp
echo "finished $(date -Is)"
