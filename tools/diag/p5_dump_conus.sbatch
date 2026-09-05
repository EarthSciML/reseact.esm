#!/bin/bash
# p5_dump_conus.sbatch -- post-optimization HLO dump of the TRANSPORT step and
# its VJP at CONUS, then the fusion census (p5_fusion_census.py). Answers
# whether ssp_vjp's 22x-over-primal cost is fusion write/read volume.
#   mkdir -p logs && sbatch tools/diag/p5_dump_conus.sbatch
#SBATCH --job-name=p5dumpC
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
export P5_PROGRAMS=${P5_PROGRAMS:-ssp_step,ssp_vjp}
echo "started $(date -Is) on $(hostname)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=48G tools/diag/p5_vjp_dump.jl
python3 tools/diag/p5_fusion_census.py logs/p5-dump-13x7x72-$SLURM_JOB_ID ${P5_PROGRAMS//,/ }
echo "finished $(date -Is)"
