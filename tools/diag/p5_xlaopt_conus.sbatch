#!/bin/bash
# p5_xlaopt_conus.sbatch -- XLA:CPU debug-option arms for the transport step and
# its VJP at CONUS (p5_vjp_time.jl arm B = P5_XLAOPT). Candidates found
# 2026-09-05: copy-insertion region analysis (the VJP carries 394 copies of the
# 2 MB extended buffer per call) and fast min/max (the PPM limiter is min/max/abs).
#   mkdir -p logs && sbatch tools/diag/p5_xlaopt_conus.sbatch
#SBATCH --job-name=p5xlaC
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
export RESEACT_NLON=13 RESEACT_NLAT=7 RESEACT_NLEV=72
export P5_PROGRAMS=ssp_step,ssp_vjp P5_REPS=30
J="julia --project=$RESEACT_RXENV -t 8 --heap-size-hint=48G"
echo "started $(date -Is) on $(hostname)"
for opt in "xla_cpu_copy_insertion_use_region_analysis=true" "xla_cpu_enable_fast_min_max=true"; do
  echo "=== ARM $opt  $(date -Is)"
  P5_XLAOPT="$opt" RESEACT_LABEL=p5xla $J tools/diag/p5_vjp_time.jl | grep -E "P5_|@compile"
done
echo "finished $(date -Is)"
