#!/bin/bash
# p7_arms_6x6x8.sbatch -- the transport-VJP compile-option arms and the RHS-only
# probe at 6x6x8, on a compute node (the interactive cgroup is shared and full).
#   mkdir -p logs && sbatch tools/diag/p7_arms_6x6x8.sbatch
#SBATCH --job-name=p7arms
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl
J="julia --project=$RESEACT_RXENV -t 8 --heap-size-hint=16G"
export P5_PROGRAMS=ssp_step,ssp_vjp
echo "started $(date -Is) on $(hostname)"
echo "=== ARM nonan"; P5_COPTS="no_nan=true;all_finite=true" RESEACT_LABEL=p5nonan $J tools/diag/p5_vjp_time.jl | grep -E "P5_|@compile"
echo "=== ARM nobatch"; P5_COPTS="disable_slice_to_batch_passes=true;disable_concat_to_batch_passes=true" RESEACT_LABEL=p5nobatch $J tools/diag/p5_vjp_time.jl | grep -E "P5_|@compile"
echo "=== ARM stock (no exclusions)"; P5_BASEX_B=none RESEACT_LABEL=p5stock $J tools/diag/p5_vjp_time.jl | grep -E "P5_|@compile"
echo "=== ARM subconst"; P5_BASEX_B=dynamic_update_to_concat RESEACT_LABEL=p5subconst $J tools/diag/p5_vjp_time.jl | grep -E "P5_|@compile"
echo "=== P7 rhs vs rhs_vjp"; $J tools/diag/p7_rhs_vjp.jl | grep -E "P7_|@compile"
python3 tools/diag/p5_fusion_census.py logs/p7-rhs-6x6x8-$SLURM_JOB_ID rhs rhs_vjp
echo "finished $(date -Is)"
