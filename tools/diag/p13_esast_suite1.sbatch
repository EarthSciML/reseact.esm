#!/bin/bash
# ===========================================================================
# p13_esast_suite1.sbatch -- the EarthSciAST Julia suite on the ess-oop-ssa
# read-redirect clone, ONE arm per job (SSA_ARM=off|on), Reactant tests ON.
# ===========================================================================
# One arm per job because `Pkg.test()` with ESM_TEST_REACTANT=1 was KILLed at
# 64 GB: the ReactantExt precompile peaks high and the traced tests compile real
# modules. 120 GB and 4 threads clears it.
#
#   mkdir -p logs
#   SSA_ARM=off sbatch --export=ALL --job-name=p13suite-off tools/diag/p13_esast_suite1.sbatch
#   SSA_ARM=on  sbatch --export=ALL --job-name=p13suite-on  tools/diag/p13_esast_suite1.sbatch
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
ESAST=${ESAST:-/projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciAST-ssa2}
ARM=${SSA_ARM:-off}
cd "$ESAST/pkg/EarthSciAST.jl"
export ESM_TEST_REACTANT=1
[ "$ARM" = on ] && export ESS_OOP_SSA=1 || unset ESS_OOP_SSA
echo "=== node $(hostname) $(git -C "$ESAST" rev-parse --short HEAD) arm=$ARM ESS_OOP_SSA=${ESS_OOP_SSA:-unset} ==="
echo "started $(date -Is)"
julia --project=. -t 4 --heap-size-hint=48G -e 'using Pkg; Pkg.test()'
echo "arm=$ARM exit=$?  finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
