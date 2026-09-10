#!/bin/bash
# ===========================================================================
# p13_esast_suite.sbatch -- the EarthSciAST Julia suite on the ess-oop-ssa
# READ-REDIRECT clone, BOTH arms of the emitter flag, Reactant tests ON.
# ===========================================================================
# The default (flag OFF) run proves the redirect arms did not perturb the stock
# emitter or its zero-allocation property; ESS_OOP_SSA=1 is what exercises the
# change itself. Run as a job because the interactive session's 40 GiB cgroup is
# shared and a concurrent CONUS probe OOM-kills the suite.
#
#   mkdir -p logs && sbatch --job-name=p13suite tools/diag/p13_esast_suite.sbatch
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
ESAST=${ESAST:-/projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciAST-ssa2}
cd "$ESAST/pkg/EarthSciAST.jl"
export ESM_TEST_REACTANT=1
echo "=== node $(hostname) $(git -C "$ESAST" rev-parse --short HEAD) ==="
for arm in off on; do
  echo "===== ESS_OOP_SSA arm=$arm  started $(date -Is)"
  if [ "$arm" = on ]; then export ESS_OOP_SSA=1; else unset ESS_OOP_SSA; fi
  julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
  echo "arm=$arm exit=$?  finished $(date -Is)"
done
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
