#!/bin/bash
# ===========================================================================
# p11_esast_suite.sbatch -- the EarthSciAST Julia suite on the
# perf/level-base-read-version clone, with the Reactant tracing tests ON.
# ===========================================================================
# ess-oop-levelbase is a no-op on host by construction (`_oop_read_version` is
# the identity there), so what the default suite proves is that it did not
# perturb the host walk or its zero-allocation property; ESM_TEST_REACTANT=1 is
# what exercises the change itself, including the new
# test/reactant_oop_levelbase_test.jl.
#
#   mkdir -p logs && sbatch --job-name=p11suite tools/diag/p11_esast_suite.sbatch
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
# ESAST names the EarthSciAST checkout under test.
ESAST=${ESAST:-/projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciAST-vjpue}
cd "$ESAST/pkg/EarthSciAST.jl"
export ESM_TEST_REACTANT=1
echo "=== node $(hostname) $(git -C "$ESAST" rev-parse --short HEAD) ==="
echo "started $(date -Is)"
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
echo "exit=$?"
echo "finished $(date -Is)"
