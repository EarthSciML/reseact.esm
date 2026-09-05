#!/bin/bash
# esastdiff_test.sbatch -- run EarthSciASTDiff's test suite (through the
# reseact run-model-jl environment, which develops it) on a compute node.
#SBATCH --job-name=esdtest
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
echo "started $(date -Is) on $(hostname)"
# Run the suite directly in the developing environment rather than through
# Pkg.test, whose sandbox re-resolves against the Manifest's recorded versions.
julia --project=run-model-jl /projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciASTDiff/pkg/EarthSciASTDiff.jl/test/runtests.jl
echo "rc=$? finished $(date -Is)"
