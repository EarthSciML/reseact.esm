#!/bin/bash
#SBATCH --job-name=esasttest
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
PKG=${ESAST_PKG:-/projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciAST-buildperf/pkg/EarthSciAST.jl}
export JULIA_NUM_THREADS=4
echo "=== node $(hostname) pkg=$PKG env-overrides: LANE_AFFINE_KEY_DISABLE=${ESS_LANE_AFFINE_KEY_DISABLE:-} STATE_BOX_DISABLE=${ESS_STATE_BOX_DISABLE:-} ==="
echo "started $(date -Is)"
julia -e "using Pkg; Pkg.activate(\"$PKG\"); Pkg.test()" 2>&1 | tee logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.txt
echo "exit=$? finished $(date -Is)"
