#!/bin/bash
# ===========================================================================
# direct_rhs_agree.sbatch -- the four compiling runs of tools/diag/direct_rhs_agree.jl
# in one allocation, ONE PROCESS EACH.
# ===========================================================================
# The two lanes need two Reactant versions, so they need two environments, and
# each half needs its own process: `ESS_OOP_SSA` is read at BUILD time, and a
# traced XLA compile of the chemistry half has been OOM-killed in the shared
# cgroup beside another Julia. Sequential inside one allocation keeps the peak
# to one compile at a time and the whole comparison to one job id.
#
#   sbatch tools/diag/direct_rhs_agree.sbatch
#
#SBATCH --job-name=dragree
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/oopretire-logs/agree-%j.log
set -u
# Slurm copies the script out of the repository, so the path is spelled out
# (the same way tools/diag/direct_rhs_census_conus.sbatch does it).
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact-directrhs || exit 1

export JULIA_DEPOT_PATH=/scratch/$USER/oopretire-depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia
export RESEACT_AGREE_DIR=/scratch/$USER/oopretire-logs/agree
export RESEACT_NLON=${RESEACT_NLON:-6}
export RESEACT_NLAT=${RESEACT_NLAT:-6}
export RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-1e-1}
mkdir -p "$RESEACT_AGREE_DIR"

# The traced lane's environment is production's -- Reactant 0.2.280 and
# EarthSciAST `main` -- with ONE change: EarthSciASTSplitter is dev-linked to the
# `faq`-node-tag fix. Without it the splitter's lifted-derivative recogniser
# matches the pre-1.1.0 `aggregate` tag that EarthSciAST normalizes away on load,
# `split_equations` copies the WHOLE tendency into BOTH parts, and the comparison
# is between two copies of the unsplit model.
PROD=/scratch/$USER/oopretire-env-280traced  # Reactant 0.2.280, EarthSciAST main
DIRECT=/scratch/$USER/oopretire-env-faq2     # Reactant 0.2.285, EarthSciAST oop-retire
J="timeout -k 15 3600 julia --heap-size-hint=24G"

# The TRACED lane as production runs it, including production's `ESS_OOP_SSA=1`.
for h in 1 2; do
  echo "=== traced half $h ==="
  RESEACT_AGREE_MODE=traced RESEACT_AGREE_HALF=$h ESS_OOP_SSA=1 \
    RESEACT_RXENV=$PROD $J --project=$PROD tools/diag/direct_rhs_agree.jl \
    || echo "traced half $h FAILED ($?)"
done

# The DIRECT lane on the traced lane's own saved inputs.
for h in 1 2; do
  echo "=== direct half $h ==="
  RESEACT_AGREE_MODE=direct RESEACT_AGREE_HALF=$h ESS_OOP_SSA=1 \
    RESEACT_RXENV=$DIRECT $J --project=$DIRECT tools/diag/direct_rhs_agree.jl \
    || echo "direct half $h FAILED ($?)"
done

echo "=== compare ==="
RESEACT_AGREE_MODE=compare RESEACT_AGREE_HALF=both \
  timeout -k 15 600 julia --project=$DIRECT tools/diag/direct_rhs_agree.jl
