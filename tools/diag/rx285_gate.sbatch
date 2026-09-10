#!/bin/bash
# ===========================================================================
# rx285_gate.sbatch -- does bumping Reactant 0.2.280 -> 0.2.285 move the
#                      GRADIENT?  The gate ladder, both versions, one job.
# ===========================================================================
# Runs tools/adjoint_gradient.jl once per Reactant version on the same node,
# same siblings, same base point, and writes a CSV per arm; then diffs the two
# 160-component gradient vectors with tools/diag/p11_csv_diff.py.
#
# The versions cannot share a process (they are different Julia environments),
# so the arms are sequential.  RX285_ORDER swaps which version runs first,
# because "which arm went first" is a measured confounder on this filesystem
# (DIFFERENTIABILITY_PLAN.md section 6); run both orders and believe only what
# survives the swap.
#
# The `ref` dot-product stage is expected to SKIP itself ("NO SLOT IS ZERO
# UNDER A ZERO SEED") on 0.2.280; it is run in both arms precisely so the log
# records whether 0.2.285 restores it.
#
#   sbatch --job-name=rx285gate-a tools/diag/rx285_gate.sbatch
#   sbatch --job-name=rx285gate-b --export=ALL,RX285_ORDER=rx285,rx280 \
#          tools/diag/rx285_gate.sbatch     # (comma in --export needs care)
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
CODE=/projects/illinois/eng/cee/ctessum/ctessum/code
export RESEACT_REPO=${RESEACT_REPO:-$CODE/reseact.esm}
cd "$RESEACT_REPO"
export RESEACT_MODEL=${RESEACT_MODEL:-$CODE/reseact.esm/reseact.esm}
export EARTHSCIMLBASE_SRC=${EARTHSCIMLBASE_SRC:-$CODE/EarthSciMLBase/src}
ENV_rx280=$RESEACT_REPO/run-model-jl
ENV_rx285=$RESEACT_REPO/envs/rx285
export RESEACT_RES=${RESEACT_RES:-4x5}
if [ "${RX285_CONUS:-0}" = "1" ]; then
  case "$RESEACT_RES" in
    4x5)   export RESEACT_NLON=13 RESEACT_NLAT=7  ;;
    2x2.5) export RESEACT_NLON=25 RESEACT_NLAT=13 ;;
    *) echo "FATAL: no CONUS box wired for RESEACT_RES=$RESEACT_RES"; exit 1 ;;
  esac
  export RESEACT_NLEV=${RESEACT_NLEV:-72}
else
  export RESEACT_NLON=${RESEACT_NLON:-6} RESEACT_NLAT=${RESEACT_NLAT:-6} RESEACT_NLEV=${RESEACT_NLEV:-8}
fi
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-3}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj,ref,fdtape}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-0}
ORDER=${RX285_ORDER:-rx280 rx285}
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} res=$RESEACT_RES nmacro=$RESEACT_ADJ_NMACRO stages=$RESEACT_ADJ_STAGES order=$ORDER ==="
echo "provenance:"
for d in EarthSciAST EarthSciASTDiff EarthSciASTSplitter.jl EarthSciIO; do
  printf '  %-24s %s %s\n' "$d" "$(git -C $CODE/$d rev-parse --short HEAD)" "$(git -C $CODE/$d rev-parse --abbrev-ref HEAD)"
done
echo "  reseact.esm            $(git -C $RESEACT_REPO rev-parse --short HEAD)"
echo "started $(date -Is)"
for ARM in $ORDER; do
  eval "E=\$ENV_$ARM"
  [ -d "$E" ] || { echo "FATAL: env $E missing"; exit 1; }
  V=$(julia --project="$E" -e 'using Pkg; foreach(p->p.name=="Reactant" && println(p.version), values(Pkg.dependencies()))')
  echo "=== ARM $ARM  env=$E  Reactant=$V  $(date -Is)"
  RESEACT_RXENV="$E" RESEACT_LABEL=rx285gate-$ARM \
    RESEACT_ADJ_CSV=logs/rx285gate-$ARM-$SLURM_JOB_ID.csv \
    julia --project="$E" -t 8 --heap-size-hint=64G tools/adjoint_gradient.jl 2>&1 | tail -230
done
echo "=== CSV DIFF (rx280 vs rx285) ==="
python3 tools/diag/p11_csv_diff.py logs/rx285gate-rx280-$SLURM_JOB_ID.csv logs/rx285gate-rx285-$SLURM_JOB_ID.csv
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
