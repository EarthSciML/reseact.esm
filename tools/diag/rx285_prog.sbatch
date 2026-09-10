#!/bin/bash
# ===========================================================================
# rx285_prog.sbatch -- per-program timings and outputs, one arm per Reactant
#                      version / excluded-pass setting, in one job.
# ===========================================================================
# Two Julia environments cannot share a process, so the arms are sequential
# processes on the SAME node; setup wall is not a usable metric here (SHARDS
# READY spans 479.5/540.7/706.5 s for identical code and an order swap moved
# `compile step` 219.8 -> 151.2 s), so what is compared is the PROGRAM time,
# best-of-two-passes median, and both arm ORDERS are run as separate jobs.
#
# RX285_ARMS is a space-separated list of <env>:<excluded-passes> where <env>
# is rx280 (run-model-jl, the pin) or rx285 (envs/rx285), and the exclusion is
# `default` (dynamic_update_to_concat,sub_const_prop), `none` (stock Enzyme-JAX
# pipeline) or an explicit comma list -- with commas written as `+` because
# --export cannot carry a bare comma.
#
#   sbatch --job-name=rx285prog-a --export=ALL,RX285_CONUS=1,RX285_ARMS='rx280:default rx285:default' \
#          tools/diag/rx285_prog.sbatch
#   sbatch --job-name=rx285excl --export=ALL,RX285_CONUS=1,RX285_ARMS='rx285:default rx285:none rx285:dynamic_update_to_concat' \
#          tools/diag/rx285_prog.sbatch
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=12:00:00
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
export RX285_PROGRAMS=${RX285_PROGRAMS:-ssp_step,ssp_vjp,ros_step,ros_vjp,rhs,rhs_vjp}
export RX285_REPS=${RX285_REPS:-40}
ARMS=${RX285_ARMS:-rx280:default rx285:default}
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} res=$RESEACT_RES progs=$RX285_PROGRAMS arms='$ARMS' ==="
echo "started $(date -Is)"
DUMPS=""
for ARM in $ARMS; do
  V=${ARM%%:*}; X=${ARM#*:}; X=${X//+/,}
  eval "E=\$ENV_$V"
  [ -d "$E" ] || { echo "FATAL: env $E missing"; exit 1; }
  LBL="$V-$(echo "$X" | tr ',' '_')"
  OUT=logs/rx285prog-$LBL-$SLURM_JOB_ID.jls
  echo "=== ARM $LBL  env=$E  excluded_passes=$X  $(date -Is)"
  RESEACT_RXENV="$E" RESEACT_LABEL="$LBL" RESEACT_EXCLUDED_PASSES="$X" RX285_OUT="$OUT" \
    julia --project="$E" -t 8 --heap-size-hint=100G tools/diag/rx285_prog.jl 2>&1 | \
    grep -E "RX285_|SHARDS READY|ERROR|error:" | tail -80
  DUMPS="$DUMPS $OUT"
done
set -- $DUMPS
echo "=== DIFFS against $1 ==="
for d in "${@:2}"; do
  julia --project="$ENV_rx285" tools/diag/rx285_prog_diff.jl "$1" "$d"
done
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
