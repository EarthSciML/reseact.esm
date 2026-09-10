#!/bin/bash
# ===========================================================================
# p14_gate_primal.sbatch -- the ess-oop-ssa SCATTER-SKIP GATE, primal programs
# ===========================================================================
# THREE arms in one process, so the acceptance question ("is the transport
# primal back to flag-off?") is answered against a flag-off build that was
# compiled and timed in the SAME process, on the same node, interleaved:
#
#   flagoff   ESS_OOP_SSA=0                  the pre-spike emitter
#   noskip    redirects on, no scatter skipped  (ESS_OOP_SSA_SKIP=0)
#   ungated   redirects on, every skippable scatter skipped == #283
#
# Both ARM ORDERS are run, sequentially in one job that holds the whole node,
# because "which arm compiled first" is a real confounder on this filesystem
# and a co-tenant inflates absolute times by up to 1.8x.
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
CODE=/projects/illinois/eng/cee/ctessum/ctessum/code
export RESEACT_REPO=${RESEACT_REPO:-$CODE/reseact.esm}
cd "$RESEACT_REPO"
export RESEACT_MODEL=${RESEACT_MODEL:-$CODE/reseact.esm/reseact.esm}
export EARTHSCIMLBASE_SRC=${EARTHSCIMLBASE_SRC:-$CODE/EarthSciMLBase/src}
export RESEACT_RXENV=${RESEACT_RXENV:-$CODE/.ssagateenv/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
export RESEACT_RES=${RESEACT_RES:-4x5}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
case "$RESEACT_RES" in
  4x5)   export RESEACT_NLON=13 RESEACT_NLAT=7  ;;
  2x2.5) export RESEACT_NLON=25 RESEACT_NLAT=13 ;;
  *) echo "FATAL: no CONUS box wired for RESEACT_RES=$RESEACT_RES"; exit 1 ;;
esac
export P14_PROGRAMS=${P14_PROGRAMS:-rhs,ssp_step,ros_step}
export P14_REPS=${P14_REPS:-40}
export P14_DUMP=${P14_DUMP:-0}
export P14_ENV_FLAGOFF=${P14_ENV_FLAGOFF:-'ESS_OOP_SSA=0'}
export P14_ENV_NOSKIP=${P14_ENV_NOSKIP:-'ESS_OOP_SSA_SKIP=0'}
export P14_ENV_UNGATED=${P14_ENV_UNGATED:-'ESS_OOP_SSA_SKIP_GATE=0'}
export P14_ENV_GATED=${P14_ENV_GATED:-''}
ARMS_FWD=${P14_ARMS:-flagoff,noskip,ungated}
ARMS_REV=$(echo "$ARMS_FWD" | tr ',' '\n' | tac | paste -sd,)
echo "=== node $(hostname) res=$RESEACT_RES nlev=$RESEACT_NLEV cpus=$SLURM_CPUS_PER_TASK progs=$P14_PROGRAMS ==="
for arms in "$ARMS_FWD" "$ARMS_REV"; do
  echo "############ ARM ORDER $arms  $(date -Is)"
  P14_ARMS=$arms julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G \
      tools/diag/p14_ssa_gate.jl
done
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
