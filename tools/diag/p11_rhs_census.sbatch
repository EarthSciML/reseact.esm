#!/bin/bash
# ===========================================================================
# p11_rhs_census.sbatch -- the transport RHS and its reverse, BOTH arms of
# ess-oop-levelbase, timed and censused at a real resolution.
# ===========================================================================
# p7_rhs_vjp.jl with P7_LEVELBASE=1: one build, one process, four programs
# (rhs-lb0, rhs-lb1, rhs_vjp-lb0, rhs_vjp-lb1), each timed and dumped, values
# gated bit for bit across the arms. Then tools/diag/p11_ue_traffic.py reports
# the whole-extended-buffer fusion and copy census per arm -- which is the
# before/after the observed prelude's adjoint is measured by:
#
#   BASELINE (slurm 10375981 / 10375862, 13x7x72, extended length 247 423 =
#   1.98 MB). Counting a whole-buffer fusion at its RESULT SHAPE overstates the
#   traffic: every dynamic_update_slice into the extended vector already ALIASES
#   its operand (the buffer assignment gives operand and result the same
#   allocation and offset), so XLA:CPU writes only the update. The honest column
#   is "real element writes", which counts an aliased whole-buffer fusion at ~0
#   and counts the copies:
#
#     program    real element writes   extended-buffer share   whole-buffer copies
#     rhs             2.49 M (  20 MB)        29.8%              2 (   4.0 MB)
#     rhs_vjp        56.33 M ( 451 MB)        51.0%             98 ( 194.0 MB)
#     ssp_step       10.43 M (  83 MB)        28.5%              8 (  15.8 MB)
#     ssp_vjp       217.36 M (1739 MB)        52.7%            394 ( 779.9 MB)
#
#   So the VJP writes ~21x the primal's bytes at the same ~5 GB/s, and 45% of
#   that traffic is copy-insertion COPIES of the extended vector. The liveness
#   that forces them is in the REVERSE, not in the forward's read threading --
#   see the RESULT block below, which is what this probe was written to
#   establish and what it measured.
#
#   mkdir -p logs
#   sbatch --job-name=p11cen-4x5  --export=ALL,RESEACT_RES=4x5   tools/diag/p11_rhs_census.sbatch
#   sbatch --job-name=p11cen-2x25 --export=ALL,RESEACT_RES=2x2.5 tools/diag/p11_rhs_census.sbatch
#
# RESULT (slurm 10430018, 4x5 = 13x7x72, ccc0235, 8 threads, EarthSciAST
# perf/level-base-read-version b23b3c49). ess-oop-levelbase IS EXACT AND DOES
# NOT PAY -- do not ship it:
#
#   program    OFF                    ON                     ratio
#   rhs         3.275 ms (min 3.181)   3.677 ms (min 3.411)   0.89
#   rhs_vjp    78.318 ms (min 75.612) 79.596 ms (min 75.742)  0.98
#   rhs values bit-for-bit; rhs_vjp lambda bit-for-bit, p-gradient 3.784e-15
#
#   whole-buffer copies   rhs 3 -> 5 (5.9 -> 9.9 MB)
#                     rhs_vjp 99 -> 102 (196.0 -> 201.9 MB)
#
# WHY. 82 of rhs_vjp's 99 whole-buffer copies copy the result of a ZEROING DUS,
# and those live in the REVERSE, where levelbase cannot reach. Each is
# `dynamic-update-slice(prev, broadcast(0)[91], <constant offset>)` over the
# whole 247 423-element cotangent, emitted IN PLACE; each producer's own
# 91-element cotangent is then assembled by an f64[91] fusion taking ~70
# WHOLE-BUFFER operands -- one per later version of the chain -- and summing a
# STATIC 91-element `slice` from each. Every version therefore has ~70
# consumers, the in-place write of version k+1 would destroy version k, and
# copy-insertion copies every version. Freezing the FORWARD's read version does
# not change that; it only moves the forward's own copy (3 -> 5), which is the
# 12 % the forward lost.
#
# THE ONE THING THAT WOULD REMOVE IT is a static rewrite:
# `slice(dynamic_update_slice(x, 0, c), [a:b]) == slice(x, [a:b])` for
# [a:b] disjoint from [c, c+91). Every window here is a compile-time constant
# and they ARE disjoint. It does not fire on Reactant 0.2.280
# (`dus_slice_simplify` is on by default and is not excluded here), and it is
# upstream of both this repo and EarthSciAST.
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
# RESEACT_REPO lets this run from a worktree; RESEACT_RXENV must name an
# environment whose EarthSciAST HAS ess-oop-levelbase (branch
# perf/level-base-read-version) -- the probes error out if it does not. A
# worktree checkout's own relative sibling paths escape it, so RESEACT_MODEL and
# EARTHSCIMLBASE_SRC (both documented overrides) are set explicitly.
CODE=/projects/illinois/eng/cee/ctessum/ctessum/code
export RESEACT_REPO=${RESEACT_REPO:-$CODE/reseact.esm}
cd "$RESEACT_REPO"
export RESEACT_MODEL=${RESEACT_MODEL:-$CODE/reseact.esm/reseact.esm}
export EARTHSCIMLBASE_SRC=${EARTHSCIMLBASE_SRC:-$CODE/EarthSciMLBase/src}
export RESEACT_RXENV=${RESEACT_RXENV:-$RESEACT_REPO/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
export RESEACT_RES=${RESEACT_RES:-4x5}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
# p5_vjp_time.jl / p7_rhs_vjp.jl `get!` RESEACT_NLON/NLAT to 6 before including
# the driver, so the resolution row's CONUS box would never be reached: set it
# here explicitly (13x7 at 4x5, 25x13 at 2x2.5 -- the same geography, which is
# the whole point of comparing resolutions).
case "$RESEACT_RES" in
  4x5)   export RESEACT_NLON=13 RESEACT_NLAT=7  ;;
  2x2.5) export RESEACT_NLON=25 RESEACT_NLAT=13 ;;
  *) echo "FATAL: no CONUS box wired for RESEACT_RES=$RESEACT_RES"; exit 1 ;;
esac
export P7_LEVELBASE=1
export P7_REPS=${P7_REPS:-40}
echo "=== node $(hostname) res=$RESEACT_RES nlev=$RESEACT_NLEV cpus=$SLURM_CPUS_PER_TASK ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/p7_rhs_vjp.jl
# The grid string p7 builds its output directory from comes from the RES row.
G=$(ls -d logs/p7-rhs-*-"$SLURM_JOB_ID" 2>/dev/null | head -1)
echo "=== dump dir $G"
for a in rhs-lb0 rhs-lb1 rhs_vjp-lb0 rhs_vjp-lb1; do
  python3 tools/diag/p11_ue_traffic.py "$G/$a"
done
python3 tools/diag/p5_fusion_census.py "$G" rhs-lb0 rhs-lb1 rhs_vjp-lb0 rhs_vjp-lb1
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
