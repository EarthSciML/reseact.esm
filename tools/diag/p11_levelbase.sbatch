#!/bin/bash
# ===========================================================================
# p11_levelbase.sbatch -- price EarthSciAST's ess-oop-levelbase (ONE read
# version per materialized-observed dependency level) on the transport step
# and its VJP, at a real resolution.
# ===========================================================================
# WHY IT IS ONE JOB WITH BOTH ARMS IN ONE PROCESS. `ESS_OOP_LEVELBASE` lives in
# a `Ref` read at TRACE time, so p5_vjp_time.jl compiles the SAME build twice
# with IDENTICAL compile options and times the two interleaved A/B/A/B on the
# same inputs with fresh device buffers. Setup wall is not a usable metric on
# this filesystem (identical baselines have spanned 479-706 s to SHARDS READY),
# and compile order alone has moved a compile time 1.45x -- so the arms have to
# be adjacent in one process, and P5_LEVELBASE=2 swaps which one compiles first.
#
#   mkdir -p logs
#   sbatch --job-name=p11lb-4x5  --export=ALL,RESEACT_RES=4x5   tools/diag/p11_levelbase.sbatch
#   sbatch --job-name=p11lb-2x25 --export=ALL,RESEACT_RES=2x2.5 tools/diag/p11_levelbase.sbatch
#
# RESEACT_RES carries the CONUS index box (13x7 at 4x5, 25x13 at 2x2.5), so
# only NLEV is set here and both arms are the same geography.
#
# THE ENVIRONMENT. `run-model-jl` is shared and its Manifest pins the canonical
# EarthSciAST checkout, so do not edit it. Copy it instead and repoint that one
# entry at a checkout that HAS ess-oop-levelbase:
#
#   mkdir run-model-jl-lb
#   cp run-model-jl/Project.toml run-model-jl-lb/
#   sed 's|/code/EarthSciAST/|/code/<your-clone>/|' run-model-jl/Manifest.toml \
#       > run-model-jl-lb/Manifest.toml
#   sbatch --export=ALL,RESEACT_RXENV=$PWD/run-model-jl-lb,... tools/diag/p11_levelbase.sbatch
#
# MEMORY. Two ssp_vjp compiles in ONE process at CONUS does NOT fit in 100G:
# slurm 10430019 (4x5, --mem=100G) got through ssp_step's pair and then
# "Aborted (core dumped)" after a 714 s ssp_vjp compile. Ask for the node
# (--mem=180G) and, for the VJP alone, pass P5_PROGRAMS=ssp_vjp so the process
# is not also holding the driver's ssp_step/ros_step programs.
#
# RESULT, ess-oop-levelbase (slurm 10430019 / 10430126, 4x5 = 13x7x72). EXACT
# AND DOES NOT PAY -- see tools/diag/p11_rhs_census.sbatch for why, and
# DIFFERENTIABILITY_PLAN.md section 6 lever 2 for the net:
#
#   program    OFF          ON           OFF/ON   values
#   ssp_step   16.391 ms    16.753 ms    0.978    bit-for-bit
#   ros_step   42.318 ms    42.694 ms    0.991    bit-for-bit
#   ros_vjp    90.806 ms    90.203 ms    1.007    bit-for-bit
#
# ARM ORDER IS NOT WHAT IS BEING MEASURED: the chemistry pair re-run with the
# arms swapped (P5_LEVELBASE=2, so "default" is ON and "+extra" is OFF) reads
# ros_step 42.784 / 42.955 and ros_vjp 91.612 / 91.538, i.e. OFF/ON of 1.004
# and 0.999 against 0.991 and 1.007 the other way round. Both ratios straddle
# 1.0 in both orders: flat, and the sign is noise.
#
# and at the RHS level, which IS the transport VJP (the 4-stage step VJP is
# four of exactly this reverse): rhs 0.89, rhs_vjp 0.98.
#
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
# RESEACT_REPO lets this run from a worktree; RESEACT_RXENV must name an
# environment whose EarthSciAST HAS ess-oop-levelbase (branch
# perf/level-base-read-version) -- the probes error out if it does not. A
# worktree checkout's own relative sibling paths escape it, so RESEACT_MODEL
# and EARTHSCIMLBASE_SRC (both documented overrides) are set explicitly.
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
export P5_PROGRAMS=${P5_PROGRAMS:-ssp_step,ssp_vjp}
export P5_REPS=${P5_REPS:-30}
J="julia --project=$RESEACT_RXENV -t 8 --heap-size-hint=64G"
echo "=== node $(hostname) res=$RESEACT_RES nlev=$RESEACT_NLEV cpus=$SLURM_CPUS_PER_TASK ==="
echo "started $(date -Is)"
echo "=== ARM ORDER off-then-on (P5_LEVELBASE=1)"
P5_LEVELBASE=1 RESEACT_LABEL=p11lb1 $J tools/diag/p5_vjp_time.jl 2>&1 | grep -E "P5_|@compile|NS=|BUILD"
echo "=== ARM ORDER on-then-off (P5_LEVELBASE=2)"
P5_LEVELBASE=2 RESEACT_LABEL=p11lb2 $J tools/diag/p5_vjp_time.jl 2>&1 | grep -E "P5_|@compile|NS=|BUILD"
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
