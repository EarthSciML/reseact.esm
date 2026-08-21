#!/bin/bash
# ===========================================================================
# adjoint_conus.sbatch -- run the CONUS adjoint on a real node.
# ===========================================================================
# WHY THIS EXISTS. An interactive Claude Code session lives in a Slurm cgroup
# capped at 40 GiB, which is NOT enough for a CONUS adjoint: the `ros_vjp`
# compile alone pushed cgroup anon past 30 GiB once the blocker-4 workaround
# (xla_cpu_prefer_vector_width=128) was switched on -- Julia RSS went 5.5 -> 13.3 GB
# during that one trace, and the XLA side allocates well beyond what Julia
# reports. Without the workaround the same compile stayed under 30 GiB, so the
# workaround is NOT free at CONUS: it roughly doubles compile memory.
#
# The ctessum partition gives 40 cores / 192 GB / 3-day walltime per node, which
# removes the constraint entirely. This is also the template the eventual week
# run should use.
#
#   sbatch tools/diag/adjoint_conus.sbatch
#   squeue -u $USER ; tail -f slurm-<jobid>.out
#
# Env overrides are read from the environment, so a variant run is e.g.
#   sbatch --export=ALL,RESEACT_ADJ_CLAMP=1 tools/diag/adjoint_conus.sbatch
# ===========================================================================
#SBATCH --job-name=reseact-adj
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Right-sized, NOT maximal: a whole-node request (40 cpus / 180G) sat PENDING on
# (Resources) for an hour -- the ctessum nodes had 39/40 CPUs idle but their
# MEMORY was already reserved by other jobs. Measured peak for this run is ~35 GiB
# (the in-session attempt was killed at a 30 GiB guard mid-ros_vjp compile), so
# 80G with headroom schedules immediately and is honest about what is needed.
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
# NB: slurm will NOT create this directory -- submit with `mkdir -p logs` first
# or the job is rejected at submission. logs/ is gitignored.
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

JULIA=/u/ctessum/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia

# The symbolic block Jacobian (RESEACT_ADJ_JAC=sym, now the default) needs
# EarthSciASTDiff. run-model-jl cannot resolve it -- its EarthSciAST checkout is
# on the 0.1.x line while EarthSciASTDiff requires 0.9.x -- so this job runs out
# of env-sym, which is run-model-jl's dependency set plus a pinned
# EarthSciASTDiff, against the corpus-pinned model that env can load.
export RESEACT_RXENV=${RESEACT_RXENV:-/u/ctessum/reseact-esast-pin/env-sym}
export RESEACT_MODEL=${RESEACT_MODEL:-/u/ctessum/reseact-esast-pin/corpus-pin/reseact.esm/reseact.esm}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
[ -f "$RESEACT_MODEL" ] || { echo "FATAL: model $RESEACT_MODEL missing"; exit 1; }

export RESEACT_NLON=${RESEACT_NLON:-13}
export RESEACT_NLAT=${RESEACT_NLAT:-7}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-0}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-3}
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-1}
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_LABEL=${RESEACT_LABEL:-conus-adj-$SLURM_JOB_ID}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== clamp=$RESEACT_ADJ_CLAMP xlafix=$RESEACT_ADJ_XLAFIX jac=$RESEACT_ADJ_JAC nmacro=$RESEACT_ADJ_NMACRO ==="
free -g | head -2

# -t 8 matches what every other measurement in this repo used, so the timings
# stay comparable; the extra cores are headroom for XLA's own thread pool.
"$JULIA" --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/adjoint_gradient.jl

echo "=== peak RSS: $(grep VmHWM /proc/self/status 2>/dev/null || echo n/a) ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State 2>/dev/null || true
