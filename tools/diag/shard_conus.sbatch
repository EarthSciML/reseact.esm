#!/bin/bash
# ===========================================================================
# shard_conus.sbatch -- CONUS timing of the chemistry SHARDS (tools/shard_chem.jl).
# ===========================================================================
# One node, the whole node: N worker processes each owning a capacity build of
# NC/N cells, the driver holding transport. Default window is 3 macro steps
# (RESEACT_ADJ_NMACRO=3) with fwd,adj, which is the configuration every
# per-step CONUS number in this repo was taken in (slurm 10042579: forward
# 46.16 s, backward 124.91 s, 0.4955 s/VJP with the stock emitter).
#
#   sbatch --export=ALL,RESEACT_ADJ_SHARDS=8 tools/diag/shard_conus.sbatch
#   sbatch --export=ALL,RESEACT_ADJ_SHARDS=0 tools/diag/shard_conus.sbatch   # reference
#
# SHARD_REPO selects the checkout (a git worktree by default -- the model's
# ../EarthSciModels refs are resolved through RESEACT_MODEL, see shard_run.sh).
#
# MEASURED 2026-09-05 (3 windows, per window): N=0 fwd 3.19 s / bwd 17.83 s
# (11.59 with the epoch-cached refresh); N=4 2.14 / 9.90; N=8 1.94 / 5.72;
# N=13 1.85 / 9.14 (pre-fix, no gain over N=8). Jobs 10371241-4, 10372581/2,
# 10372724/5, 10372885. The 48 h window at N=8 (RESEACT_ADJ_NMACRO=576,
# --time=08:00:00, slurm 10372969): 1 h 15 m, 1.202 / 3.657 s per window.
# Full tables in tools/shard_chem.jl.
# ===========================================================================
#SBATCH --job-name=shconus
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/.claude/worktrees/agent-afee560f0db87627c/logs/%x-%j.out

set -euo pipefail
SHARD_REPO=${SHARD_REPO:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/.claude/worktrees/agent-afee560f0db87627c}
cd "$SHARD_REPO"
export RESEACT_RXENV=${RESEACT_RXENV:-$SHARD_REPO/run-model-jl}
export RESEACT_NLON=13 RESEACT_NLAT=7 RESEACT_NLEV=72
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-3}
export RESEACT_ADJ_SHARDS=${RESEACT_ADJ_SHARDS:-8}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_LABEL=${RESEACT_LABEL:-conus-s${RESEACT_ADJ_SHARDS}-${SLURM_JOB_ID}}
export RESEACT_ADJ_CSV=logs/${RESEACT_LABEL}.csv
export SHARD_LOG=logs/${RESEACT_LABEL}.log
export JULIA_ARGS=${JULIA_ARGS:--t 8 --heap-size-hint=64G}
echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE}M shards=$RESEACT_ADJ_SHARDS nmacro=$RESEACT_ADJ_NMACRO ==="
echo "Cpus_allowed_list: $(grep Cpus_allowed_list /proc/self/status | cut -f2)"
echo "load $(cut -d' ' -f1-3 /proc/loadavg)   started $(date -Is)"
tools/diag/shard_run.sh
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
