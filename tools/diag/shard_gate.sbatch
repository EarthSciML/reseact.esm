#!/bin/bash
# ===========================================================================
# shard_gate.sbatch -- the 6x6x8 correctness gate for the chemistry SHARDS.
# ===========================================================================
# fwd,adj over 3 macro steps at 6x6x8 with RESEACT_ADJ_SHARDS=N (0 = the
# unsharded reference), un-jittered, clamp on. The gate compares the gradient
# CSV and the accept/reject ladder of N=2,4 against N=0 (see the P5 report).
#
#   sbatch --export=ALL,RESEACT_ADJ_SHARDS=0 tools/diag/shard_gate.sbatch
#   sbatch --export=ALL,RESEACT_ADJ_SHARDS=2 tools/diag/shard_gate.sbatch
#   sbatch --export=ALL,RESEACT_ADJ_SHARDS=4 tools/diag/shard_gate.sbatch
# ===========================================================================
#SBATCH --job-name=shgate
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/.claude/worktrees/agent-afee560f0db87627c/logs/%x-%j.out

set -euo pipefail
SHARD_REPO=${SHARD_REPO:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/.claude/worktrees/agent-afee560f0db87627c}
cd "$SHARD_REPO"
export RESEACT_RXENV=${RESEACT_RXENV:-$SHARD_REPO/run-model-jl}
export RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-3}
export RESEACT_ADJ_SHARDS=${RESEACT_ADJ_SHARDS:-2}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_LABEL=${RESEACT_LABEL:-gate-s${RESEACT_ADJ_SHARDS}-${SLURM_JOB_ID}}
export RESEACT_ADJ_CSV=logs/${RESEACT_LABEL}.csv
export SHARD_LOG=logs/${RESEACT_LABEL}.log
export JULIA_ARGS=${JULIA_ARGS:--t 4 --heap-size-hint=12G}
echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE}M shards=$RESEACT_ADJ_SHARDS stages=$RESEACT_ADJ_STAGES ==="
echo "load $(cut -d' ' -f1-3 /proc/loadavg)   started $(date -Is)"
tools/diag/shard_run.sh
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
