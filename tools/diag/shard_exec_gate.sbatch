#!/bin/bash
# ===========================================================================
# shard_exec_gate.sbatch -- the gate for the chemistry-shard EXECUTOR SEAM.
# ===========================================================================
# Runs tools/adjoint_gradient.jl at 6x6x8, 3 windows, un-jittered, clamp on --
# the same configuration as the 2026-09-05 sharding gate (slurm 10366675/6/7)
# whose numbers are quoted in tools/shard_chem.jl's header -- against a
# CHECKOUT and an EXECUTOR chosen by the environment, so the same job file
# produces the before/after pair and the second-executor arm.
#
#   SEAM_TREE            the checkout to run (required; a `git archive HEAD`
#                        copy under /scratch.local for the "before" arm)
#   RESEACT_ADJ_SHARDS   shard count (default 4; 0 = unsharded)
#   RESEACT_SHARD_EXEC   process (default) | inprocess   -- ignored by the
#                        pre-seam checkout, which has only the process path
#   RESEACT_ADJ_STAGES   default fwd,adj; the unsharded arm uses
#                        fwd,adj,ref,fdtape to get the dot-product identity
#                        and the finite-difference gate
#
# Submit FROM the worktree (logs/ and the CSVs are relative to the submit dir):
#   mkdir -p logs
#   sbatch --export=ALL,SEAM_TREE=...,RESEACT_ADJ_SHARDS=4 tools/diag/shard_exec_gate.sbatch
#
# MEMORY. The process executor spawns N workers; at 6x6x8 each is ~2 GB, so 64G
# is ample. Do NOT run this interactively -- the session cgroup is 40 GiB and
# shared between agents.
# ===========================================================================
#SBATCH --job-name=shardexec
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

SUBMITDIR=$SLURM_SUBMIT_DIR
TREE=${SEAM_TREE:?SEAM_TREE must name the checkout to run}
cd "$TREE" || exit 1

JULIA=/u/ctessum/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia

# One environment for BOTH arms, so the package versions cannot be a
# confounder: the main checkout's run-model-jl. It is shared -- read only.
export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_NLON=${RESEACT_NLON:-6}
export RESEACT_NLAT=${RESEACT_NLAT:-6}
export RESEACT_NLEV=${RESEACT_NLEV:-8}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-3}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-0}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-1}
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-1}
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_ADJ_SHARDS=${RESEACT_ADJ_SHARDS:-4}
export RESEACT_SHARD_EXEC=${RESEACT_SHARD_EXEC:-process}
export RESEACT_LABEL=${RESEACT_LABEL:-shardexec-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-$SUBMITDIR/logs/$RESEACT_LABEL.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== tree=$TREE  shards=$RESEACT_ADJ_SHARDS  exec=$RESEACT_SHARD_EXEC  stages=$RESEACT_ADJ_STAGES ==="
git -C "$TREE" rev-parse HEAD 2>/dev/null || echo "(tree is not a git checkout: archived copy)"

"$JULIA" --project="$RESEACT_RXENV" -t 8 --heap-size-hint=16G tools/adjoint_gradient.jl

echo "=== exit $? ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State 2>/dev/null || true
