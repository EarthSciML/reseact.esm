#!/bin/bash
# ===========================================================================
# adjoint_conus_5d.sbatch -- d(mean surface O3)/d(theta) over FIVE DAYS of CONUS,
# every runtime scalar in the model in ONE backward sweep.
# ===========================================================================
# THIS IS THE DEFAULT CONFIGURATION OF run_reseact_adjoint.jl, which is why this
# file sets almost nothing. The runner's preset block is 13x7x72, 1,440 macro
# steps of 300 s = 120 h, clamp_nonneg ON, jac=:sym, the XLA:CPU race workaround
# on, un-jittered, stages fwd,adj -- and every one of those is a `get!`, so
# anything exported here would silently WIN over it. Keep that list in the runner
# where it is documented, not duplicated here where it can drift.
# (tools/diag/adjoint_conus_48h.sbatch is the opposite convention -- it drives
# tools/adjoint_gradient.jl directly and therefore has to set all of them.)
#
# WHAT IT SHOULD COST (2026-09-05). The runner now defaults to 8 chemistry
# shards (RESEACT_ADJ_SHARDS=8) on top of the fast step, and the 48 h window
# under exactly these defaults (slurm 10372969) ran the loop in 2,799 s --
# forward 1.202 s/window, backward 3.657 s/window -- 1 h 15 m all in, MaxRSS
# ~4 GB per worker plus the driver. Five days is 2.5x that window, so expect
# ~2 h of loop plus ~20 min of setup; --time below carries ~4x margin. The
# resources are a WHOLE NODE: 40 cpus (8 workers x ~3 cores each plus the
# driver's transport programs) and 160 GB. The gradient is unchanged: the
# accept/reject ladder is byte-identical to the August single-process run, J
# agrees to 13 digits and every nonzero component to 1.3e-11.
#
# History, same window: August single process 26,222 s (8 h 02 m all in);
# 2026-09-04 fast step 7,668 s (2 h 38 m); 2026-09-05 sharded 2,799 s.
#
# 120 h has still NOT been run end to end; days 3-5 are unpaid-for rather than
# known-bad (the 36.2 h forward death was missing PBL mixing, since fixed).
#
# AND, until 2026-09-06, this file COULD NOT HAVE RUN AT ALL as written. Every
# sharded gradient on record was launched through tools/adjoint_gradient.jl
# directly; this one goes through run_reseact_adjoint.jl, which wraps each arm
# in its own module, and tools/shard_chem.jl's rpc applicator was an anonymous
# closure carrying that module name. Worker 2 died deserializing it --
# `UndefVarError: _AdjointArm not defined in Main` -- at the first rpc, before a
# single shard was built. Fixed by building the applicator in `Main`; the path
# is now exercised end to end (slurm 10386108, 6x6x8, 2 shards, replay 0.0).
#
#   mkdir -p logs && sbatch tools/diag/adjoint_conus_5d.sbatch
#   squeue -u $USER ; tail -f logs/adj5d-<jobid>.out
#
# Shorter window / single process, without editing anything (the environment
# wins over the runner's defaults):
#   sbatch --export=ALL,RESEACT_ADJ_NMACRO=576 tools/diag/adjoint_conus_5d.sbatch
#   sbatch --export=ALL,RESEACT_ADJ_SHARDS=0 --cpus-per-task=16 --mem=80G tools/diag/adjoint_conus_5d.sbatch
# ===========================================================================
#SBATCH --job-name=adj5d
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
# 8 workers at ~4 GB each plus the driver's ~40 GB compile peak; 160G is what
# the 48 h shard run requested and it fits the 192 GB nodes.
#SBATCH --mem=160G
# ~2.3 h projected all in; 12 h is ~5x that.
#SBATCH --time=12:00:00
# NB: slurm will NOT create this directory -- submit with `mkdir -p logs` first
# or the job is rejected at submission. logs/ is gitignored.
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

# run-model-jl, no overrides: since the esm 1.0.0 migration it develops all four
# packages at their release versions (EarthSciAST 0.1.1, EarthSciASTDiff 0.1.0,
# EarthSciASTSplitter 0.1.0, EarthSciIO 0.1.2), so jac=:sym needs no pinned env
# and the model is this repo's own working copy.
export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_LABEL=${RESEACT_LABEL:-conus-adj5d-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-logs/adj5d-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
free -g | head -2
echo "started $(date -Is)"

# -t 8 for the driver; the 8 shard workers are separate processes and take the
# rest of the node.
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G run_reseact_adjoint.jl

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
