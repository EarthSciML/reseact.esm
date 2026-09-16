#!/bin/bash
# ===========================================================================
# adjoint_conus_48h_direct.sbatch -- the 48 h CONUS gradient, against the
# record taken before the compiled lane changed.
# ===========================================================================
# This is `tools/diag/adjoint_conus_48h.sbatch`'s configuration value for value
# -- the record below is what the run is being compared with -- on the compiled
# lane as it stands: every program in it (the two operator-split halves, the
# symbolic Jacobian band model, and the capacity build inside each of the 8
# chemistry shard workers) is emitted by EarthSciAST's direct StableHLO emitter
# from the compiled tree-walk IR.
#
# THE RECORD IT IS COMPARED WITH (slurm 10372969, 2026-09-05, 1 h 15 m all in,
# the emitter of the day): J = 30.1943301698531 ppb mean surface O3 over 48 h;
# forward 692.6 s, backward 2,106.7 s, 47 min of loop; 19 of 160 gradient
# components nonzero, dJ/d(NEIRegrid.scale) = -2.114,
# dJ/d(DryDepositionGas.kappa) = -1.617, dJ/d(Transport3D.g_acc) = +0.669. See
# run_reseact_adjoint.jl's header.
#
# THE STEP COST IS THE DATUM. The emitter walks the plan data rather than a
# traced broadcast, so a consumer references its producer's value directly
# instead of going through a flat buffer; whether that is worth what the record
# measured is what this run answers.
#
# ENVIRONMENT. Not `run-model-jl` -- that project develops the released packages,
# and this needs three unreleased things at once: EarthSciAST's direct emitter
# and the `faq`-node-tag fixes in EarthSciASTSplitter and EarthSciASTDiff,
# without which the operator split copies the whole tendency into both halves
# and the symbolic Jacobian comes back empty. See README.md.
#
# AND IT HAS TO BE *THIS* WORKTREE'S ENVIRONMENT. The `cd` and the RESEACT_RXENV
# default below have been wrong twice now, each time inherited from the worktree
# this file was copied out of, and each time the run measured a DIFFERENT
# checkout than the one the question was about. They are `reseact-fix` and
# `retire-conus-env` here, which is EarthSciAST `retire-oop` -- the emitter
# carrying the three grid-blindness fixes (the read piece cap, the
# model-relative gather-base budget, and the canonical slot-ordered base) that
# this run exists to measure at CONUS. Check the provenance block the driver
# prints against that before believing a number out of it.
#
# WHERE TO RUN IT, which is a scheduling question and not a taste one. The
# `ctessum` partition is five nodes and they are held by whole-node jobs; a
# 40-cpu request queued 2026-09-14 was estimated to start 2026-09-17. The same
# request on `scavenger` (which includes these nodes and 58 others, MaxTime
# 1 day, AllowQos=normal, PreemptMode=REQUEUE) starts against whatever is idle:
#
#   sbatch --partition=scavenger --qos=normal tools/diag/adjoint_conus_48h_direct.sbatch
#
# Preemption requeues from the beginning, so this is a race against the
# `ctessum` copy rather than a replacement for it -- keep both and cancel the
# loser.
#
# NOT IN A SESSION, and the margin is the reason rather than the rule. The
# interactive Slurm cgroup is 40.03 GiB (42.98 GB) and this configuration --
# 4x5, 48 h, 8 shards, jac=:sym, clamp on -- measured 42.9 GB (slurm 10386109,
# tools/diag/adjoint_res_scaling.sbatch). That is 99.8% of the cgroup with the
# cgroup empty. Nor do fewer shards buy it: the peak is the `ssp_vjp` COMPILE,
# which is the driver's and which `RESEACT_ADJ_SHARDS` does not touch (the knob
# only moves the chemistry programs off the driver, and `ros_vjp` is already
# skipped under it), so dropping to N=4 saves worker RSS and leaves the peak
# where it is. An in-session attempt has already been made and killed at a
# 30 GiB guard mid-`ros_vjp` compile (adjoint_conus.sbatch's header).
#
#   sbatch tools/diag/adjoint_conus_48h_direct.sbatch
# ===========================================================================
#SBATCH --job-name=adj48d
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
# A WHOLE NODE: 8 shard workers at ~4 GB each plus the driver's compile peak,
# the same shape adjoint_conus_5d.sbatch asks for.
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
# The emitter before the read fixes did not get through `@compile ssp_step` at
# this grid in 6.5 h (slurm 10567298, XLA's own `HloCSE`, then a section-memory
# abort). 24 h is ~19x the record's 1 h 15 m, which is margin for a surprise
# rather than a projection.
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/oopretire-logs/conus-fix/adj48-direct-%j.log

set -u
# Slurm copies the script out of the repository, so the path is spelled out.
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact-fix || exit 1

export JULIA_DEPOT_PATH=/scratch/$USER/oopretire-depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia
export RESEACT_RXENV=${RESEACT_RXENV:-/scratch/$USER/retire-conus-env}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

# --- the record's configuration, value for value --------------------------
export RESEACT_NLON=${RESEACT_NLON:-13}
export RESEACT_NLAT=${RESEACT_NLAT:-7}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-576}   # 576 * 300 s = 48 h
export RESEACT_ADJ_STAGES=${RESEACT_ADJ_STAGES:-fwd,adj}
export RESEACT_ADJ_JAC=${RESEACT_ADJ_JAC:-sym}
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-1}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-0}
export RESEACT_ADJ_CLAMP=${RESEACT_ADJ_CLAMP:-1}
export RESEACT_ADJ_SHARDS=${RESEACT_ADJ_SHARDS:-8}
export RESEACT_ADJ_OBJ=${RESEACT_ADJ_OBJ:-SuperFast.O3:surf}
export RESEACT_LABEL=${RESEACT_LABEL:-conus-adj48-direct-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-/scratch/$USER/oopretire-logs/conus-fix/adj48-direct-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== env=$RESEACT_RXENV jac=$RESEACT_ADJ_JAC clamp=$RESEACT_ADJ_CLAMP" \
     "ujitter=$RESEACT_ADJ_UJITTER nmacro=$RESEACT_ADJ_NMACRO shards=$RESEACT_ADJ_SHARDS ==="
free -g | head -2
echo "started $(date -Is)"

# -t 8 for the driver; the 8 shard workers are separate processes and take the
# rest of the node.
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G run_reseact_adjoint.jl

echo "exit=$? finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
