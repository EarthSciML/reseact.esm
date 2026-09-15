#!/bin/bash
# ===========================================================================
# adjoint_conus_48h_direct.sbatch -- the 48 h CONUS gradient through the
# DIRECT StableHLO emitter (RESEACT_RHS=direct), against the traced record.
# ===========================================================================
# This is `tools/diag/adjoint_conus_48h.sbatch`'s run with ONE thing changed:
# every compiled program in it -- the two operator-split halves, the symbolic
# Jacobian band model, and the capacity build inside each of the 8 chemistry
# shard workers -- is emitted by EarthSciAST's direct StableHLO emitter instead
# of by tracing the `:oop` broadcast emitter through Reactant. Everything else
# is the configuration of the record below, value for value, because the record
# is what the run is being compared with.
#
# THE RECORD IT IS COMPARED WITH (slurm 10372969, 2026-09-05, traced lane,
# 1 h 15 m all in): J = 30.1943301698531 ppb mean surface O3 over 48 h; forward
# 692.6 s, backward 2,106.7 s, 47 min of loop; 19 of 160 gradient components
# nonzero, dJ/d(NEIRegrid.scale) = -2.114, dJ/d(DryDepositionGas.kappa) =
# -1.617, dJ/d(Transport3D.g_acc) = +0.669. See run_reseact_adjoint.jl's header.
#
# THE RECORD RAN WITH `ESS_OOP_SSA=1`, AND SO DOES THIS, deliberately. The flag
# is read at BUILD time, so leaving it on keeps the two runs' `:oop` build
# products identical. It cannot reach the emitted program here: the SSA spike's
# arms live inside `_oop_eval_acck` / `_oop_run_acc_vec` / `_oop_fill_level`,
# which are the functions a TRACE walks, and the direct emitter walks the plan
# data itself and never calls them. `ext/reactant_direct/` contains no reference
# to the SSA tables at all. The direct lane is SSA by construction -- a consumer
# references its producer's value because there is no flat buffer to go through
# -- which is the thing the spike was reaching for from the other side.
#
# WHICH MAKES THIS RUN'S STEP COST THE DATUM THAT DECIDES THE SPIKE'S FUTURE.
# If the direct lane's loop matches or beats the traced-plus-SSA record, then
# once the traced path is retired the spike has nothing left to accelerate.
#
# ENVIRONMENT. Not `run-model-jl` -- that project develops the released
# packages, and this needs three unreleased things at once: EarthSciAST
# `oop-retire` (the emitter), and the `faq`-node-tag fixes in EarthSciASTSplitter
# and EarthSciASTDiff without which the operator split copies the whole tendency
# into both halves and the symbolic Jacobian comes back empty. See README.md.
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
# The direct lane's compile cost at CONUS has never been measured; 24 h is ~19x
# the traced record's 1 h 15 m, which is margin for a surprise rather than a
# projection.
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/oopretire-logs/port/adj48-direct-%j.log

set -u
# Slurm copies the script out of the repository, so the path is spelled out.
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact-directrhs || exit 1

export JULIA_DEPOT_PATH=/scratch/$USER/oopretire-depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia
export RESEACT_RXENV=${RESEACT_RXENV:-/scratch/$USER/oopretire-env-faq2}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_RHS=${RESEACT_RHS:-direct}

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
export ESS_OOP_SSA=${ESS_OOP_SSA:-1}
export RESEACT_LABEL=${RESEACT_LABEL:-conus-adj48-$RESEACT_RHS-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-/scratch/$USER/oopretire-logs/port/adj48-$RESEACT_RHS-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
echo "=== RHS=$RESEACT_RHS env=$RESEACT_RXENV jac=$RESEACT_ADJ_JAC clamp=$RESEACT_ADJ_CLAMP" \
     "ujitter=$RESEACT_ADJ_UJITTER nmacro=$RESEACT_ADJ_NMACRO shards=$RESEACT_ADJ_SHARDS ==="
free -g | head -2
echo "started $(date -Is)"

# -t 8 for the driver; the 8 shard workers are separate processes and take the
# rest of the node.
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G run_reseact_adjoint.jl

echo "exit=$? finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
