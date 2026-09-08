# ===========================================================================
# shard_worker.jl -- the Distributed WORKER end of the ProcessExecutor.
# ===========================================================================
# Loaded on every Distributed worker by tools/shard_exec.jl (never on the
# driver). This file is now only the process-shaped part: activate the project,
# load the repo's support modules into this process's `Main`, load
# tools/shard_kernel.jl -- which is the actual per-shard chemistry program, and
# knows nothing about processes -- and expose it under the five names the
# driver's `_rpc` resolves in `Main`.
#
# One process holds exactly ONE shard, so a single `ShardState` in a `Ref` is
# the whole of this file's state. The kernel takes its state explicitly, so
# nothing here is load-bearing for correctness: an executor that runs several
# shards in one process (tools/shard_exec.jl's `InProcessExecutor`) holds a
# Vector of the same states and calls the same functions.
#
# WHY A PROCESS, HISTORICALLY. Measured on THIS CPU (tools/diag/exec_overlap.jl,
# host_device_overlap.jl, steptime_shard.jl): one XLA:CPU execution keeps ~3 of
# 16 cores busy, N concurrent executions in ONE process do not overlap (1.34x
# for 8), `xla_force_host_platform_device_count` changes nothing, and two
# half-domain PROCESSES advance the domain 2.66x faster than one. Every one of
# those premises is CPU-specific, which is exactly why the mechanism now sits
# behind an interface -- see tools/shard_exec.jl.
# ===========================================================================
import Pkg
const REPO = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Printf, Statistics, Logging
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using Reactant
using EarthSciASTDiff
const EA = EarthSciAST
const RX = Reactant
const EZ = Reactant.Enzyme
# The backend is env-gated (default cpu, i.e. today's behaviour) so a device run
# does not need a source edit here or in the driver.
try; RX.set_default_backend(get(ENV, "RESEACT_BACKEND", "cpu")); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(RXDIR, "rx_native_patch.jl"))
include(joinpath(RXDIR, "rx_traced_integrator.jl"))
const RTI = RxTracedIntegrator
include(joinpath(RXDIR, "rx_sym_block_jac.jl"))
using .RxSymBlockJac
include(joinpath(REPO, "tools", "capacity_chem.jl")); using .CapacityChem

# ShardKernel reads CapacityChem / RxTracedIntegrator / RxSymBlockJac and the
# split_common top-levels off its parent module, which is this `Main`.
include(joinpath(REPO, "tools", "shard_kernel.jl"))

const S = Ref{Any}(nothing)

# --- the five RPC names the driver resolves in `Main` -----------------------
function shard_prepare_doc!(cfg::Dict)
    st, info = ShardKernel.prepare_doc!(cfg)
    S[] = st
    return info
end

shard_build!(cfg::Dict, ca::Dict{String,Any}, pshapes::Dict{String,Any},
             lanes0::Dict{String,Any}) = ShardKernel.build!(S[], cfg, ca, pshapes, lanes0)

shard_refresh!(lanes::Dict{String,Any}) = ShardKernel.refresh!(S[], lanes)

shard_step(uc::Vector{Float64}, t::Float64, dt::Float64) =
    ShardKernel.chem_step(S[], uc, t, dt)

shard_vjp(uc::Vector{Float64}, lamc::Vector{Float64}, t::Float64, dt::Float64) =
    ShardKernel.chem_vjp(S[], uc, lamc, t, dt)

shard_pid() = getpid()
shard_rss() = ShardKernel.rss_gb()
