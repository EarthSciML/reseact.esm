# ===========================================================================
# shard_exec.jl -- the chemistry-shard EXECUTOR seam.
# ===========================================================================
# tools/shard_chem.jl owns the DECOMPOSITION: which cells belong to which
# shard, the capacity documents, the lane buffers, the gather/scatter between
# the runner's species-major full vector and a shard's sub-vector, the
# per-shard partial-norm reduction, and the map from a shard's parameter keys
# onto the driver's PNAMES. This file owns the other half: HOW the shards are
# actually run. The two are separate on purpose.
#
# WHY. The decomposition is device-shaped already -- a flat C x 1 x 1 lane grid
# with every piece of geometry and forcing delivered through per-lane buffers,
# so the chemistry kernel performs no global gather. The eight-process fan-out
# is NOT: it exists because of measurements that are true of THIS CPU and of
# nothing else (one XLA:CPU execution keeps ~3 of 16 cores busy; N concurrent
# executions in one process reach only 1.34x for N=8;
# `xla_force_host_platform_device_count` changes nothing; two half-domain
# PROCESSES advance the domain 2.66x faster than one). On an A100/H100 every
# one of those premises is false: kernels overlap, and the right shape is one
# process per device with the arrays sharded across devices -- not eight OS
# processes contending for one card. So the mechanism sits behind an interface
# and the decomposition does not have to know which one it got.
#
# ---------------------------------------------------------------------------
# THE EXECUTOR CONTRACT
# ---------------------------------------------------------------------------
# An executor manages N shard SLOTS, numbered 1..N. It must implement:
#
#   exec_name(ex)                     -> String, for the log
#   exec_start!(ex, n; say)           acquire n slots. May fail if it cannot
#                                     supply n (a device executor with fewer
#                                     devices than n should say so, loudly).
#   exec_map(ex, f, ks)               run `f(k)` for every k in `ks` and return
#                                     the results IN THE ORDER OF `ks`. This is
#                                     the fan-out primitive and the only place
#                                     an executor expresses concurrency.
#   exec_prepare_doc!(ex, k, cfg)     -> info   (ShardKernel.prepare_doc!)
#   exec_build!(ex, k, cfg, ca,
#               pshapes, lanes)       -> binfo  (ShardKernel.build!)
#   exec_refresh!(ex, k, lanes)       -> nothing (ShardKernel.refresh!)
#   exec_step(ex, k, uc, t, dt)       -> (unew, sse, seconds)
#   exec_vjp(ex, k, uc, lamc, t, dt)  -> (lam_in, dJdp over pkeys, seconds)
#   exec_footprint(ex)                -> String, one line of memory accounting
#   exec_shutdown!(ex)                release the slots
#
# INVARIANTS an implementation must uphold. These are not style; the numbers in
# tools/shard_chem.jl's header depend on every one of them.
#
#  1. LAYOUT AT THE BOUNDARY. `uc` / `lamc` arrive species-major, NS runs of
#     C_k cells, in the runner's species order and the decomposition's cell
#     order, and `unew` / `lam_in` must come back in exactly that layout. The
#     permutation into whatever the shard's own build wants is the executor
#     side's business (ShardKernel does it through `capsel`). The driver never
#     sees a capacity-layout vector.
#
#  2. THE PARTIAL NORM. `exec_step` returns `sse_k`, the SUM over shard k's
#     states of the squared scaled residual, NOT a norm. The decomposition
#     rebuilds the global controller norm as `sqrt(sum_k sse_k / N)`, adding
#     the partials IN SHARD ORDER. An executor must not normalise, must not
#     pre-reduce across shards in a nondeterministic order, and must not
#     change the shard order of the sum -- the accept/reject ladder is
#     reproducible only because that order is fixed. (This reassociates the
#     reduction relative to an unsharded run, which is why the sharded gate is
#     a TOLERANCE gate at the 2.9e-14 class and not a bit-identity gate.)
#
#  3. THE LANE GATHER/SCATTER. Forcing and geometry reach a shard ONLY through
#     `exec_refresh!`, as a Dict of dense per-lane arrays the decomposition
#     gathered on the host (CapacityChem.gather_forcing! / gather_geometry!).
#     A shard performs no global indexing of its own. `exec_refresh!` is called
#     once per GEOS-FP epoch, for every slot, before any step at that epoch;
#     an executor may defer the host->device copy inside that call but must not
#     defer it past the next `exec_step`/`exec_vjp`.
#
#  4. dJ/dp. `exec_vjp` returns the shard's gradient over ITS OWN parameter
#     keys, in the order `binfo.pkeys` gave at build. The decomposition sums
#     them into the driver's PNAMES, in shard order, and has already asserted
#     at setup that every shard key is one of the driver's and holds the
#     driver's value. A parameter the capacity document pruned contributes
#     exactly zero and must not appear.
#
#  5. TIMING. The third element of `exec_step`/`exec_vjp` is the seconds THE
#     SHARD ITSELF spent (upload + execute + read back), measured on the
#     executor side. The decomposition reports max over shards (the critical
#     path) and the sum (what one stream would pay), and subtracts the max from
#     its own fan-out wall to get "host round-trip + imbalance". An executor
#     that returns 0.0 here still runs correctly, but that diagnostic dies.
#
#  6. NO DRIVER STATE. An executor may not read the driver's `merged_param`,
#     `p`, `UBASE` or any other driver global. Everything it needs arrives in
#     `cfg` / `ca` / `pshapes` / `lanes`. This is what makes a remote or
#     device slot possible at all.
#
# ---------------------------------------------------------------------------
# WHAT A DEVICE (GPU) EXECUTOR WOULD IMPLEMENT -- UNEXERCISED
# ---------------------------------------------------------------------------
# There is no GPU on this cluster that may be used, so no device executor is
# written here and NOTHING below has been run against one. What such an
# executor has to do, stated so that the next person writing it does not have
# to re-derive it:
#
#   * `exec_start!` sets `Reactant.set_default_backend("gpu")` (the drivers now
#     read RESEACT_BACKEND for that) and checks `length(Reactant.devices())`
#     against n. The natural n is the DEVICE count, not 8; the decomposition
#     takes n from the caller and imposes nothing.
#   * `exec_prepare_doc!` / `exec_build!` can call ShardKernel unchanged: the
#     capacity document, the symbolic Jacobian, the gather plan and the ROS23
#     step/VJP are all backend-agnostic Reactant code. The only device-specific
#     part is WHERE the `ConcreteRArray`s live, which is a `Sharding.Mesh` /
#     `NamedSharding` argument on the `ConcreteRArray` constructors inside
#     `ShardKernel.build!` -- the one place that would need a keyword.
#   * `exec_map` is where the shape actually differs: on a GPU the right answer
#     is very likely N = 1 slot holding the WHOLE domain with the lane axis
#     sharded across devices by `NamedSharding`, so `exec_map` is a plain call
#     and the overlap happens inside XLA. The decomposition already supports
#     that: N = 1 is a legal shard count and costs it nothing.
#   * `exec_refresh!` becomes a host->device copy of the lane Dict; invariant 3
#     is what makes that a bulk `copyto!` and not a scatter.
#   * `exec_footprint` should report device memory, not RSS.
#
# The seam is what is being delivered here, not a device backend. The
# InProcessExecutor below exists to prove the seam is real -- it runs the same
# shards, through the same decomposition, with no Distributed anywhere.
# ===========================================================================

abstract type ShardExecutor end

# generic fallbacks
exec_footprint(::ShardExecutor) = "footprint: not reported by this executor"
exec_shutdown!(::ShardExecutor) = nothing

# The primary method dispatches on the executor (`exec_map(ex, f, ks)`); this
# argument-swapped one exists so call sites can use do-block syntax, which is
# how the two multi-line fan-outs in tools/shard_chem.jl are written.
exec_map(f::Function, ex::ShardExecutor, ks) = exec_map(ex, f, ks)

# ---------------------------------------------------------------------------
# 1. ProcessExecutor -- one Distributed.jl worker process per shard.
# ---------------------------------------------------------------------------
# THE DEFAULT, and unchanged in behaviour from the code that lived inline in
# shard_chem.jl before 2026-09-08: the same `addprocs` flags, the same
# `_RPC_APPLY` trick, the same `asyncmap` fan-out, the same taskset pinning.
#
#   RESEACT_SHARD_THREADS       julia threads per worker (default 2)
#   RESEACT_SHARD_HEAP          --heap-size-hint per worker (default 8G)
#   RESEACT_SHARD_PIN           1 = pin each worker to its own CPU range
# ---------------------------------------------------------------------------
using Distributed

const SHARD_THREADS = parse(Int, get(ENV, "RESEACT_SHARD_THREADS", "2"))
const SHARD_HEAP    = get(ENV, "RESEACT_SHARD_HEAP", "8G")
const SHARD_PIN     = get(ENV, "RESEACT_SHARD_PIN", "0") == "1"

# Call a WORKER-side function by NAME. Both sides define `shard_step` /
# `shard_vjp` (different signatures), so a driver function object must not be
# shipped; the applicator below is serialized with its code and resolves the
# name in the worker's own Main.
#
# IT IS BUILT IN `Main` ON PURPOSE, and that is not a style choice. Serializing
# an anonymous function ships its code plus its DEFINING MODULE, and the worker
# resolves that module by name before it can reconstruct the function. This file
# is `include`d into whatever scope the driver runs in -- which is `Main` when
# tools/adjoint_gradient.jl is run directly, and the module `_AdjointArm` when
# run_reseact_adjoint.jl runs it (it wraps each arm in its own module so the two
# drivers' top-level constants cannot collide). A closure defined there dies on
# the worker with
#     UndefVarError: `_AdjointArm` not defined in `Main`
# during deserialization, at the FIRST rpc -- i.e. the sharded path worked only
# via the direct driver, and the repo's own documented five-day command
# (adjoint_conus_5d.sbatch -> run_reseact_adjoint.jl with RESEACT_ADJ_SHARDS=8)
# could not spawn a shard. `Core.eval(Main, ...)` gives the applicator a module
# every worker has by construction, so both entry points work.
const _RPC_APPLY = Core.eval(Main, :((f, a...) -> getfield(Main, f)(a...)))
_rpc(pid::Int, fname::Symbol, args...) =
    remotecall_fetch(_RPC_APPLY, pid, fname, args...)

mutable struct ProcessExecutor <: ShardExecutor
    pids::Vector{Int}
    ProcessExecutor() = new(Int[])
end

exec_name(::ProcessExecutor) = "process"

function _allowed_cpus()
    for line in eachline("/proc/self/status")
        startswith(line, "Cpus_allowed_list:") || continue
        out = Int[]
        for part in split(strip(split(line, ':')[2]), ',')
            if occursin('-', part)
                a, b = parse.(Int, split(part, '-')); append!(out, a:b)
            else
                push!(out, parse(Int, part))
            end
        end
        return out
    end
    return collect(0:(Sys.CPU_THREADS - 1))
end

function exec_start!(ex::ProcessExecutor, n::Int; say = println)
    say("  executor: $n Distributed worker processes ($SHARD_THREADS threads, heap $SHARD_HEAP each)")
    exeflags = ["--project=$(Base.active_project())", "-t", string(SHARD_THREADS),
                "--heap-size-hint=$SHARD_HEAP"]
    pids = addprocs(n; exeflags = exeflags, dir = REPO)
    length(pids) == n || error("ProcessExecutor: asked for $n workers, got $(length(pids))")
    ex.pids = pids
    if SHARD_PIN
        # each worker on its own CPU slice of the job's allowed set
        allowed = _allowed_cpus()
        per = max(1, length(allowed) ÷ n)
        for (k, pid) in enumerate(pids)
            cpus = allowed[((k - 1) * per + 1):min(k * per, length(allowed))]
            wp = remotecall_fetch(getpid, pid)
            run(pipeline(`taskset -pc $(join(cpus, ',')) $wp`; stdout = devnull))
        end
        say("  workers pinned: $per cpus each of $(length(allowed)) allowed")
    end
    wsrc = joinpath(REPO, "tools", "shard_worker.jl")
    Distributed.remotecall_eval(Main, pids, :(include($wsrc)))
    return nothing
end

# separate PROCESSES overlap nearly free on XLA:CPU (2.66x for two half-domain
# processes); `asyncmap` is what turns the blocking `remotecall_fetch`es into
# that overlap.
exec_map(ex::ProcessExecutor, f, ks) = asyncmap(f, ks)

exec_prepare_doc!(ex::ProcessExecutor, k::Int, cfg::Dict) =
    _rpc(ex.pids[k], :shard_prepare_doc!, cfg)
exec_build!(ex::ProcessExecutor, k::Int, cfg::Dict, ca::Dict{String,Any},
            pshapes::Dict{String,Any}, lanes::Dict{String,Any}) =
    _rpc(ex.pids[k], :shard_build!, cfg, ca, pshapes, lanes)
exec_refresh!(ex::ProcessExecutor, k::Int, lanes::Dict{String,Any}) =
    _rpc(ex.pids[k], :shard_refresh!, lanes)
exec_step(ex::ProcessExecutor, k::Int, uc::Vector{Float64}, t::Float64, dt::Float64) =
    _rpc(ex.pids[k], :shard_step, uc, t, dt)
exec_vjp(ex::ProcessExecutor, k::Int, uc::Vector{Float64}, lamc::Vector{Float64},
         t::Float64, dt::Float64) = _rpc(ex.pids[k], :shard_vjp, uc, lamc, t, dt)

function exec_footprint(ex::ProcessExecutor)
    rss = [_rpc(pid, :shard_rss) for pid in ex.pids]
    @sprintf("worker RSS: max %.1f GB, total %.1f GB", maximum(rss), sum(rss))
end

exec_shutdown!(ex::ProcessExecutor) = (isempty(ex.pids) || rmprocs(ex.pids); ex.pids = Int[]; nothing)

# ---------------------------------------------------------------------------
# 2. InProcessExecutor -- every shard in the DRIVER's process, sequentially.
# ---------------------------------------------------------------------------
# The seam's proof of life, and nothing more. It runs the identical
# `ShardKernel` programs on the identical decomposition with no Distributed
# anywhere, so a disagreement between it and the ProcessExecutor is a
# decomposition bug and cannot be anything else.
#
# It is SLOWER than the process path on this CPU and is expected to be: the
# measurements at the top of this file say concurrent XLA:CPU executions in one
# process do not overlap, so a sequential in-process fan-out pays the sum of
# the shard device times where the process fan-out pays the max. That is the
# point -- the seam is what changes, not the arithmetic. It is also useful
# where the process path cannot go: a 40 GiB cgroup will not hold eight 5.5 GB
# workers, and this path holds one copy of everything.
#
# Its `exec_map` is the honest one-line statement of the difference. A device
# executor replaces exactly that method plus the `ConcreteRArray` placement.
# ---------------------------------------------------------------------------
const SHARD_EXEC_KIND = lowercase(get(ENV, "RESEACT_SHARD_EXEC", "process"))
SHARD_EXEC_KIND in ("process", "inprocess") ||
    error("RESEACT_SHARD_EXEC must be `process` (default) or `inprocess`, got `$SHARD_EXEC_KIND`")

# Only loaded when asked for: ShardKernel pulls in EarthSciASTDiff, and the
# process path has no reason to pay that on the driver.
if SHARD_EXEC_KIND == "inprocess"
    include(joinpath(REPO, "tools", "shard_kernel.jl"))
end

mutable struct InProcessExecutor <: ShardExecutor
    states::Vector{Any}          # ShardKernel.ShardState, one per slot
    InProcessExecutor() = new(Any[])
end

exec_name(::InProcessExecutor) = "inprocess"

function exec_start!(ex::InProcessExecutor, n::Int; say = println)
    say("  executor: $n shards IN THIS PROCESS, run sequentially (no Distributed)")
    ex.states = Vector{Any}(nothing, n)
    return nothing
end

# The whole difference between the two executors, in one method: no overlap is
# attempted, because on XLA:CPU there is none to be had inside one process.
exec_map(ex::InProcessExecutor, f, ks) = map(f, ks)

function exec_prepare_doc!(ex::InProcessExecutor, k::Int, cfg::Dict)
    st, info = ShardKernel.prepare_doc!(cfg)
    ex.states[k] = st
    return info
end
exec_build!(ex::InProcessExecutor, k::Int, cfg::Dict, ca::Dict{String,Any},
            pshapes::Dict{String,Any}, lanes::Dict{String,Any}) =
    ShardKernel.build!(ex.states[k], cfg, ca, pshapes, lanes)
exec_refresh!(ex::InProcessExecutor, k::Int, lanes::Dict{String,Any}) =
    ShardKernel.refresh!(ex.states[k], lanes)
exec_step(ex::InProcessExecutor, k::Int, uc::Vector{Float64}, t::Float64, dt::Float64) =
    ShardKernel.chem_step(ex.states[k], uc, t, dt)
exec_vjp(ex::InProcessExecutor, k::Int, uc::Vector{Float64}, lamc::Vector{Float64},
         t::Float64, dt::Float64) = ShardKernel.chem_vjp(ex.states[k], uc, lamc, t, dt)

exec_footprint(ex::InProcessExecutor) =
    @sprintf("one process, RSS %.1f GB holding all %d shards", ShardKernel.rss_gb(),
             length(ex.states))

# ---------------------------------------------------------------------------
"""
    make_shard_executor(kind = RESEACT_SHARD_EXEC) -> ShardExecutor

`process` (default, unchanged behaviour) or `inprocess`. Adding a device
backend means one more branch here and one more implementation above -- and
nothing at all in tools/shard_chem.jl, tools/capacity_chem.jl or the driver.
"""
function make_shard_executor(kind::AbstractString = SHARD_EXEC_KIND)
    kind == "process"   && return ProcessExecutor()
    kind == "inprocess" && return InProcessExecutor()
    error("unknown shard executor `$kind`")
end
