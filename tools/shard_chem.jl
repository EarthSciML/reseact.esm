# ===========================================================================
# shard_chem.jl -- PROCESS-LEVEL SHARDING of the chemistry half (driver side).
# ===========================================================================
# `RESEACT_ADJ_SHARDS=N` splits the NC cells of the chemistry half into N
# contiguous equal-count shards, each owned by its OWN Julia process
# (Distributed.jl worker, tools/shard_worker.jl) holding a capacity build of the
# chemistry half at exactly its cell count. Every chemistry step of the forward
# pass, the fixed-sequence replay and every chemistry VJP of the backward sweep
# then fan out over the N processes and are gathered back here. The transport
# half is untouched: one full-domain program on this process, as before.
#
# WHY. Measured facts, not re-derived here (tools/diag/exec_overlap.jl,
# host_device_overlap.jl, steptime_shard.jl, the p5 VJP census):
#   * one XLA:CPU execution keeps ~3 of 16 cores busy, and ~9% parallel
#     efficiency is what the 8,920-op step gets from the thread pool, because
#     each op is one elementwise pass over NC doubles -- below the parallel-for
#     threshold;
#   * concurrent executions in ONE process do not overlap (1.34x for 8), and
#     `xla_force_host_platform_device_count` changes nothing;
#   * separate PROCESSES overlap nearly free: two half-domain processes advanced
#     the domain 2.66x faster than one, ~1.5x of it a working-set effect (the
#     per-cell cost FALLS as the shard shrinks: 57 -> 37 -> 30 us/cell at
#     6552 / 3500 / 1638 cells) and the rest a second execution stream.
#   Chemistry is exactly block-diagonal per cell, so which process runs a cell
#   cannot change its arithmetic. What DOES change: (a) the accept/reject norm
#   is rebuilt from per-shard partial sums, which reassociates the reduction;
#   (b) the per-cell program is the CAPACITY build (a C x 1 x 1 lane grid with
#   its geometry through lane buffers), verified to 2.9e-14 relative cell for
#   cell against the reference RHS, not bit-identical. So the sharded run is
#   the same integration at roundoff, and the gate is a tolerance gate, not a
#   byte comparison -- the driver reports both (the ladder signature and the
#   gradient) so the difference is visible rather than assumed.
#
# WHAT CROSSES THE SOCKET PER STEP: each shard's NS x C_k doubles out and back
# (85,176 doubles = 0.7 MB in total per CONUS step, split N ways), a scalar
# partial norm, and for a VJP the shard's dJ/dp (a few dozen doubles). The
# host round-trip is timed separately from the workers' own device time so its
# share is a measurement (`shard_report`).
#
# The gradient w.r.t. `p`: each worker's capacity document keeps every scalar
# parameter the chemistry half reads (`param_coords = true` keeps the lat/lon
# coordinate FORMULAS, so dJ/d(Transport3D.lat0_deg) and dJ/d(lon0_deg) keep
# their photolysis term -- see capacity_chem.jl); a parameter the document
# pruned is one the chemistry does not depend on and contributes zero. Worker
# parameter VALUES are asserted equal to the driver's `p` at setup.
#
# ---------------------------------------------------------------------------
# MEASURED (2026-09-05, all via sbatch on partition ctessum; the interactive
# cgroup stalls Julia builds in Lustre page faults and is not usable for this).
#
# GATE, 6x6x8, fwd,adj, 3 windows, un-jittered, clamp on (slurm 10366675/6/7 =
# N 0/2/4): the accept/reject ladder 3/0,116/3 2/1,92/2 3/0,30/0 is
# BYTE-IDENTICAL; J = 39.6832930037426 in all three (the CSV's 16th digit moves
# by one); 141/160 (N=2) and 142/160 (N=4) gradient components identical to
# every printed digit, all 19 nonzero components within 9.9e-13 relative, no
# component changes between zero and nonzero; structural identity 6.9e-16 /
# 1.7e-15 / 1.4e-15. At CONUS (below) the same holds at 3.9e-12.
#
# CONUS 13x7x72, 3 windows (T0 = 5400, spin-up), whole node (40 cpus, 160G),
# per window, from the driver's own DECOMPOSITION (ms/call in brackets):
#
#   N   fwd    C.step        bwd     C.replay      C.vjp         T.vjp    everything-else
#   0   3.19   2.39 [43.7]  17.83*  2.21 [41.7]   4.95 [93.4]   0.92 [347]   9.68* (refresh)
#   0   3.26   2.37 [43.4]  11.59   2.21 [41.8]   4.90 [92.5]   0.92 [345]   3.33  (JIT, see below)
#   4   2.14   1.18 [21.6]   9.90   1.09 [20.7]   2.36 [44.6]   0.92 [346]   5.46* (refresh)
#   8   1.94   0.95 [17.4]   5.72   0.82 [15.5]   1.64 [30.9]   0.93 [349]   2.20  (JIT)
#   13  1.85   0.95 [17.4]   9.14*  0.84         ~1.7          0.9        (refresh, pre-fix)
#   (* = before the epoch-cached refresh, commit 74bdc82; jobs 10371241-4,
#    10372581/2, 10372724/5, 10372885)
#   Chemistry at N=8: step 2.5x, replay 2.7x, VJP 3.0x. N=13 buys NOTHING over
#   N=8: the slowest shard's device time is 12 ms/call at 504 cells against
#   the ~3 ms per-call floor, and host round-trip + imbalance rises to 30%.
#
# TWO THINGS THE DECOMPOSITION FOUND THAT ARE NOT SHARDING:
#   * refresh_forcing cost 6.9 s/call at CONUS and the backward sweep called
#     it on EVERY macro step (576x over 48 h) where the forward pass calls it
#     at the 64 epoch boundaries -- the largest single term of the 48 h sweep.
#     Fixed (refresh only when the epoch changes; identical numbers).
#   * the residual "everything-else" of a 3-window run (10 s at N=0, 6.6 s at
#     N=8) is a CONSTANT, not per step: uploads 0.4 ms, lambda mask 0.3 ms,
#     gacc 0.2 ms, tape construction 1.2 ms per step, GC 0.14 s over the whole
#     sweep -- and over 576 windows it is 8.4 s total (0.01 s/window). It is
#     the first-call JIT of the sweep's own functions.
#   The transport VJP costs 331-349 ms/call AS THE DRIVER CALLS IT, i.e. the
#   standalone probe's number; the driver adds nothing to it.
#
# THE 48 h CONUS GRADIENT AT N=8 (slurm 10372969, 1 h 15 m all in, setup
# ~11 min: build 224 s, shards 674 s incl. spawn):
#   forward  692.6 s = 1.202 s/window   (was 1,425 s = 2.474 in 10359755)
#     C.step 0.69 [13.7 ms x 29,055]  T.step 0.07 [20.2]  refresh 0.43 [64 x 3.9 s]  GC 0.06
#   backward 2,106.7 s = 3.657 s/window (was 6,243 s = 10.84)
#     C.replay 0.64 [14.2]  C.vjp 1.36 [30.1 x 26,114]  T.replay 0.07  T.vjp 1.07 [331 x 1,859]
#     refresh 0.44 [64 calls]  GC 0.11 (1,193 pauses, 336 GB allocated)  everything-else 0.01
#   accept/reject ladder BYTE-IDENTICAL to 10359755; J = 30.1943301698531 (13
#   digits); all 19 nonzero gradient components within 1.3e-11; replay 0.0 at
#   every checkpoint; 0 flaky-reverse retries; structural identity 4.8e-15.
#   Shard fan-out: host round-trip + imbalance 10.4% (fwd) / 9.4% (bwd);
#   worker RSS 3.8-4.1 GB each, 29-31 GB total.
#   => 4.86 s/window all in; FIVE DAYS projects to ~1.9 h (compile excluded),
#   against ~5.3 h on the same accounting before this work. What remains per
#   window: chemistry 2.69 s (at N=8's per-call floor), transport VJP 1.07 s
#   (the emitter lever), refresh 0.87 s (cacheable: the forward pass already
#   sampled every epoch), replay 0.71 s (RESEACT_ADJ_KEEPTAPE=1 removes it:
#   ~19 GB for 48 h, ~48 GB for 5 days, which a 160G node with 31 GB of
#   workers can hold), transport steps 0.14 s.
# ---------------------------------------------------------------------------
#
# Plain include into the driver's scope (like subcycle_chem.jl), gated by
# RESEACT_ADJ_SHARDS > 0; with the knob off nothing here is loaded.
#
#   RESEACT_ADJ_SHARDS          N worker processes (0 = off)
#   RESEACT_SHARD_THREADS       julia threads per worker (default 2)
#   RESEACT_SHARD_HEAP          --heap-size-hint per worker (default 8G)
#   RESEACT_SHARD_PIN           1 = pin each worker to its own CPU range via
#                               taskset (default 0)
# ===========================================================================
using Distributed

const SHARD_THREADS = parse(Int, get(ENV, "RESEACT_SHARD_THREADS", "2"))
const SHARD_HEAP    = get(ENV, "RESEACT_SHARD_HEAP", "8G")
const SHARD_PIN     = get(ENV, "RESEACT_SHARD_PIN", "0") == "1"

# runner_layout / reference_geometry / SUB_CAPMP; the subcycle's own machinery
# is only ever built under RESEACT_SUBCYCLE=1, which is exclusive with sharding.
include(joinpath(REPO, "tools", "subcycle_chem.jl"))

mutable struct ShardStats
    calls::Int
    vjps::Int
    t_rpc::Float64      # wall time of the fan-out/gather, driver side
    t_dev::Float64      # max over shards of the worker's own device time, per call
    t_dev_sum::Float64  # sum over shards of device time (what N=1 would pay)
    t_pack::Float64     # driver-side sub-vector assembly
end
ShardStats() = ShardStats(0, 0, 0.0, 0.0, 0.0, 0.0)

# Call a WORKER-side function by NAME. Both sides define `shard_step` /
# `shard_vjp` (different signatures), so a driver function object must not be
# shipped; the closure below is serialized with its code and resolves the name
# in the worker's own Main.
_rpc(pid::Int, fname::Symbol, args...) =
    remotecall_fetch((f, a...) -> getfield(Main, f)(a...), pid, fname, args...)

mutable struct ShardSet
    pids::Vector{Int}
    ranges::Vector{UnitRange{Int}}     # runner cell positions per shard
    C::Vector{Int}
    meta::Vector{Any}
    pa::Vector{Dict{String,Any}}       # driver-side mirrors of each worker's lane buffers
    cells::Vector{NTuple{3,Int}}       # runner cell position -> reference (i, j, k)
    geom
    pmap::Vector{Vector{Int}}          # worker parameter index -> index into PNAMES
    stats::ShardStats
end

function _shard_ranges(nc::Int, n::Int)
    base, extra = divrem(nc, n)
    r = UnitRange{Int}[]; lo = 1
    for k in 1:n
        len = base + (k <= extra ? 1 : 0)
        push!(r, lo:(lo + len - 1)); lo += len
    end
    return r
end

function _gather_lanes!(S::ShardSet, k::Int)
    lane_cells = [S.cells[c] for c in S.ranges[k]]
    CapacityChem.gather_forcing!(S.pa[k], merged_param, lane_cells;
                                 lon0 = S.geom.lon0, lat0 = S.geom.lat0)
    CapacityChem.gather_geometry!(S.pa[k], S.meta[k], lane_cells; Ap = S.geom.ap, Bp = S.geom.bp,
                                  latp = S.geom.latp, lonp = S.geom.lonp, E = S.geom.E,
                                  lonc = S.geom.lonc, nlon = S.geom.nlon)
    return S.pa[k]
end

"""
    build_shards(n) -> ShardSet

Spawn `n` worker processes, build one capacity shard on each (concurrently),
prime their forcing from the driver's current `merged_param`, and check their
parameter vectors against `p`.
"""
function build_shards(n::Int)
    n >= 1 || error("build_shards: n must be >= 1")
    say("\n---- SHARDS: $n chemistry worker processes ($SHARD_THREADS threads, heap $SHARD_HEAP each) ----")
    tsp = time()
    exeflags = ["--project=$(Base.active_project())", "-t", string(SHARD_THREADS),
                "--heap-size-hint=$SHARD_HEAP"]
    pids = addprocs(n; exeflags = exeflags, dir = REPO)
    length(pids) == n || error("build_shards: asked for $n workers, got $(length(pids))")
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
    say(@sprintf("  spawned + loaded %d workers in %.1f s", n, time() - tsp))

    spnames, cells = runner_layout(var_map, NS, NC)
    geom = reference_geometry(BINSP, p, merged_const, GRID_MP)
    ranges = _shard_ranges(NC, n)
    Cs = length.(ranges)
    say("  shard sizes: " * join(string.(Cs), ", ") * " cells (no padding)")

    # phase 1: capacity documents (the load is grid-independent, ~40 s each, in parallel)
    cfg1 = [Dict{String,Any}("wid" => k, "C" => Cs[k], "NS" => NS, "model" => MODEL,
                             "capmp" => SUB_CAPMP) for k in 1:n]
    t1 = time()
    infos = asyncmap(k -> _rpc(pids[k], :shard_prepare_doc!, cfg1[k]), 1:n)
    say(@sprintf("  capacity documents ready on all shards (%.1f s)", time() - t1))

    # phase 2: builds + compiles, with exactly the arrays each document reads
    metas = Any[]; pas = Dict{String,Any}[]
    for k in 1:n
        vars = Set{String}(infos[k].variables)
        m = (; C = Cs[k], variables = vars, lane_arrays = infos[k].lane_arrays,
               emis_arrays = infos[k].emis_arrays, lonc = infos[k].lonc)
        push!(metas, m)
        push!(pas, CapacityChem.lane_buffers(m, merged_param, Cs[k]))
    end
    S0 = ShardSet(pids, ranges, Cs, metas, pas, cells, geom, Vector{Int}[], ShardStats())
    cfg2 = Dict{String,Any}("ov" => ov, "spnames" => spnames, "T0" => T0, "DT0C" => DT0C,
                            "ATOL_C" => ATOL_C, "RTOL" => RTOL, "JACMODE" => JACMODE,
                            "XLAFIX" => XLAFIX, "EXCLP" => EXCLP, "want_vjp" => want("adj"))
    t2 = time()
    binfo = asyncmap(1:n) do k
        vars = metas[k].variables
        ca = Dict{String,Any}(kk => v for (kk, v) in merged_const if kk in vars)
        pshapes = Dict{String,Any}(kk => size(v) for (kk, v) in merged_param
                                   if kk in vars && v isa AbstractArray)
        _gather_lanes!(S0, k)
        _rpc(pids[k], :shard_build!, cfg2, ca, pshapes, pas[k])
    end
    say(@sprintf("  shard builds + compiles done (%.1f s wall, concurrent)", time() - t2))
    for k in 1:n
        b = binfo[k]
        say(@sprintf("    shard %2d  C=%-5d build %6.1f s  jacobian %6.1f s  compile step %6.1f s  vjp %6.1f s  rss %.1f GB",
                     k, Cs[k], b.tbuild, b.tjac, b.tcstep, b.tcvjp, b.rss))
    end
    # parameter vectors: every worker key must be one of ours, at our value
    pmap = Vector{Vector{Int}}(undef, n)
    pidx = Dict{Symbol,Int}(kk => i for (i, kk) in enumerate(PNAMES))
    for k in 1:n
        b = binfo[k]
        m = Int[]
        for (j, ks) in enumerate(b.pkeys)
            sym = Symbol(ks)
            haskey(pidx, sym) || error("shard $k has a parameter `$ks` the driver's p does not")
            v0 = Float64(getfield(p, sym))
            b.pvals[j] == v0 || error("shard $k: parameter `$ks` = $(b.pvals[j]) differs from the driver's $v0")
            push!(m, pidx[sym])
        end
        pmap[k] = m
    end
    nsh = length(unique(vcat(pmap...)))
    say(@sprintf("  parameters: %d of the driver's %d scalars reach the chemistry shards (values checked)",
                 nsh, length(PNAMES)))
    S = ShardSet(pids, ranges, Cs, metas, pas, cells, geom, pmap, ShardStats())
    # WARM UP both RPC paths once, so the first-call JIT (the driver's fan-out
    # closures, the workers' execution wrappers) is paid here and not inside
    # the timed forward pass / backward sweep. Measured at 6x6x8: ~5 s of the
    # first window's wall was this. The results are discarded.
    tw = time()
    shard_step(S, copy(UBASE), T0, DT0C)
    want("adj") && shard_vjp(S, copy(UBASE), copy(WOBJ), T0, DT0C)
    S.stats = ShardStats()
    say(@sprintf("  warm-up step%s %.1f s", want("adj") ? " + VJP" : "", time() - tw))
    say(@sprintf("  SHARDS READY in %.1f s", time() - tsp))
    return S
end

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

"Push the driver's current forcing to every shard (one call per GEOS-FP epoch)."
function shard_refresh_all!(S::ShardSet)
    asyncmap(1:length(S.pids)) do k
        _gather_lanes!(S, k)
        _rpc(S.pids[k], :shard_refresh!, S.pa[k])
    end
    return nothing
end

# runner species-major full vector <-> shard sub-vector (NS runs of C_k)
function _pack(S::ShardSet, k::Int, u::Vector{Float64})
    r = S.ranges[k]; C = length(r); lo = first(r)
    out = Vector{Float64}(undef, NS * C)
    @inbounds for s in 1:NS
        bo = (s - 1) * C; bu = (s - 1) * NC + lo - 1
        for l in 1:C; out[bo + l] = u[bu + l]; end
    end
    return out
end
function _unpack!(u::Vector{Float64}, S::ShardSet, k::Int, uc::Vector{Float64})
    r = S.ranges[k]; C = length(r); lo = first(r)
    @inbounds for s in 1:NS
        bo = (s - 1) * C; bu = (s - 1) * NC + lo - 1
        for l in 1:C; u[bu + l] = uc[bo + l]; end
    end
    return u
end

"""
    shard_step(S, u, t, dt) -> (unew, EEst)

One chemistry step of the whole domain, fanned out over the shards. `EEst` is
`sqrt(sum_k sse_k / N)`, the per-shard partial sums added in shard order.
"""
function shard_step(S::ShardSet, u::Vector{Float64}, t::Float64, dt::Float64)
    st = S.stats; n = length(S.pids)
    tp = time()
    ucs = [_pack(S, k, u) for k in 1:n]
    st.t_pack += time() - tp
    tr = time()
    res = asyncmap(k -> _rpc(S.pids[k], :shard_step, ucs[k], t, dt), 1:n)
    st.t_rpc += time() - tr
    tp = time()
    unew = Vector{Float64}(undef, N)
    sse = 0.0; tdmax = 0.0; tdsum = 0.0
    for k in 1:n
        _unpack!(unew, S, k, res[k][1])
        sse += res[k][2]
        tdmax = max(tdmax, res[k][3]); tdsum += res[k][3]
    end
    st.t_pack += time() - tp
    st.calls += 1; st.t_dev += tdmax; st.t_dev_sum += tdsum
    return unew, sqrt(sse / N)
end

"""
    shard_vjp(S, u, lam, t, dt) -> (lambda_in, dJdp aligned with PNAMES)
"""
function shard_vjp(S::ShardSet, u::Vector{Float64}, lam::Vector{Float64}, t::Float64, dt::Float64)
    st = S.stats; n = length(S.pids)
    tp = time()
    ucs = [_pack(S, k, u) for k in 1:n]
    lcs = [_pack(S, k, lam) for k in 1:n]
    st.t_pack += time() - tp
    tr = time()
    res = asyncmap(k -> _rpc(S.pids[k], :shard_vjp, ucs[k], lcs[k], t, dt), 1:n)
    st.t_rpc += time() - tr
    tp = time()
    lin = Vector{Float64}(undef, N)
    g = zeros(Float64, length(PNAMES))
    tdmax = 0.0; tdsum = 0.0
    for k in 1:n
        _unpack!(lin, S, k, res[k][1])
        gp = res[k][2]; m = S.pmap[k]
        @inbounds for j in eachindex(m); g[m[j]] += gp[j]; end
        tdmax = max(tdmax, res[k][3]); tdsum += res[k][3]
    end
    st.t_pack += time() - tp
    st.vjps += 1; st.t_dev += tdmax; st.t_dev_sum += tdsum
    return lin, g
end

function shard_report(S::ShardSet)
    st = S.stats; nc = st.calls + st.vjps
    nc == 0 && return
    say(@sprintf("  shards: %d step calls + %d VJP calls over %d processes", st.calls, st.vjps, length(S.pids)))
    say(@sprintf("    fan-out wall %.2f s (%.2f ms/call); of which slowest-shard device %.2f s (%.2f ms/call) => host round-trip + imbalance %.2f s (%.1f%%)",
                 st.t_rpc, 1000 * st.t_rpc / nc, st.t_dev, 1000 * st.t_dev / nc,
                 st.t_rpc - st.t_dev, 100 * (st.t_rpc - st.t_dev) / max(st.t_rpc, eps())))
    say(@sprintf("    sum of shard device time %.2f s (%.2f ms/call, what one stream would pay for the same lanes); driver pack/unpack %.2f s",
                 st.t_dev_sum, 1000 * st.t_dev_sum / nc, st.t_pack))
    rss = [_rpc(pid, :shard_rss) for pid in S.pids]
    say(@sprintf("    worker RSS: max %.1f GB, total %.1f GB", maximum(rss), sum(rss)))
end
