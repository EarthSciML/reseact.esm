# ===========================================================================
# shard_worker.jl -- ONE SHARD of the chemistry half, in its OWN PROCESS.
# ===========================================================================
# Loaded on every Distributed worker by tools/shard_chem.jl (never on the
# driver). A worker owns a CAPACITY build of the chemistry half
# (tools/capacity_chem.jl) at C = the exact number of cells in its shard, so
# there is no padding lane anywhere, and it compiles the ROS23 step (with the
# per-cell error norm) and the ROS23 VJP on it. It then answers three RPCs:
#
#   shard_refresh!(arrays)         new forcing (one call per GEOS-FP epoch)
#   shard_step(uc, t, dt)          one step of its cells  -> (unew, sum of
#                                  squared scaled errors, device seconds)
#   shard_vjp(uc, lamc, t, dt)     one VJP of its cells   -> (lambda_in, dJ/dp
#                                  over the worker's parameter keys, seconds)
#
# WHY A PROCESS AND NOT A THREAD OR A DEVICE. Measured (tools/diag/exec_overlap.jl,
# host_device_overlap.jl, steptime_shard.jl): one XLA:CPU execution keeps ~3 of
# 16 cores busy, N concurrent executions in ONE process do not overlap (1.34x
# for 8), `xla_force_host_platform_device_count` changes nothing, and two
# half-domain PROCESSES advance the domain 2.66x faster than one. Chemistry is
# exactly block-diagonal per cell, so the arithmetic of a cell does not depend
# on which process runs it; only the accept/reject norm reassociates, and the
# driver rebuilds that from per-shard partial sums in a fixed order.
#
# THE STATE LAYOUT AT THE BOUNDARY. The driver sends the shard's cells in ITS
# layout -- species-major, `NS` runs of `C` cells, the runner's species order --
# and the worker permutes into its own build's layout through `capsel` (the
# same `(s-1)*C + l -> capacity index` map tools/subcycle_chem.jl uses), and
# back on the way out. Same for lambda. dJ/dp comes back as a Vector ordered by
# `pkeys`, the worker's own parameter names, which the driver maps onto its
# PNAMES once at setup; a parameter the capacity document pruned is one the
# chemistry half does not depend on, and contributes exactly zero.
#
# `param_coords = true` for the capacity document: the coordinate arrays stay
# expressions over lat0_deg/lon0_deg/dlat_deg/dlon_deg, so the photolysis path
# of dJ/d(Transport3D.lat0_deg) and dJ/d(lon0_deg) survives (see
# capacity_chem.jl). That is the one difference from the forward-only subcycle
# and bucket builds.
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
try; RX.set_default_backend("cpu"); catch; end

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

const WID = Ref(0)
wsay(s) = (println("  [shard $(WID[])] " * s); flush(stdout))

# Everything the RPCs need, filled by `shard_build!`.
mutable struct ShardState
    C::Int
    NS::Int
    f                       # the capacity RHS (oop evaluator)
    pa::Dict{String,Any}    # its param_arrays (lane buffers), bound into the build
    meta
    u0c::Vector{Float64}
    pc                      # the capacity build's parameter NamedTuple
    pkeys::Vector{Symbol}
    vmc::Dict{String,Int}
    capsel::Vector{Int}
    masks
    jacE
    plan
    gjb
    dev_bufs
    dev_bufsJ
    th
    cstep
    cvjp
    docCAP0
    cd
    ShardState(C::Int, NS::Int) = new(C, NS, nothing, Dict{String,Any}(), nothing, Float64[],
                                      nothing, Symbol[], Dict{String,Int}(), Int[], nothing,
                                      nothing, nothing, nothing, nothing, nothing, nothing,
                                      nothing, nothing, nothing, nothing)
end
const S = Ref{Any}(nothing)

_devp(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))

"""
    shard_prepare_doc!(cfg) -> (variables, lane_arrays, emis_arrays, lonc)

Phase 1 of setup: load the model at the capacity metaparameters, split it, and
perform the capacity surgery at this shard's C. Returns the names the capacity
document still reads, so the driver can send exactly the constant arrays and
forcing shapes the build needs (the NEI inventory, 137,241 cells x 69 species,
is pruned away and never crosses the socket).
"""
function shard_prepare_doc!(cfg::Dict)
    WID[] = cfg["wid"]
    st = ShardState(cfg["C"], cfg["NS"])
    tl = time()
    st.docCAP0 = Logging.with_logger(Logging.NullLogger()) do
        file = EA.load_path(cfg["model"]; metaparameters = cfg["capmp"])
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
        index_promoted_refs_by_loop!(EA.flattened_to_esm(parts[2]), promoted)
    end
    st.cd, st.meta = CapacityChem.capacity_doc(st.docCAP0, st.C; say = wsay,
                                               param_coords = true)
    S[] = st
    wsay(@sprintf("capacity document ready at C=%d (%.1f s)", st.C, time() - tl))
    return (variables = collect(st.meta.variables), lane_arrays = st.meta.lane_arrays,
            emis_arrays = st.meta.emis_arrays, lonc = st.meta.lonc)
end

"""
    shard_build!(cfg, ca, pshapes, lanes0) -> info

Phase 2: build the capacity evaluator with the constant arrays `ca` and forcing
buffers shaped like `pshapes`, prime the lanes with `lanes0` (the driver's first
gather), prepare the symbolic Jacobian and its gather plan, and compile the
step and the VJP. `info` carries the parameter values and keys, `capsel`, and
the one-off timings.
"""
function shard_build!(cfg::Dict, ca::Dict{String,Any}, pshapes::Dict{String,Any},
                      lanes0::Dict{String,Any})
    st = S[]
    C = st.C; NS = st.NS
    st.pa = CapacityChem.lane_buffers(st.meta, pshapes, C)
    _copy_lanes!(st.pa, lanes0)
    ov = Dict{String,Float64}(k => v for (k, v) in cfg["ov"] if k in st.meta.variables)
    tb = time()
    f, u0c, pc, _, vmc = Logging.with_logger(Logging.NullLogger()) do
        EA.build_evaluator(st.cd; form = :oop, parameter_overrides = ov,
                           const_arrays = ca, param_arrays = st.pa)
    end
    tbuild = time() - tb
    length(u0c) == NS * C ||
        error("shard_build!: capacity build at C=$C has $(length(u0c)) states, expected NS*C = $(NS*C)")
    st.f = f; st.u0c = copy(u0c); st.pc = pc; st.vmc = vmc
    st.pkeys = collect(keys(pc))
    st.masks = RTI.species_masks(vmc, NS, C)
    spnames = cfg["spnames"]::Vector{String}
    st.capsel = Vector{Int}(undef, NS * C)
    for s in 1:NS, l in 1:C
        nm = @sprintf("%s[%d,1,1]", spnames[s], l)
        haskey(vmc, nm) || error("shard_build!: the C=$C build has no state `$nm`")
        st.capsel[(s - 1) * C + l] = vmc[nm]
    end
    # the primed base point must evaluate finite, or the lane gather is not
    # reaching this build (a zero lane NaNs through log(PS/Pc))
    let du = EA.rhs_with_buffers(f)(u0c, pc, cfg["T0"], EA.forcing_buffers(f))
        nb = count(!isfinite, du)
        nb == 0 || error("shard_build!: the C=$C RHS returns $nb of $(length(du)) NON-FINITE derivatives at the primed base point")
    end

    tj = time()
    st.jacE = Logging.with_logger(Logging.NullLogger()) do
        EarthSciASTDiff.prepare_jacobian(EA.coerce_esm_file(st.cd); model_name = "Flattened",
            wrt = :states, build_kwargs = (; form = :oop, parameter_overrides = ov,
                                             const_arrays = ca, param_arrays = st.pa))
    end
    tjac = time() - tj
    st.jacE.oop || error("shard_build!: the C=$C band model came back IN-PLACE")
    String(st.jacE.structure) == "block_diagonal" ||
        error("shard_build!: the C=$C Jacobian is $(st.jacE.structure), not block_diagonal")
    st.plan = RxSymBlockJac.block_jac_plan(st.jacE;
                  runner_names = first.(sort(collect(vmc), by = last)))
    st.gjb = EA.rhs_with_buffers(st.jacE.fJ!)
    let w = validate_plan(st.plan, st.jacE, u0c, pc, cfg["T0"];
                          gjb = st.gjb, bufs = EA.forcing_buffers(st.jacE.fJ!))
        w <= 1e-12 || error("shard_build!: the C=$C gather plan does not reproduce the host Jacobian (worst relative $w)")
    end
    st.dev_bufs  = map(RX.ConcreteRArray, EA.forcing_buffers(f))
    st.dev_bufsJ = map(RX.ConcreteRArray, EA.forcing_buffers(st.jacE.fJ!))
    st.th = (p = _devp(pc), bufs = st.dev_bufs, bufsJ = st.dev_bufsJ)

    frhs = EA.rhs_with_buffers(f)
    plan = st.plan; gjb = st.gjb; masks = st.masks
    atol = cfg["ATOL_C"]::Float64; rtol = cfg["RTOL"]::Float64; jacmode = cfg["JACMODE"]::Symbol
    gC(u, th, t) = frhs(u, th.p, t, th.bufs)
    gJ(u, th, t) = RxSymBlockJac.block_jac(plan,
                       gjb(RxSymBlockJac.gather_uj(plan, u), th.p, t, th.bufsJ))
    stepfn(u, th, t, dt) = RTI.ros23_step((uu, tt) -> gC(uu, th, tt), u, t, dt,
                                          NS, C, masks, atol, rtol;
                                          unrolled = true, jac = jacmode,
                                          symjac = (uu, tt) -> gJ(uu, th, tt),
                                          cellwise = true)
    vjpfn(u, th, lam, t, dt) = RTI.ros23_step_vjp(gC, u, th, t, dt, lam, NS, C, masks,
                                                  atol, rtol; jac = jacmode, gj = gJ,
                                                  active_bufs = false)
    copts = _copts(cfg)
    UD = RX.ConcreteRArray(copy(u0c)); LD = RX.ConcreteRArray(zeros(NS * C))
    TD = RX.ConcreteRNumber(cfg["T0"]); DD = RX.ConcreteRNumber(cfg["DT0C"])
    tc = time()
    st.cstep = RX.@compile compile_options=copts stepfn(UD, st.th, TD, DD)
    tcs = time() - tc
    tcv = 0.0
    if cfg["want_vjp"]
        tc = time()
        st.cvjp = RX.@compile compile_options=copts vjpfn(UD, st.th, LD, TD, DD)
        tcv = time() - tc
    end
    wsay(@sprintf("C=%d build %.1f s  jacobian %.1f s  compile step %.1f s  vjp %.1f s  (%d params)",
                  C, tbuild, tjac, tcs, tcv, length(st.pkeys)))
    return (pkeys = String.(st.pkeys),
            pvals = Float64[Float64(getfield(pc, k)) for k in st.pkeys],
            tbuild = tbuild, tjac = tjac, tcstep = tcs, tcvjp = tcv, rss = _rss_gb())
end

function _copts(cfg)
    excl = cfg["EXCLP"]::Vector{String}
    if cfg["XLAFIX"]
        RX.CompileOptions(; sync = true,
                          xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
                          (isempty(excl) ? (;) : (; excluded_passes = excl))...)
    else
        RX.CompileOptions(; sync = true, (isempty(excl) ? (;) : (; excluded_passes = excl))...)
    end
end

_rss_gb() = try
    parse(Float64, split(read("/proc/self/statm", String))[2]) * 4096 / 1024^3
catch; NaN end

function _copy_lanes!(pa::Dict{String,Any}, lanes::Dict{String,Any})
    for (k, v) in lanes
        haskey(pa, k) || error("shard: lane buffer `$k` is not in this build's param_arrays")
        size(pa[k]) == size(v) ||
            error("shard: lane buffer `$k` has shape $(size(pa[k])) here, $(size(v)) from the driver")
        copyto!(pa[k], v)
    end
    return nothing
end

"New forcing for this shard's lanes (one call per GEOS-FP epoch)."
function shard_refresh!(lanes::Dict{String,Any})
    st = S[]
    _copy_lanes!(st.pa, lanes)
    EA.sync_forcing!(st.dev_bufs, EA.forcing_buffers(st.f))
    EA.sync_forcing!(st.dev_bufsJ, EA.forcing_buffers(st.jacE.fJ!))
    return nothing
end

# runner-layout shard vector (NS runs of C) <-> capacity layout
function _to_cap(st::ShardState, v::Vector{Float64})
    out = Vector{Float64}(undef, length(v))
    @inbounds for i in eachindex(v); out[st.capsel[i]] = v[i]; end
    return out
end
function _from_cap(st::ShardState, vc::Vector{Float64})
    out = Vector{Float64}(undef, length(vc))
    @inbounds for i in eachindex(vc); out[i] = vc[st.capsel[i]]; end
    return out
end

"""
    shard_step(uc, t, dt) -> (unew, sse, seconds)

`sse` is the SUM over this shard's states of the squared scaled residual --
`NS * sum(cell_err.^2)`, i.e. the partial sum the global `EEst = sqrt(sse/N)`
reduces over -- so the driver can rebuild the global norm exactly up to
reassociation.
"""
function shard_step(uc::Vector{Float64}, t::Float64, dt::Float64)
    st = S[]
    td = time()
    res = st.cstep(RX.ConcreteRArray(_to_cap(st, uc)), st.th, RX.ConcreteRNumber(t),
                   RX.ConcreteRNumber(dt))
    raw = Array(res[1]); ce = Array(res[3])
    tdev = time() - td
    sse = 0.0
    @inbounds for e in ce; sse += e * e; end
    return _from_cap(st, raw), st.NS * sse, tdev
end

"""
    shard_vjp(uc, lamc, t, dt) -> (lambda_in, dJdp over pkeys, seconds)
"""
function shard_vjp(uc::Vector{Float64}, lamc::Vector{Float64}, t::Float64, dt::Float64)
    st = S[]
    td = time()
    r = st.cvjp(RX.ConcreteRArray(_to_cap(st, uc)), st.th, RX.ConcreteRArray(_to_cap(st, lamc)),
                RX.ConcreteRNumber(t), RX.ConcreteRNumber(dt))
    lin = Array(r[1]); gp = r[2].p
    tdev = time() - td
    return _from_cap(st, lin), Float64[Float64(getfield(gp, k)) for k in st.pkeys], tdev
end

shard_pid() = getpid()
shard_rss() = _rss_gb()
