# ===========================================================================
# shard_kernel.jl -- ONE SHARD of the chemistry half, as a plain object.
# ===========================================================================
# This is the per-shard chemistry PROGRAM, with no opinion whatsoever about
# where it runs. It owns a CAPACITY build of the chemistry half
# (tools/capacity_chem.jl) at C = the exact number of cells in its shard, so
# there is no padding lane anywhere, compiles the ROS23 step (with the per-cell
# error norm) and the ROS23 VJP on it, and answers five calls:
#
#   prepare_doc!(cfg)                 -> (st, info)   the capacity document
#   build!(st, cfg, ca, pshapes, l0)  -> binfo        build + Jacobian + compile
#   refresh!(st, lanes)                               new forcing for its lanes
#   chem_step(st, uc, t, dt)          -> (unew, sse, seconds)
#   chem_vjp(st, uc, lamc, t, dt)     -> (lam_in, dJdp over pkeys, seconds)
#
# Until 2026-09-08 this code lived inside tools/shard_worker.jl and could only
# be reached over a Distributed socket, which welded the DECOMPOSITION (which
# cells belong to which shard, the lane buffers, the gather/scatter, the
# partial-norm reduction -- tools/shard_chem.jl) to ONE fan-out MECHANISM
# (eight OS processes). The decomposition is already device-shaped: a flat
# C x 1 x 1 lane grid with all geometry and forcing delivered through per-lane
# buffers, so the kernel does no global gathers at all. The mechanism is not:
# on a GPU, kernels DO overlap, and the right shape is one process per device
# with the arrays sharded across devices, not eight OS processes contending for
# one card. Splitting this file out is what makes the mechanism replaceable --
# see tools/shard_exec.jl for the executor contract.
#
# THE STATE LAYOUT AT THE BOUNDARY. The caller passes the shard's cells in the
# DRIVER's layout -- species-major, `NS` runs of `C` cells, the runner's species
# order -- and the kernel permutes into its own build's layout through `capsel`
# (the same `(s-1)*C + l -> capacity index` map tools/subcycle_chem.jl uses),
# and back on the way out. Same for lambda. dJ/dp comes back as a Vector
# ordered by `pkeys`, the shard's own parameter names, which the driver maps
# onto its PNAMES once at setup; a parameter the capacity document pruned is one
# the chemistry half does not depend on, and contributes exactly zero.
#
# `param_coords = true` for the capacity document: the coordinate arrays stay
# expressions over lat0_deg/lon0_deg/dlat_deg/dlon_deg, so the photolysis path
# of dJ/d(Transport3D.lat0_deg) and dJ/d(lon0_deg) survives (see
# capacity_chem.jl). That is the one difference from the forward-only subcycle
# and bucket builds.
#
# HOW IT FINDS ITS DEPENDENCIES. The repo's support modules are `include`d
# scripts, not packages, so there is no `using` that reaches them; and the
# driver's own scope is `Main` when tools/adjoint_gradient.jl is run directly
# but the module `_AdjointArm` when run_reseact_adjoint.jl runs it. So this
# module reads them off its PARENT -- whatever scope `include`d it -- once, into
# const aliases. That is right in both cases, and on a worker as well. The parent must
# already have: `CapacityChem`, `RxTracedIntegrator`, `RxSymBlockJac`, and the
# split_common.jl top-levels `stencil_following_rule` /
# `index_promoted_refs_by_loop!`.
# ===========================================================================
module ShardKernel

using LinearAlgebra, Printf, Logging
using EarthSciAST
using EarthSciASTSplitter: split_system
using Reactant
using EarthSciASTDiff

const HOST = parentmodule(@__MODULE__)
const EA = EarthSciAST
const RX = Reactant

# CONST, resolved once here rather than per call. Both loaders have these in
# place before they include this file, so the deferred lookup the first cut used
# bought nothing. NOTE it is also not a speedup: the hypothesis that deferred
# lookups cost TRACE time (a captured Module reached through a non-inferable
# getproperty inside the traced closures) was MEASURED AND NOT CONFIRMED -- the
# whole setup delta sits in the `compile step` wall, which is four concurrent
# XLA compiles finishing within 0.3 s of each other (169.0 +/- 0.3 s in one arm,
# 219.8 +/- 0.1 s in the other), i.e. core contention, and it did not move when
# these consts went in. Keep the change for the explicit precondition below;
# do not cite it as performance. (slurm 10426936 / 10427441.)
#
# The price is an ordering requirement: the parent must already have these when
# it includes this file. Both loaders do, and the check below says so plainly
# if a third one does not.
for _n in (:CapacityChem, :RxTracedIntegrator, :RxSymBlockJac,
           :stencil_following_rule, :index_promoted_refs_by_loop!)
    isdefined(HOST, _n) || error(
        "shard_kernel.jl: its parent module $(HOST) has no `$(_n)`. This file must be " *
        "included AFTER capacity_chem.jl, rx_traced_integrator.jl, rx_sym_block_jac.jl " *
        "and split_common.jl are loaded into that same scope.")
end
const CC   = HOST.CapacityChem
const RTI  = HOST.RxTracedIntegrator
const RSBJ = HOST.RxSymBlockJac
const stencil_following_rule      = HOST.stencil_following_rule
const index_promoted_refs_by_loop! = HOST.index_promoted_refs_by_loop!

# Everything one shard needs, filled by `prepare_doc!` then `build!`.
mutable struct ShardState
    wid::Int
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
    ShardState(wid::Int, C::Int, NS::Int) =
        new(wid, C, NS, nothing, Dict{String,Any}(), nothing, Float64[],
            nothing, Symbol[], Dict{String,Int}(), Int[], nothing,
            nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, nothing)
end

wsay(st::ShardState, s) = (println("  [shard $(st.wid)] " * s); flush(stdout))

_devp(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))

"""
    prepare_doc!(cfg) -> (st, info)

Phase 1 of setup: load the model at the capacity metaparameters, split it, and
perform the capacity surgery at this shard's C. `info` names what the capacity
document still reads, so the caller can send exactly the constant arrays and
forcing shapes the build needs (the NEI inventory, 137,241 cells x 69 species,
is pruned away and never crosses any boundary).
"""
function prepare_doc!(cfg::Dict)
    st = ShardState(cfg["wid"], cfg["C"], cfg["NS"])
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
    st.cd, st.meta = CC.capacity_doc(st.docCAP0, st.C; say = s -> wsay(st, s),
                                        param_coords = true)
    wsay(st, @sprintf("capacity document ready at C=%d (%.1f s)", st.C, time() - tl))
    return st, (variables = collect(st.meta.variables), lane_arrays = st.meta.lane_arrays,
                emis_arrays = st.meta.emis_arrays, lonc = st.meta.lonc)
end

"""
    build!(st, cfg, ca, pshapes, lanes0) -> info

Phase 2: build the capacity evaluator with the constant arrays `ca` and forcing
buffers shaped like `pshapes`, prime the lanes with `lanes0` (the caller's first
gather), prepare the symbolic Jacobian and its gather plan, and compile the
step and the VJP. `info` carries the parameter values and keys and the one-off
timings.
"""
function build!(st::ShardState, cfg::Dict, ca::Dict{String,Any},
                pshapes::Dict{String,Any}, lanes0::Dict{String,Any})
    # CC / RTI / RSBJ are the const module aliases above -- see the note there:
    # capturing them as locals here is what made the traced closures dynamic.
    C = st.C; NS = st.NS
    st.pa = CC.lane_buffers(st.meta, pshapes, C)
    copy_lanes!(st.pa, lanes0)
    ov = Dict{String,Float64}(k => v for (k, v) in cfg["ov"] if k in st.meta.variables)
    tb = time()
    f, u0c, pc, _, vmc = Logging.with_logger(Logging.NullLogger()) do
        EA.build_evaluator(st.cd; form = :oop, parameter_overrides = ov,
                           const_arrays = ca, param_arrays = st.pa)
    end
    tbuild = time() - tb
    length(u0c) == NS * C ||
        error("shard build: capacity build at C=$C has $(length(u0c)) states, expected NS*C = $(NS*C)")
    st.f = f; st.u0c = copy(u0c); st.pc = pc; st.vmc = vmc
    st.pkeys = collect(keys(pc))
    st.masks = RTI.species_masks(vmc, NS, C)
    spnames = cfg["spnames"]::Vector{String}
    st.capsel = Vector{Int}(undef, NS * C)
    for s in 1:NS, l in 1:C
        nm = @sprintf("%s[%d,1,1]", spnames[s], l)
        haskey(vmc, nm) || error("shard build: the C=$C build has no state `$nm`")
        st.capsel[(s - 1) * C + l] = vmc[nm]
    end
    # the primed base point must evaluate finite, or the lane gather is not
    # reaching this build (a zero lane NaNs through log(PS/Pc))
    let du = EA.rhs_with_buffers(f)(u0c, pc, cfg["T0"], EA.forcing_buffers(f))
        nb = count(!isfinite, du)
        nb == 0 || error("shard build: the C=$C RHS returns $nb of $(length(du)) NON-FINITE derivatives at the primed base point")
    end

    tj = time()
    st.jacE = Logging.with_logger(Logging.NullLogger()) do
        EarthSciASTDiff.prepare_jacobian(EA.coerce_esm_file(st.cd); model_name = "Flattened",
            wrt = :states, build_kwargs = (; form = :oop, parameter_overrides = ov,
                                             const_arrays = ca, param_arrays = st.pa))
    end
    tjac = time() - tj
    st.jacE.oop || error("shard build: the C=$C band model came back IN-PLACE")
    String(st.jacE.structure) == "block_diagonal" ||
        error("shard build: the C=$C Jacobian is $(st.jacE.structure), not block_diagonal")
    st.plan = RSBJ.block_jac_plan(st.jacE;
                  runner_names = first.(sort(collect(vmc), by = last)))
    st.gjb = EA.rhs_with_buffers(st.jacE.fJ!)
    let w = RSBJ.validate_plan(st.plan, st.jacE, u0c, pc, cfg["T0"];
                               gjb = st.gjb, bufs = EA.forcing_buffers(st.jacE.fJ!))
        w <= 1e-12 || error("shard build: the C=$C gather plan does not reproduce the host Jacobian (worst relative $w)")
    end
    st.dev_bufs  = map(RX.ConcreteRArray, EA.forcing_buffers(f))
    st.dev_bufsJ = map(RX.ConcreteRArray, EA.forcing_buffers(st.jacE.fJ!))
    st.th = (p = _devp(pc), bufs = st.dev_bufs, bufsJ = st.dev_bufsJ)

    frhs = EA.rhs_with_buffers(f)
    plan = st.plan; gjb = st.gjb; masks = st.masks
    atol = cfg["ATOL_C"]::Float64; rtol = cfg["RTOL"]::Float64; jacmode = cfg["JACMODE"]::Symbol
    gC(u, th, t) = frhs(u, th.p, t, th.bufs)
    gJ(u, th, t) = RSBJ.block_jac(plan, gjb(RSBJ.gather_uj(plan, u), th.p, t, th.bufsJ))
    stepfn(u, th, t, dt) = RTI.ros23_step((uu, tt) -> gC(uu, th, tt), u, t, dt,
                                          NS, C, masks, atol, rtol;
                                          unrolled = true, jac = jacmode,
                                          symjac = (uu, tt) -> gJ(uu, th, tt),
                                          cellwise = true)
    vjpfn(u, th, lam, t, dt) = RTI.ros23_step_vjp(gC, u, th, t, dt, lam, NS, C, masks,
                                                  atol, rtol; jac = jacmode, gj = gJ,
                                                  active_bufs = false)
    copts = shard_copts(cfg)
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
    wsay(st, @sprintf("C=%d build %.1f s  jacobian %.1f s  compile step %.1f s  vjp %.1f s  (%d params)",
                      C, tbuild, tjac, tcs, tcv, length(st.pkeys)))
    return (pkeys = String.(st.pkeys),
            pvals = Float64[Float64(getfield(pc, k)) for k in st.pkeys],
            tbuild = tbuild, tjac = tjac, tcstep = tcs, tcvjp = tcv, rss = rss_gb())
end

function shard_copts(cfg)
    excl = cfg["EXCLP"]::Vector{String}
    if cfg["XLAFIX"]
        RX.CompileOptions(; sync = true,
                          xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
                          (isempty(excl) ? (;) : (; excluded_passes = excl))...)
    else
        RX.CompileOptions(; sync = true, (isempty(excl) ? (;) : (; excluded_passes = excl))...)
    end
end

rss_gb() = try
    parse(Float64, split(read("/proc/self/statm", String))[2]) * 4096 / 1024^3
catch; NaN end

function copy_lanes!(pa::Dict{String,Any}, lanes::Dict{String,Any})
    for (k, v) in lanes
        haskey(pa, k) || error("shard: lane buffer `$k` is not in this build's param_arrays")
        size(pa[k]) == size(v) ||
            error("shard: lane buffer `$k` has shape $(size(pa[k])) here, $(size(v)) from the driver")
        copyto!(pa[k], v)
    end
    return nothing
end

"New forcing for this shard's lanes (one call per GEOS-FP epoch)."
function refresh!(st::ShardState, lanes::Dict{String,Any})
    copy_lanes!(st.pa, lanes)
    EA.sync_forcing!(st.dev_bufs, EA.forcing_buffers(st.f))
    EA.sync_forcing!(st.dev_bufsJ, EA.forcing_buffers(st.jacE.fJ!))
    return nothing
end

# runner-layout shard vector (NS runs of C) <-> capacity layout
function to_cap(st::ShardState, v::Vector{Float64})
    out = Vector{Float64}(undef, length(v))
    @inbounds for i in eachindex(v); out[st.capsel[i]] = v[i]; end
    return out
end
function from_cap(st::ShardState, vc::Vector{Float64})
    out = Vector{Float64}(undef, length(vc))
    @inbounds for i in eachindex(vc); out[i] = vc[st.capsel[i]]; end
    return out
end

"""
    chem_step(st, uc, t, dt) -> (unew, sse, seconds)

`sse` is the SUM over this shard's states of the squared scaled residual --
`NS * sum(cell_err.^2)`, i.e. the partial sum the global `EEst = sqrt(sse/N)`
reduces over -- so the caller can rebuild the global norm exactly up to
reassociation. See tools/shard_exec.jl: that is a contract every executor owes,
not an implementation detail of this one.
"""
function chem_step(st::ShardState, uc::Vector{Float64}, t::Float64, dt::Float64)
    td = time()
    res = st.cstep(RX.ConcreteRArray(to_cap(st, uc)), st.th, RX.ConcreteRNumber(t),
                   RX.ConcreteRNumber(dt))
    raw = Array(res[1]); ce = Array(res[3])
    tdev = time() - td
    sse = 0.0
    @inbounds for e in ce; sse += e * e; end
    return from_cap(st, raw), st.NS * sse, tdev
end

"""
    chem_vjp(st, uc, lamc, t, dt) -> (lambda_in, dJdp over pkeys, seconds)
"""
function chem_vjp(st::ShardState, uc::Vector{Float64}, lamc::Vector{Float64},
             t::Float64, dt::Float64)
    td = time()
    r = st.cvjp(RX.ConcreteRArray(to_cap(st, uc)), st.th, RX.ConcreteRArray(to_cap(st, lamc)),
                RX.ConcreteRNumber(t), RX.ConcreteRNumber(dt))
    lin = Array(r[1]); gp = r[2].p
    tdev = time() - td
    return from_cap(st, lin), Float64[Float64(getfield(gp, k)) for k in st.pkeys], tdev
end

end # module ShardKernel
