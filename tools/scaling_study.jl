#!/usr/bin/env julia
# ===========================================================================
# tools/scaling_study.jl -- how ReSEACT wall time scales with problem size.
# ===========================================================================
# Runs a ladder of (grid, window) configurations through the SAME machinery as
# run_reseact.jl and reports, per point:
#
#   build_s   prepare_split_docs + build_split_run (both halves)
#   first_s   the FIRST macro step -- dominated by first-call codegen compile
#   solve_s   the remaining steps  -- the MARGINAL cost that actually scales
#   nT / nC   accepted transport / chemistry steps
#
# Separating first_s from solve_s is the point of this script. A single short
# window is almost entirely one-time compile, so a naive "wall time vs cells"
# curve measures the compiler, not the model. `solve_s / (nT+nC)` is the number
# that extrapolates.
#
# Everything runs in ONE Julia session so the ~60 s startup + package load is
# paid once rather than per point; each grid still gets its own build (the grid
# dims are .esm metaparameters, folded at load).
#
# Env:
#   RESEACT_MODEL     .esm to run          (default: repo-root reseact.esm)
#   RESEACT_SWEEP     "vertical" | "horizontal" | "temporal" | "all" (default all)
#   RESEACT_T0        start time, s        (default 64800)
#   RESEACT_CSV       write results here   (default: none)
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)
# The kernel-class merge stays ON, matching the runners. This script used to force
# it OFF by default, which meant the ladder measured a configuration nothing
# actually runs in -- and specifically the one whose IR grows with the grid, which
# is the thing a scaling study exists to characterise. Set
# ESS_KERNEL_CLASS_MERGE_DISABLE=1 explicitly for an A/B against the old numbers.
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf, Logging

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(RXDIR, "op_split.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const SWEEP = get(ENV, "RESEACT_SWEEP", "all")
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "64800"))
const CSV   = get(ENV, "RESEACT_CSV", "")
const RTOL, ATOL, ATOL_T = 1e-4, 1e-9, 1e-6
const LU = LinearSolve.LUFactorization()
const PS_REF = 101325.0
const MACRO_DT = 300.0

# (nlon, nlat, nlev, window_seconds)
# Minimum extent per axis is 6, not 1: the PPM rules carve three boundary regions
# at each end plus an interior [4, N-3], so N=4 inverts that bound and the model
# fails to load (`makearray_region_inverted`). N=6 makes the interior empty, which
# IS legal ([start, start-1] contributes no elements, esm-spec §4.3.2).
const LADDERS = Dict(
    # vertical: cells scale 1x -> 9x at fixed horizontal extent
    "vertical"   => [(7, 7, n, 600.0) for n in (8, 12, 18, 24, 36, 72)],
    # horizontal: NLAT caps at 17 and NLON at 57 for this native slice
    "horizontal" => [(7, 7, 8, 600.0), (11, 11, 8, 600.0),
                     (15, 15, 8, 600.0), (21, 17, 8, 600.0)],
    # temporal: fixed grid, growing simulated window
    "temporal"   => [(7, 7, 8, w) for w in (300.0, 600.0, 1800.0, 3600.0, 10800.0)],
)

function build_for(nlon, nlat, nlev)
    mp = Dict("NLON" => nlon, "NLAT" => nlat, "NLEV" => nlev)
    tb = time()
    docs, run, ff = Logging.with_logger(Logging.NullLogger()) do
        d = prepare_split_docs(MODEL; metaparameters = mp)
        f = reseact_forcing(CHEMDIR)
        f = merge(f, (; const_arrays = GridResize.slice_hybrid_coefs(f.const_arrays, nlev)))
        r = build_split_run(d, (T0, T0 + 1.0); providers = f.providers,
                            parameters = f.parameters, const_arrays = f.const_arrays)
        (d, r, f)
    end
    return run, ff, time() - tb
end

# Builds are cached by grid: the temporal sweep holds the grid fixed, and a
# rebuild there would re-pay ~85 s of build AND ~350 s of first-call codegen
# compile per point while measuring nothing new. Reusing one build also means the
# temporal points share the exact same compiled RHS, so their marginal costs are
# comparable to each other rather than to five separate compilations.
const BUILD_CACHE = Dict{Tuple{Int,Int,Int},Any}()
function cached_build(nlon, nlat, nlev)
    get!(BUILD_CACHE, (nlon, nlat, nlev)) do
        build_for(nlon, nlat, nlev)
    end
end

function run_point(nlon, nlat, nlev, window)
    warm = haskey(BUILD_CACHE, (nlon, nlat, nlev))   # compile already paid?
    run, ff, build_s = cached_build(nlon, nlat, nlev)
    P = cellmajor_perm(run.var_map)
    ft! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
    fchem! = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
    jac!, mkjp = block_fd_jac(fchem!, P.NS, P.NC)
    u = run.u0[P.sm_of_cm]
    foreach(d -> d.materialize!(), run.dms)
    let dp0 = hydrostatic_dp(run.merged_param, ff.const_arrays, T0; slice = run.slice),
        mb = P.base_pos["Transport3D.m"]
        for c in P.cells
            u[(P.cell_pos[c] - 1) * P.NS + mb] = dp0(c[1], c[2], c[3])
        end
    end
    tgz(g, u, p, t) = (fill!(g, 0); nothing)
    fc = SciMLBase.ODEFunction(fchem!; jac = jac!, jac_prototype = mkjp(), tgrad = tgz)
    algs = (OrdinaryDiffEqSSPRK.SSPRK43(),
            OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false, linsolve = LU))
    disc = Dict(String(k) => p for (k, p) in ff.providers if !EA.provider_is_const(p))
    refresh = t -> begin
        for (k, p) in disc
            run.merged_param[k] .= EA._provider_const_field(EA.provider_sample(p, t), k)
        end
        foreach(d -> d.materialize!(), run.dms)
    end
    step!(t0, t1) = lie_trotter_solve(ft!, fc, u, (t0, t1), run.p, algs;
        macro_dt = t1 - t0, reltols = (RTOL, RTOL), abstols = (ATOL_T, ATOL),
        refresh = refresh, forcing_tstops = run.tstops, clamp_nonneg = true)

    # First macro step separately: it carries the one-time codegen compile.
    t = T0; dt1 = min(MACRO_DT, window)
    tf = time(); res = step!(t, t + dt1); first_s = time() - tf
    u = res.u; t += dt1; nT = res.naT; nC = res.naC; ok = true
    ts = time()
    while t < T0 + window - 1e-9
        tn = min(t + MACRO_DT, T0 + window)
        res = step!(t, tn)
        u = res.u; t = tn; nT += res.naT; nC += res.naC
        (res.retcode == SciMLBase.ReturnCode.Success ||
         res.retcode == SciMLBase.ReturnCode.Default) || (ok = false)
    end
    solve_s = time() - ts
    o3 = u[P.base_pos["SuperFast.O3"]:P.NS:P.N]
    return (; nlon, nlat, nlev, window, cells = P.NC, nstates = P.N,
            build_s, first_s, solve_s, nT, nC, warm,
            o3_min = minimum(o3), o3_max = maximum(o3),
            ok = ok && all(isfinite, u))
end

sweeps = SWEEP == "all" ? ["vertical", "horizontal", "temporal"] : [SWEEP]
rows = NamedTuple[]
for sw in sweeps
    say("\n===== sweep: $sw =====")
    say("  grid          cells  nstates  window_s  build_s  first_s  solve_s  nT   nC   marg_ms/step warm  ok")
    for (nlon, nlat, nlev, w) in LADDERS[sw]
        r = try
            run_point(nlon, nlat, nlev, w)
        catch e
            say("  $(nlon)x$(nlat)x$(nlev) w=$(w)  FAILED: $(sprint(showerror, e)[1:min(160,end)])")
            continue
        end
        push!(rows, merge(r, (; sweep = sw)))
        steps = max(r.nT + r.nC, 1)
        say(@sprintf("  %2dx%2dx%-3d %7d %8d %9.0f %8.1f %8.1f %8.1f %4d %4d %13.2f %5s  %s",
            r.nlon, r.nlat, r.nlev, r.cells, r.nstates, r.window, r.build_s,
            r.first_s, r.solve_s, r.nT, r.nC, 1000 * r.solve_s / steps, r.warm, r.ok))
    end
end

if !isempty(CSV)
    open(CSV, "w") do io
        println(io, "sweep,nlon,nlat,nlev,cells,nstates,window,build_s,first_s,solve_s,nT,nC,warm,o3_min,o3_max,ok")
        for r in rows
            println(io, join((r.sweep, r.nlon, r.nlat, r.nlev, r.cells, r.nstates, r.window,
                              r.build_s, r.first_s, r.solve_s, r.nT, r.nC, r.warm,
                              r.o3_min, r.o3_max, r.ok), ","))
        end
    end
    say("\nwrote $CSV")
end
say("DONE scaling_study")
