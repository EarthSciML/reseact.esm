#!/usr/bin/env julia
# ===========================================================================
# tools/diurnal_run_reactant.jl -- the traced (Reactant/XLA) twin of
# tools/diurnal_run.jl: a LONG run, macro-stepped, with live forcing.
# ===========================================================================
# `run_reseact_reactant.jl` compiles the whole window as ONE XLA program taking
# ONE Lie-Trotter step, and says so in its own log: "single-window solve (no
# interior refresh)". That is the right shape for a 60 s validation window and
# the wrong shape for anything longer -- at a week it would mean a 604,800 s
# splitting interval and meteorology frozen at the start of the run. It would
# return a number, and the number would not be a simulation.
#
# This driver keeps the traced advance and puts the MACRO-STEP LOOP back around
# it, so the traced arm runs the same scheme the native arm does and the two are
# comparable step for step:
#
#   compile ONCE  ->  call 2016 times  ->  refresh forcing between calls
#
# Three properties of the traced program make that work, none of them accidental:
#
#   * `t0`, `t1`, `dt0T`, `dt0C` are RUNTIME arguments (ConcreteRNumber) and both
#     adaptive loops are `stablehlo.while` regions, so the compiled program does
#     not encode the window. One @compile serves every macro step, whatever its
#     length -- the compile cost is paid once and amortized over the whole run.
#   * the forcing buffers are REAL XLA INPUTS (`rhs_with_buffers` / B2 form), and
#     EarthSciAST guarantees that an in-place `copyto!` into those device arrays
#     between calls is observed on the next call with NO RETRACE. That is what
#     makes a discrete-cadence refresh survive compilation.
#   * the state stays on device between calls: `u` is fed straight back in, and
#     so are the final `dt`s, so the step-size controller carries across macro
#     steps exactly as it does within one.
#
# The forcing refresh needs a DiscreteMaterializer, which `run_reseact_reactant.jl`
# does not build (it never refreshes). Both halves get one here, and a refresh is
# merged_param .= provider_sample -> dm.materialize!() -> sync_forcing! to device.
#
# The stop grid follows `lie_trotter_solve`'s, INCLUDING its fencepost: the
# forcing-boundary set is half-open on the right, because a boundary that lands
# on a macro-step edge would otherwise be excluded from the step that ends there
# AND the step that starts there, and never refresh at all. See op_split.jl.
#
# Env: the same as tools/diurnal_run.jl (RESEACT_LABEL / LON0 / LAT0 / NLON /
# NLAT / NLEV / T0 / SOLVE_SECS / MACRO_DT / CSV / ZARR / OUT_EVERY /
# FLUSH_EVERY), plus
#   RESEACT_DT0T / RESEACT_DT0C  initial transport/chemistry dt (15.0 / 0.5)
#   RESEACT_RXENV                Julia env WITH Reactant (default run-model-jl)
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using SciMLBase, DiffEqCallbacks
using LinearAlgebra, Printf, Statistics, Logging
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using Reactant
using Blosc                      # activates EarthSciIOBloscExt for the Zarr sink
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
# block_jac.jl needs `BlockDiagonal` in scope even though this driver only wants
# `cellmajor_perm` from it (the block-diagonal Jacobian is the HOST arm's tool;
# the traced arm builds its own inside ros23_step).
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", "diurnal-rx")
const T0         = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "70200"))
const MACRO_DT   = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const CSV        = get(ENV, "RESEACT_CSV", "")
const ZARR       = get(ENV, "RESEACT_ZARR", "")
const OUT_EVERY  = parse(Int, get(ENV, "RESEACT_OUT_EVERY", "1"))
const FLUSH_EVERY = parse(Int, get(ENV, "RESEACT_FLUSH_EVERY", "12"))
const DT0T       = parse(Float64, get(ENV, "RESEACT_DT0T", "15.0"))
const DT0C       = parse(Float64, get(ENV, "RESEACT_DT0C", "0.5"))
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(lon0 = _env("LON0", 14), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 7), nlat = _env("NLAT", 7),
                           nlev = _env("NLEV", 72))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const T_END    = T0 + SOLVE_SECS
const NDAYS    = forcing_days_for(T0, T_END)
const RTOL, ATOL_T, ATOL_C = 1e-4, 1e-6, 1e-9

# Row-label sun angle at the domain centre (see diurnal_run.jl).
const K_GAMMA, D2R = 0.01721420632103996, 0.017453292519943295
const LON_MID = sum(SLICE.lon_deg) / 2
const LAT_MID = sum(SLICE.lat_deg) / 2
function cos_sza(t, lat, lon)
    g = K_GAMMA * (-12 / 24 + t / 86400)
    dec = 0.006918 - 0.399912cos(g) + 0.070257sin(g) - 0.006758cos(2g) +
          0.000907sin(2g) - 0.002697cos(3g) + 0.00148sin(3g)
    eqt = 229.18 * (0.000075 + 0.001868cos(g) - 0.032077sin(g) -
                    0.014615cos(2g) - 0.040849sin(2g))
    w = D2R * ((t / 60 + eqt + 4lon) / 4 - 180)
    return clamp(sin(D2R * lat) * sin(dec) + cos(D2R * lat) * cos(dec) * cos(w), -1, 1)
end

say("=== $LABEL : traced macro-step run ($(basename(MODEL))) " *
    "grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF T0=$(round(Int,T0)) " *
    "window=$(round(SOLVE_SECS/3600, digits=2)) h macro_dt=$(round(Int,MACRO_DT)) ===")
say(@sprintf("    slice: lon %.1f..%.1f, lat %.1f..%.1f (native origin %d,%d); forcing spans %d daily files",
    SLICE.lon_deg[1], SLICE.lon_deg[2], SLICE.lat_deg[1], SLICE.lat_deg[2],
    SLICE.lon0, SLICE.lat0, NDAYS))

# --------------------------------------------------------------------------- #
# 1. Split + build both halves :oop, over shared live forcing buffers, each with
#    its OWN DiscreteMaterializer so a refresh re-derives that half's caches.
# --------------------------------------------------------------------------- #
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = docs1 = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff, docs1
    file = EA.load(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
    docs1 = docs[1]
    f0 = reseact_forcing(CHEMDIR; ndays = NDAYS)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld
            discrete[k] = prov
        end
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part 2 var_map != part 1")
        end
    end
end
foreach(d -> d.materialize!(), dms)
say(@sprintf("BUILD %.2f s   nstates=%d  discrete_providers=%d",
    time() - tb, length(u0), length(discrete)))

# Seed m(0) from the real GEOS-FP surface pressure (species-major state here).
let dp0 = hydrostatic_dp(merged_param, ff.const_arrays, T0; slice = SLICE)
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dp0(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
                      parse(Int, mm.captures[3]))
    end
end

include(joinpath(RXDIR, "rx_native_patch.jl"))       # AFTER using Reactant/EarthSciAST
include(joinpath(RXDIR, "rx_merge_lib.jl"))
include(joinpath(RXDIR, "rx_traced_integrator.jl"))

const PARTNAME = ("transport", "pointwise")
host_bufs = [EA.forcing_buffers(fo[i]) for i in 1:2]
g4 = Vector{Any}(undef, 2)
for i in 1:2
    g4[i] = EA.rhs_with_buffers(fo[i]); kdesc = "stock"
    try
        mg = RxOopMerge.merge_oop_rhs(getfield(fo[i], :rhs))
        dmax = maximum(abs, mg.rhs(u0, p, T0, host_bufs[i]) .- fo[i](u0, p, T0))
        dmax == 0.0 ? (g4[i] = mg.rhs;
                       kdesc = "merged $(mg.stats.n_kernels)->$(mg.stats.n_merged)") :
            say("  half[$i]=$(PARTNAME[i]) merge NOT bit-identical (maxabs=$dmax) -> stock")
    catch e
        say("  half[$i]=$(PARTNAME[i]) merge FAILED ($(sprint(showerror, e))) -> stock")
    end
    say("  half[$i]=$(PARTNAME[i]) rhs: $kdesc")
end

P = cellmajor_perm(var_map)
const NS = P.NS; const NC = P.NC; const N = P.N
masks = RxTracedIntegrator.species_masks(var_map, NS, NC)

# --------------------------------------------------------------------------- #
# 2. The traced advance: ONE Lie-Trotter step over [t0, t1]. Identical body to
#    run_reseact_reactant.jl's -- the difference is entirely in who calls it.
# --------------------------------------------------------------------------- #
const CTRL_T = RxTracedIntegrator.pictrl_ssprk43()
const CTRL_C = RxTracedIntegrator.pictrl_ros23()
adv = let gT = g4[1], gC = g4[2], NS = NS, NC = NC, masks = masks
    (uR, pR, t0R, t1R, dtTR, dtCR, bufsT, bufsC) -> begin
        fT = (u, t, dtc, aux) -> RxTracedIntegrator.ssprk43_step(
            (uu, tt) -> gT(uu, aux.p, tt, aux.bufs), u, t, dtc, ATOL_T, RTOL)
        uT, _, dtTend, naT, nrT = RxTracedIntegrator.adaptive_solve(
            fT, uR, t0R, t1R, dtTR, CTRL_T, (p = pR, bufs = bufsT); clamp_nonneg = true)
        fC = (u, t, dtc, aux) -> RxTracedIntegrator.ros23_step(
            (uu, tt) -> gC(uu, aux.p, tt, aux.bufs), u, t, dtc, NS, NC, masks, ATOL_C, RTOL;
            unrolled = true)   # traced Jacobian loop is a net loss on the small chem RHS
        uC, _, dtCend, naC, nrC = RxTracedIntegrator.adaptive_solve(
            fC, uT, t0R, t1R, dtCR, CTRL_C, (p = pR, bufs = bufsC); clamp_nonneg = true)
        return uC, naT, nrT, naC, nrC, dtTend, dtCend
    end
end
_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
dev_bufs = [map(RX.ConcreteRArray, host_bufs[i]) for i in 1:2]
push_forcing!() = for i in 1:2; EA.sync_forcing!(dev_bufs[i], EA.forcing_buffers(fo[i])); end
push_forcing!()

PRd = _dev(p)
UR = RX.ConcreteRArray(copy(u0))
say("  @compile of ONE macro step (t0/t1/dt are runtime args, so this compile " *
    "serves every step of the run)...")
tc = time()
xadv = RX.@compile sync=true adv(UR, PRd, RX.ConcreteRNumber(T0),
                                 RX.ConcreteRNumber(T0 + MACRO_DT),
                                 RX.ConcreteRNumber(DT0T), RX.ConcreteRNumber(DT0C),
                                 dev_bufs[1], dev_bufs[2])
say(@sprintf("TRACED @compile: %.1f s", time() - tc))
try; report_patch_stats(); catch; end

# --------------------------------------------------------------------------- #
# 3. Forcing refresh -- the whole point of macro-stepping. EarthSciAST documents
#    that copyto! into the device buffers between calls is observed with NO
#    retrace (rhs_with_buffers), so this costs one host->device copy, not a
#    recompile.
# --------------------------------------------------------------------------- #
function refresh_forcing(t)
    for (k, prov) in discrete
        merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), dms)
    push_forcing!()
end

# Stop grid: the macro lattice UNION the GEOS-FP cadence boundaries. Half-open on
# the right, exactly as lie_trotter_solve builds it -- a boundary landing on a
# macro edge must belong to the step that ENDS there, or it is skipped by both
# neighbours and never refreshes (op_split.jl documents the measurement).
tstops = sort!(unique!(Float64[t for prov in values(discrete)
                               for t in EarthSciIO.refresh_times(prov)]))
fstops = Set(round(t; digits = 6) for t in tstops if T0 + 1e-6 < t <= T_END + 1e-6)
stops  = sort!(unique!(vcat(collect((T0 + MACRO_DT):MACRO_DT:(T_END - 1e-9)),
                            collect(fstops), Float64[T_END])))
say(@sprintf("  %d macro stops, %d of them GEOS-FP cadence boundaries", length(stops),
             length(fstops)))

# --------------------------------------------------------------------------- #
# 4. Digest + the continuity regression check (same quantities as diurnal_run.jl,
#    computed on the SPECIES-MAJOR state the traced arm carries).
# --------------------------------------------------------------------------- #
idxof(pre) = sort!([idx for (nm, idx) in var_map if startswith(nm, pre)])
const O3I = idxof("SuperFast.O3["); const OHI = idxof("SuperFast.OH[")
const NO2I = idxof("SuperFast.NO2["); const MI = idxof("Transport3D.m[")
const MCELL = let d = Dict{NTuple{3,Int},Int}()
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        d[(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
           parse(Int, mm.captures[3]))] = idx
    end
    d
end
const NLON_ = maximum(c[1] for c in keys(MCELL))
const NLAT_ = maximum(c[2] for c in keys(MCELL))
const _dA = Float64.(ff.const_arrays["Transport3D.dA"])
const _dB = Float64.(ff.const_arrays["Transport3D.dB"])
w_I3(t) = (t - 10800.0 * floor(t / 10800.0)) / 10800.0
function continuity_drift(u, t)
    F = merged_param["GEOSFP.GEOSFP_I3.PS"]; w = w_I3(t)
    isw(c) = c[1] == 1 || c[1] == NLON_ || c[2] == 1 || c[2] == NLAT_
    s2i = 0.0; ni = 0; s2w = 0.0; nw = 0; worsti = 0.0; wci = (0, 0, 0)
    for (c, idx) in MCELL
        i, j, k = c
        ps = 100.0 * ((1 - w) * F[1, SLICE.lat0 + j, SLICE.lon0 + i] +
                            w * F[2, SLICE.lat0 + j, SLICE.lon0 + i])
        dp = _dA[k] + _dB[k] * ps
        r = (u[idx] - dp) / dp
        if isw(c)
            s2w += r * r; nw += 1
        else
            s2i += r * r; ni += 1
            abs(r) > abs(worsti) && (worsti = r; wci = c)
        end
    end
    return (rms_int = sqrt(s2i / max(ni, 1)), rms_wall = sqrt(s2w / max(nw, 1)),
            worst_int = worsti, cell_int = wci)
end

sink = nothing
if !isempty(ZARR)
    out_times = collect(T0:(MACRO_DT * OUT_EVERY):T_END)
    sink = EA.build_zarr_sink(var_map, ZARR; output_times = out_times,
                              meta = EA.derive_output_meta(docs1))
    EA.sink_open!(sink)
    say("  gridded output -> $ZARR  ($(length(out_times)) records)")
end
sink_put!(t, u) = sink === nothing ? nothing :
    EA.sink_write!(sink, EA.StateSnapshot(Float64(t), [(u, (1:N,))], Dict{String,Array}()))

rows = NamedTuple[]
push_row!(t, u, wall) = push!(rows, (t = t, hours = t / 3600,
    cos_sza = cos_sza(t, LAT_MID, LON_MID),
    o3_min = minimum(u[O3I]), o3_mean = mean(u[O3I]), o3_max = maximum(u[O3I]),
    oh_max = maximum(u[OHI]), no2_mean = mean(u[NO2I]),
    m_min = minimum(u[MI]), wall = wall))
report(t, u, wall) = begin
    push_row!(t, u, wall); r = rows[end]; d = continuity_drift(u, t)
    say(@sprintf("   %6.2f  %7.4f  %9.5f  %9.5f  %9.5f  %10.3e  %9.3e  | %8.2e %9.2e %9.2e %-12s %6.1f",
        r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max, r.oh_max, r.no2_mean,
        d.rms_int, d.rms_wall, d.worst_int, string(d.cell_int), r.wall))
end

# --------------------------------------------------------------------------- #
# 5. The macro-step loop. State and step sizes stay on device between calls.
# --------------------------------------------------------------------------- #
say("  t_hours  cos_sza     O3_min     O3_mean     O3_max      OH_max     NO2_mean  | rms_int  rms_wall  worst_int @cell_int    wall_s")
report(T0, copy(u0), 0.0); sink_put!(T0, copy(u0))

ts = time(); tcur = T0; nT = nC = nrTt = nrCt = 0; nstep = 0; nrefresh = 0; ok = true
dtT = RX.ConcreteRNumber(DT0T); dtC = RX.ConcreteRNumber(DT0C)
for tnext in stops
    tnext <= tcur + 1e-9 && continue
    res = xadv(UR, PRd, RX.ConcreteRNumber(tcur), RX.ConcreteRNumber(tnext),
               dtT, dtC, dev_bufs[1], dev_bufs[2])
    global UR = res[1]
    global dtT = res[6]; global dtC = res[7]
    global nT += Int(round(Float64(res[2]))); global nrTt += Int(round(Float64(res[3])))
    global nC += Int(round(Float64(res[4]))); global nrCt += Int(round(Float64(res[5])))
    global tcur = tnext; global nstep += 1
    uh = Array(UR)
    if !all(isfinite, uh)
        global ok = false
        say("  !! non-finite state after macro step ending at t=$tnext -- stopping")
        break
    end
    report(tnext, uh, time() - ts)
    if nstep % OUT_EVERY == 0
        sink_put!(tnext, uh)
        sink === nothing || (nstep % (OUT_EVERY * FLUSH_EVERY) == 0 && EA.sink_flush!(sink))
    end
    if round(tnext; digits = 6) in fstops
        refresh_forcing(tnext); global nrefresh += 1
    end
end
solve_s = time() - ts
sink === nothing || (EA.sink_flush!(sink); EA.sink_close!(sink);
                     say("  gridded output committed: $ZARR"))

o3m = [r.o3_mean for r in rows]
say(@sprintf("RESULT label=%s cells=%d nstates=%d window_h=%.2f macro_dt=%.0f nmacro=%d nrefresh=%d nT=%d/%d nC=%d/%d solve_s=%.1f",
    LABEL, NC, N, SOLVE_SECS / 3600, MACRO_DT, nstep, nrefresh, nT, nrTt, nC, nrCt, solve_s))
say(@sprintf("DIURNAL O3_mean: start=%.5f min=%.5f max=%.5f end=%.5f  peak-to-trough=%.5f ppb  ok=%s",
    o3m[1], minimum(o3m), maximum(o3m), o3m[end], maximum(o3m) - minimum(o3m), ok))
if !isempty(CSV)
    open(CSV, "w") do io
        println(io, "t,hours,cos_sza,o3_min,o3_mean,o3_max,oh_max,no2_mean,m_min,wall")
        for r in rows
            println(io, join((r.t, r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max,
                              r.oh_max, r.no2_mean, r.m_min, r.wall), ","))
        end
    end
    say("wrote $CSV")
end
say("DONE $LABEL")
