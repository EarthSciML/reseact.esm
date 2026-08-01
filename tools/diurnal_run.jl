#!/usr/bin/env julia
# ===========================================================================
# tools/diurnal_run.jl -- long ReSEACT run that RECORDS a time series.
# ===========================================================================
# `run_reseact.jl` reports the final state only, which is all a 60 s validation
# window needs. Verifying a DIURNAL cycle needs the trajectory, so this driver
# reuses exactly the same machinery (prepare_split_docs / build_split_run /
# lie_trotter_solve) and samples the state after every Lie-Trotter macro step:
# solar geometry, the domain O3 min/mean/max, and OH -- the radical that makes
# the day/night contrast unambiguous.
#
# It is a DRIVER, not a second solver: the split, the forcing wiring, the
# operator-split stepping and the block-diagonal chemistry Jacobian are the same
# code paths run_reseact.jl uses. The only additions are the sampling loop and
# the report.
#
# Full spatiotemporal output goes through EarthSciAST's STREAMING SINK surface
# (streaming-output-sinks RFC §16) -- `build_zarr_sink` + sink_open!/write!/close!,
# the same protocol `simulate` drives via its `sinks=` kwarg. The sink API is
# explicitly specified to work without a solver embedded, which is what lets this
# macro-step loop feed it directly. Axis names and CF dimension coordinates come
# from `derive_output_meta` over the split run doc, so the store carries real
# `lon`/`lat`/`lev` names rather than positional ones. The console table below is
# a domain-reduced digest of the same records, for reading progress at a glance.
#
# NOTE the sink is state-only in its current wave (`sink_observed_names(::ZarrSink)`
# returns nothing), so the store holds SuperFast species + Transport3D.m. The
# j-rates are observeds and do not appear; cos_sza is recomputed here for labelling.
#
# Env (shared with run_reseact.jl unless noted):
#   RESEACT_MODEL, RESEACT_LABEL, RESEACT_NLON/NLAT/NLEV, RESEACT_MACRO_DT
#   RESEACT_LON0/LAT0  native GEOS-FP 4x5 slice origin (default 14/29 = the
#                      central US; 11 with NLON=13 gives lon -125..-65, CONUS)
#   RESEACT_T0         start time, s since 2013-01-01T00:00:00Z (default 0)
#   RESEACT_SOLVE_SECS window length, s                        (default 75600 = 21 h)
#   RESEACT_ZARR       gridded output store                    (default: none)
#   RESEACT_OUT_EVERY  write one record every N macro steps    (default 1)
#   RESEACT_FLUSH_EVERY  commit the sink every N written records (default 12)
#   RESEACT_CSV        also write the digest as CSV            (default: none)
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)
haskey(ENV, "ESS_KERNEL_CLASS_MERGE_DISABLE") || (ENV["ESS_KERNEL_CLASS_MERGE_DISABLE"] = "1")
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf, Statistics
# EarthSciIO keeps its compressors as WEAKDEPS so a base install stays light; the
# Zarr sink's default :diagnostic profile is Blosc-zstd, and loading Blosc is what
# activates EarthSciIOBloscExt to supply the encode. Without it sink_flush! throws.
using Blosc

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(RXDIR, "op_split.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", "diurnal")
# The window defaults are a single diurnal swing, NOT a limit. They used to be
# both: the forcing was one day of GEOS-FP files and each sample bracketed two
# records inside ONE file, so the usable span was [5400, 75600] = 19.5 h and
# nothing longer could be expressed. `reseact_forcing(...; ndays)` now hands the
# providers a `t -> url` resolver and EarthSciIO locates each record inside its
# own day's file, so the span is bounded only by how much data you fetch --
# `forcing_days_for` sizes that from the window. At lon -95 (UTC-6.3 h) the
# default covers local 19:10 -> 14:40 next day: evening, a full night, sunrise,
# solar noon and afternoon.
const T0         = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "70200"))
const MACRO_DT   = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const CSV        = get(ENV, "RESEACT_CSV", "")
const ZARR       = get(ENV, "RESEACT_ZARR", "")
const OUT_EVERY  = parse(Int, get(ENV, "RESEACT_OUT_EVERY", "1"))
const FLUSH_EVERY = parse(Int, get(ENV, "RESEACT_FLUSH_EVERY", "12"))  # sink writes per flush
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
# One origin -> metaparameters + the degree-space parameters + the index base
# hydrostatic_dp and continuity_drift read. See `native_slice`: the .esm cannot
# tie the two currencies together, so nothing else may spell them out.
const SLICE = native_slice(lon0 = _env("LON0", 14), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 7), nlat = _env("NLAT", 7),
                           nlev = _env("NLEV", 72))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const T_END = T0 + SOLVE_SECS
const RTOL, ATOL, ATOL_T = 1e-4, 1e-9, 1e-6
const LU = LinearSolve.LUFactorization()
const PS_REF = 101325.0

# The model's own solar chain, mirrored here ONLY to label the output rows with
# the sun angle the run is actually seeing (Transport3D.cos_sza_c is an internal
# observed, not a state, so it is not in `u`). Same constants, same formulas.
const K_GAMMA, D2R = 0.01721420632103996, 0.017453292519943295
# The digest's cos_sza column labels ONE point -- the domain centre -- so a wider
# slice does not go on being described by the sun over the middle of the old one.
const LON_MID = sum(SLICE.lon_deg) / 2
const LAT_MID = sum(SLICE.lat_deg) / 2
function cos_sza(t, lat, lon)
    g = K_GAMMA * (-12 / 24 + t / 86400)                       # doy0=1, hour0=0
    dec = 0.006918 - 0.399912cos(g) + 0.070257sin(g) - 0.006758cos(2g) +
          0.000907sin(2g) - 0.002697cos(3g) + 0.00148sin(3g)
    eqt = 229.18 * (0.000075 + 0.001868cos(g) - 0.032077sin(g) -
                    0.014615cos(2g) - 0.040849sin(2g))
    w = D2R * ((t / 60 + eqt + 4lon) / 4 - 180)
    return clamp(sin(D2R * lat) * sin(dec) + cos(D2R * lat) * cos(dec) * cos(w), -1, 1)
end

const NDAYS = forcing_days_for(T0, T_END)
say("=== $LABEL : build ($(basename(MODEL))) grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF " *
    "T0=$(round(Int,T0)) window=$(round(SOLVE_SECS/3600, digits=2)) h macro_dt=$(round(Int,MACRO_DT)) ===")
say(@sprintf("    slice: lon %.1f..%.1f, lat %.1f..%.1f (native origin %d,%d); forcing spans %d daily files",
    SLICE.lon_deg[1], SLICE.lon_deg[2], SLICE.lat_deg[1], SLICE.lat_deg[2],
    SLICE.lon0, SLICE.lat0, NDAYS))
docs = prepare_split_docs(MODEL; metaparameters = GRID_MP)
ff = reseact_forcing(CHEMDIR; ndays = NDAYS)
ff = merge(ff, (; const_arrays = GridResize.slice_hybrid_coefs(ff.const_arrays, NLEV_EFF)))
# Pull every file the cadence will need BEFORE the solve. A mid-solve fetch is
# not wrong, but it puts a multi-hour run at the mercy of the network at an
# arbitrary macro step; this fails fast instead, while nothing is invested.
let tp = time(), n = sum(length(EarthSciIO.prefetch(pr)) for pr in values(ff.providers))
    say(@sprintf("    prefetch: %d provider-files warm in %.1f s", n, time() - tp))
end
tb = time()
run = build_split_run(docs, (T0, T_END);
    providers = ff.providers, const_arrays = ff.const_arrays,
    parameters = ff.parameters, slice = SLICE)
say(@sprintf("BUILD %.2f s  nstates=%d", time() - tb, length(run.u0)))

P = cellmajor_perm(run.var_map)
f_trans_cm! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)
u = run.u0[P.sm_of_cm]
foreach(d -> d.materialize!(), run.dms)
# m(0) from the REAL GEOS-FP PS, not a constant -- see hydrostatic_dp.
let dp0 = hydrostatic_dp(run.merged_param, ff.const_arrays, T0; slice = run.slice),
    mb = P.base_pos["Transport3D.m"]
    for c in P.cells
        u[(P.cell_pos[c] - 1) * P.NS + mb] = dp0(c[1], c[2], c[3])
    end
end
o3rng = P.base_pos["SuperFast.O3"]:P.NS:P.N
ohrng = P.base_pos["SuperFast.OH"]:P.NS:P.N
no2rng = P.base_pos["SuperFast.NO2"]:P.NS:P.N

tgz(g, u, p, t) = (fill!(g, 0); nothing)
fc = SciMLBase.ODEFunction(f_chem_cm!; jac = jac_cm!, jac_prototype = mkjp(), tgrad = tgz)
inner_algs = (OrdinaryDiffEqSSPRK.SSPRK43(),
              OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false, linsolve = LU))
disc = Dict(String(k) => prov for (k, prov) in ff.providers if !EA.provider_is_const(prov))
refresh_forcing = t -> begin
    for (k, prov) in disc
        run.merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), run.dms)
end

# --- gridded streaming output (EarthSciAST sink protocol; EarthSciIO Zarr v3) ---
# The sink writes the flat state in var_map (species-major) order, but the solve
# runs cell-major, so un-permute before every write.
to_sm(ucm) = (usm = similar(ucm); @inbounds for i in 1:P.N; usm[P.sm_of_cm[i]] = ucm[i]; end; usm)
sink = nothing
if !isempty(ZARR)
    out_times = collect(T0:(MACRO_DT * OUT_EVERY):T_END)
    meta = EA.derive_output_meta(docs[1])
    sink = EA.build_zarr_sink(run.var_map, ZARR; output_times = out_times, meta = meta)
    EA.sink_open!(sink)
    say("  gridded output -> $ZARR  ($(length(out_times)) records)")
end
sink_put!(t, ucm) = sink === nothing ? nothing :
    EA.sink_write!(sink, EA.StateSnapshot(Float64(t),
        [(to_sm(ucm), (1:P.N,))], Dict{String,Array}()))

# --------------------------------------------------------------------------- #
# CONTINUITY DIAGNOSTIC.
#
# The drift between the integrated air mass m and the hydrostatic thickness
# dp = dA[k] + dB[k]*PS(t) that m is supposed to equal. This is the quantity the
# pressure fixer exists to kill, and it is what long runs used to die of: the
# face fluxes are built from GEOS-FP's time-AVERAGED A3 winds while PS is
# INSTANTANEOUS I3, so the raw discrete identity d(dp)/dt + div(M) = 0 does not
# hold (Jockel et al. 2001) and m wandered away from dp at ~3e-6 1/s.
#
# With the column-local fixer in place -- Mz diagnosed from the CORRECTED
# horizontal divergence and the continuity equation reading the same divh_fix --
# this should now sit at ROUNDOFF (measured 1.3e-17 1/s rms on the RHS itself;
# see tools/continuity_residual.jl). So the numbers below are no longer a
# characterisation of a known defect, they are a REGRESSION CHECK: anything that
# climbs off 1e-16 means the fixer has been broken, most likely by the m equation
# and Mz falling out of step about which divergence they use.
#
# Interior and wall are still reported separately. The split used to discriminate
# a genuine continuity bias (interior, wants a pressure fixer) from a boundary
# artifact (open lateral wall, wants a BC fix); now it mostly guards against a
# regression that shows up only in the boundary REGIONS of the PPM stencils,
# which the interior rms alone would hide. At 7x7 almost every cell IS a boundary
# cell, so a wider domain makes this far more readable.
const _PSk = "GEOSFP.GEOSFP_I3.PS"
const _dA = Float64.(ff.const_arrays["Transport3D.dA"])
const _dB = Float64.(ff.const_arrays["Transport3D.dB"])
const NLON_ = maximum(c[1] for c in P.cells); const NLAT_ = maximum(c[2] for c in P.cells)
const NLEV_ = maximum(c[3] for c in P.cells)
"w_I3 for the I3 cadence: 3-hourly instantaneous anchored at 00:00Z."
w_I3(t) = (t - 10800.0 * floor(t / 10800.0)) / 10800.0
function continuity_drift(u, t)
    F = run.merged_param[_PSk]          # NATIVE [time, lat, lon], hPa
    w = w_I3(t)
    # Split INTERIOR from WALL. The open lateral walls prescribe inflow tracer
    # mixing ratios (qbc_*) but take the AIR-MASS flux straight from the wind
    # field, which is not constrained to close the interior column budget -- a
    # different defect from an interior continuity bias, and one a pressure fixer
    # would NOT correct. Reporting them separately is what decides which to fix.
    worst = 0.0; wc = (0, 0, 0); s2i = 0.0; ni = 0; s2w = 0.0; nw = 0
    worsti = 0.0; wci = (0, 0, 0)
    isw(c) = c[1] == 1 || c[1] == NLON_ || c[2] == 1 || c[2] == NLAT_
    for c in P.cells
        i, j, k = c
        ps = 100.0 * ((1 - w) * F[1, SLICE.lat0 + j, SLICE.lon0 + i] +
                            w * F[2, SLICE.lat0 + j, SLICE.lon0 + i])
        dp = _dA[k] + _dB[k] * ps
        mv = u[(P.cell_pos[c] - 1) * P.NS + P.base_pos["Transport3D.m"]]
        r = (mv - dp) / dp
        if isw(c)
            s2w += r * r; nw += 1
        else
            s2i += r * r; ni += 1
            abs(r) > abs(worsti) && (worsti = r; wci = c)
        end
        abs(r) > abs(worst) && (worst = r; wc = c)
    end
    return (rms = sqrt((s2i + s2w) / max(ni + nw, 1)),
            rms_int = sqrt(s2i / max(ni, 1)), rms_wall = sqrt(s2w / max(nw, 1)),
            worst = worst, cell = wc, onwall = isw(wc),
            worst_int = worsti, cell_int = wci, nint = ni, nwall = nw)
end

rows = NamedTuple[]
push_row!(t, u, wall) = push!(rows, (t = t, hours = t / 3600,
    cos_sza = cos_sza(t, LAT_MID, LON_MID),
    o3_min = minimum(u[o3rng]), o3_mean = mean(u[o3rng]), o3_max = maximum(u[o3rng]),
    oh_max = maximum(u[ohrng]), no2_mean = mean(u[no2rng]),
    m_min = minimum(u[P.base_pos["Transport3D.m"]:P.NS:P.N]), wall = wall))

say("  t_hours  cos_sza     O3_min     O3_mean     O3_max      OH_max     NO2_mean  | rms_int  rms_wall  worst_int @cell_int    wall_s")
push_row!(T0, u, 0.0); sink_put!(T0, u)
r = rows[end]
let d = continuity_drift(u, T0)
    say(@sprintf("   %6.2f  %7.4f  %9.5f  %9.5f  %9.5f  %10.3e  %9.3e  | %8.2e %9.2e %9.2e %-12s %6.1f",
        r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max, r.oh_max, r.no2_mean,
        d.rms_int, d.rms_wall, d.worst_int, string(d.cell_int), r.wall))
end

ts = time(); nT = nC = 0; ok = true; nstep = 0
t = T0
while t < T_END - 1e-9
    global nstep += 1
    tnext = min(t + MACRO_DT, T_END)
    res = lie_trotter_solve(f_trans_cm!, fc, u, (t, tnext), run.p, inner_algs;
        macro_dt = tnext - t, reltols = (RTOL, RTOL), abstols = (ATOL_T, ATOL),
        refresh = refresh_forcing, forcing_tstops = run.tstops, clamp_nonneg = true)
    # A macro step that did not succeed must NOT be papered over by advancing t:
    # the state comes back unchanged, so the run marches on producing a frozen
    # trajectory that still LOOKS like a completed simulation. Report which
    # species is responsible and stop.
    if !(res.retcode == SciMLBase.ReturnCode.Success ||
         res.retcode == SciMLBase.ReturnCode.Default)
        global ok = false
        say("  !! macro step [$t, $tnext] retcode=$(res.retcode) -- diagnosing state:")
        for (nm, b) in sort(collect(P.base_pos), by = last)
            v = res.u[b:P.NS:P.N]
            say(@sprintf("     %-22s min=%- .6e max=%- .6e  nneg=%d nnan=%d",
                nm, minimum(v), maximum(v), count(<(0), v), count(isnan, v)))
        end
        say("  !! stopping at t=$t (see above); partial series below")
        break
    end
    global u = res.u; global nT += res.naT; global nC += res.naC
    global t = tnext
    push_row!(t, u, time() - ts)
    if nstep % OUT_EVERY == 0
        sink_put!(t, u)
        # Commit written slabs periodically. Over a multi-day run the difference
        # between "flushed at the end" and "flushed as you go" is the difference
        # between a crash at hour 40 costing an hour and costing the whole run.
        # (The digest survives regardless -- every row is printed as it is taken.)
        sink === nothing || (nstep % (OUT_EVERY * FLUSH_EVERY) == 0 && EA.sink_flush!(sink))
    end
    r = rows[end]
    d = continuity_drift(u, t)
    say(@sprintf("   %6.2f  %7.4f  %9.5f  %9.5f  %9.5f  %10.3e  %9.3e  | %8.2e %9.2e %9.2e %-12s %6.1f",
        r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max, r.oh_max, r.no2_mean,
        d.rms_int, d.rms_wall, d.worst_int, string(d.cell_int), r.wall))
end
solve_s = time() - ts
if sink !== nothing
    EA.sink_flush!(sink); EA.sink_close!(sink)
    say("  gridded output committed: $ZARR")
end

o3m = [r.o3_mean for r in rows]
say(@sprintf("RESULT label=%s cells=%d nstates=%d window_h=%.2f macro_dt=%.0f nmacro=%d nT=%d nC=%d solve_s=%.1f build_states=%d",
    LABEL, P.NC, P.N, SOLVE_SECS / 3600, MACRO_DT, length(rows) - 1, nT, nC, solve_s, P.N))
say(@sprintf("DIURNAL O3_mean: start=%.5f min=%.5f max=%.5f end=%.5f  peak-to-trough=%.5f ppb  ok=%s",
    o3m[1], minimum(o3m), maximum(o3m), o3m[end], maximum(o3m) - minimum(o3m), ok && all(isfinite, u)))

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
