#!/usr/bin/env julia
# ===========================================================================
# run_reseact.jl -- run the full ReSEACT model with the NATIVE in-place runner.
# ===========================================================================
# THE DEFAULT RUN IS A ONE-WEEK CONUS SIMULATION: 13x7x72 cells (lon -125..-65,
# lat 26..50; 6,552 cells, 85,176 states) over 168 h of GEOS-FP meteorology,
# macro-stepped at 300 s. Measured on this machine at `julia -t 8`: 683 s build
# + 15,227 s solve (4.2 h wall, ~40x realtime). Override any of it through the
# environment (below) -- nothing about the driver is CONUS-specific.
#
# It builds the operator-split model as two in-place `f!(du,u,p,t)` closures and
# drives a Lie-Trotter operator split entirely on the CPU:
#
#   transport (non-stiff, stencil terms)    -> SSPRK43        (explicit)
#   chemistry (stiff, cell-local pointwise) -> Rosenbrock23   (implicit, with a
#                                              block-diagonal Jacobian: one
#                                              NS x NS block per cell)
#
# The two halves share ONE state vector and ONE set of live GEOS-FP forcing
# buffers, so f_full(u) = f_transport(u) + f_chemistry(u) exactly. This is the
# REFERENCE runner: `run_reseact_reactant.jl` is validated against it.
#
# The Lie-Trotter coupling is done by the REAL SciML operator-splitting solver,
# OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov, via the thin
# `lie_trotter_solve` driver in tools/reactant_handoff/op_split.jl. That package
# USED to be unusable here: it wraps each sub-operator in a plain ODEProblem,
# whose init() densified our BlockDiagonal jac_prototype into an N x N dense
# matrix (O(N^3) LU, the exact cost the split exists to avoid). The root cause
# was a single missing `similar(::BlockDiagonal, ::Type)` method; blockdiag_similar.jl
# supplies it. See SPLIT_SOLVER.md and op_split.jl for the full story.
#
# Two things a week-long run needs that a validation window does not, and that
# are therefore built in rather than optional:
#
#   * BISECTION ON FAILURE. The pre-dawn NO minimum (first hit at sim-hour 11.33)
#     makes Rosenbrock23's TRIAL sub-steps go negative and convergence collapse.
#     `clamp_nonneg` is INERT against it -- it inspects only ACCEPTED states, and
#     those never go negative. `lie_trotter_solve_bisect` retries the failed macro
#     step over two halves instead, which clears it in 2 bisections and costs LESS
#     chemistry work than the failed 300 s attempt. The durable fix is upstream
#     (the splitting package silently drops callbacks, so `PositiveDomain` cannot
#     fire) -- see HELPERS.md §3.
#   * STREAMING OUTPUT. Full spatiotemporal output goes through EarthSciAST's
#     sink protocol (`derive_output_meta` -> `build_zarr_sink` ->
#     sink_open!/write!/flush!/close!), the same surface `simulate` drives via
#     `sinks=`, flushed every FLUSH_EVERY records. Over a multi-day run the
#     difference between "flushed at the end" and "flushed as you go" is the
#     difference between a crash at hour 40 costing an hour and costing the run.
#     The sink is state-only in its current wave, so the store holds SuperFast
#     species + Transport3D.m; j-rates are observeds and do not appear.
#
# The console table is a domain-reduced digest of the same records -- solar
# geometry, O3 min/mean/max, OH (the radical that makes the day/night contrast
# unambiguous), NO2, and the continuity regression check -- for reading progress
# at a glance.
#
# THREADS. Run this under `julia -t 8` (or however many cores you have). The
# threaded codegen tier is what took the CONUS week run from ~1:1 realtime to
# ~40x; it activates only when EarthSciAST's Polyester extension is loadable, so
# the driver reports below whether it is actually on rather than letting a
# 10x-slow run look normal.
#
# Env:
#   RESEACT_MODEL      .esm to run                    (default: repo-root reseact.esm)
#   RESEACT_LABEL      tag for the RESULT line        (default "reseact")
#   RESEACT_LON0/LAT0  native GEOS-FP 4x5 slice origin (default 11/29 = CONUS;
#                      14 with NLON=7 is the old central-US box)
#   RESEACT_NLON/NLAT/NLEV  grid dims                 (default 13/7/72 = CONUS.
#                      Native 4x5 bounds: NLON <= 57, NLAT <= 17, NLEV <= 72.)
#   RESEACT_T0         start, s since 2016-01-01T00:00:00Z (default 5400 = 01:30Z).
#                      Also the model's solar epoch, so t IS the UTC clock.
#   RESEACT_SOLVE_SECS window length, s               (default 604800 = 7 days)
#   RESEACT_MACRO_DT   Lie-Trotter interval, s        (default 300)
#   RESEACT_MIN_MACRO_DT  bisection floor, s          (default 9 = five halvings)
#   RESEACT_JAC        "ad" (default) or "fd"         chemistry block Jacobian
#   RESEACT_ZARR       gridded output store           (default: none)
#   RESEACT_OUT_EVERY  write one record every N macro steps  (default 12 = hourly)
#   RESEACT_FLUSH_EVERY  commit the sink every N written records (default 12)
#   RESEACT_CSV        also write the digest as CSV   (default: none)
#   RESEACT_STATEDUMP  raw state written iff the solve fails  (default: none)
#   RESEACT_RUN_ENV    Julia env to activate          (default: <repo>/run-model-jl)
#
# Helper code it pulls in (see HELPERS.md for the migration plan):
#   prototypes/reseact_3d_chem/split_common.jl     prepare_split_docs, build_split_run,
#                                                  reseact_forcing, native_slice,
#                                                  validate_reseact, hydrostatic_dp
#   prototypes/reseact_3d_chem/blockdiag_local.jl  BlockDiagonal (from EarthSciMLBase)
#   prototypes/reseact_3d_chem/block_jac.jl        cellmajor_perm/rhs, block_{fd,ad}_jac
#   tools/reactant_handoff/op_split.jl             lie_trotter_solve{,_bisect}
#   tools/grid_resize.jl                           slice_hybrid_coefs (NLEV < 72)
# ===========================================================================
import Pkg
const REPO = @__DIR__
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)
# The kernel-class merge stays ON. It used to be forced off here, inheriting a
# default measured on a 7x7x7 box where the merged codegen tier's first-call
# compile swamped a ~25 s interpreted solve -- a real result, for a grid small
# enough that the unmerged IR was affordable in the first place. At CONUS that
# trade reverses: the unmerged per-cell IR is the thing that does not fit, and
# paying a fixed compile to keep the kernel count grid-independent is the whole
# point of the merge. ESS_KERNEL_CLASS_MERGE_DISABLE=1 forces the old behaviour.
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf, Statistics
# EarthSciIO keeps its compressors as WEAKDEPS so a base install stays light; the
# Zarr sink's default :diagnostic profile is Blosc-zstd, and loading Blosc is what
# activates EarthSciIOBloscExt to supply the encode. Without it sink_flush! throws.
using Blosc
# Polyester activates EarthSciASTPolyesterExt, the THREADED codegen tier. It is a
# weakdep, so `using` it is the whole activation -- and it is worth ~10x on a long
# run, which is too large a difference to leave to whether the caller remembered.
# Loaded before the build, because the tier has to be live when the kernels are
# generated, not merely by the time they are called. If it is not installed in the
# active environment this stays false and the banner below says so out loud.
const POLYESTER_OK = try; @eval using Polyester; true; catch; false; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(RXDIR, "op_split.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", "reseact")
# One week of GEOS-FP, starting 01:30Z on 2016-01-01. T0=5400 is not arbitrary:
# it is the first A3 record centre, so the very first macro step already has both
# bracketing records of every cadence. `reseact_forcing(...; ndays)` hands the
# providers a `t -> url` resolver and EarthSciIO locates each record inside its
# OWN day's file, so the span is bounded only by how much data is fetched --
# `forcing_days_for` sizes that from the window. A window that crosses a MONTH
# boundary throws by design (NEI is a frozen monthly inventory), which is why the
# week starts on the 1st.
const T0         = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "604800"))
const MACRO_DT   = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const CSV        = get(ENV, "RESEACT_CSV", "")
const ZARR       = get(ENV, "RESEACT_ZARR", "")
const OUT_EVERY  = parse(Int, get(ENV, "RESEACT_OUT_EVERY", "12"))     # hourly at macro_dt=300
const FLUSH_EVERY = parse(Int, get(ENV, "RESEACT_FLUSH_EVERY", "12"))  # sink writes per flush
const STATEDUMP  = get(ENV, "RESEACT_STATEDUMP", "")   # raw state written iff the solve fails
# Floor on the bisection retry: five halvings from 300 s. A run needing chemistry
# steps shorter than this has a different problem, and should fail loudly.
const MIN_MACRO_DT = parse(Float64, get(ENV, "RESEACT_MIN_MACRO_DT", "9.0"))
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
# One origin -> metaparameters + the degree-space parameters + the index base
# hydrostatic_dp and continuity_drift read. See `native_slice`: the .esm cannot
# tie the two currencies together, so nothing else may spell them out.
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 13), nlat = _env("NLAT", 7),
                           nlev = _env("NLEV", 72))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const T_END = T0 + SOLVE_SECS
const RTOL, ATOL, ATOL_T = 1e-4, 1e-9, 1e-6
const LU = LinearSolve.LUFactorization()

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
# Threading is the difference between ~40x realtime and ~1:1, so say which one
# this process is going to be BEFORE spending hours in it.
let nt = Threads.nthreads(),
    ext = POLYESTER_OK && Base.get_extension(EA, :EarthSciASTPolyesterExt) !== nothing
    say(@sprintf("    threads=%d  polyester_ext=%s", nt, ext))
    (nt > 1 && ext) ||
        say("    !! SLOW CONFIGURATION -- the threaded codegen tier is off. Relaunch " *
            "with `julia -t 8`, in an environment where Polyester is installed " *
            "(see HELPERS.md). A week run takes roughly 10x longer without it.")
end
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
docs = prepare_split_docs(MODEL; metaparameters = GRID_MP)
ff = reseact_forcing(CHEMDIR; ndays = NDAYS)
# When NLEV<72 the `lev` axis is shorter than the 72-entry hybrid table; slice the
# vertical coefs to match (a truncated column, k=1 = surface).
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

# --------------------------------------------------------------------------- #
# Cell-major reorder + block chemistry Jacobian (the block-diagonal structure is
# only contiguous once the state is permuted species-major -> cell-major).
# --------------------------------------------------------------------------- #
P = cellmajor_perm(run.var_map)
f_trans_cm! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
# AD is the default: the species here span thirty orders (NO titrates to ~1e-26
# through the night, OH lives at 1e-13) and no finite difference step is
# defensible across that range -- see block_jac.jl. RESEACT_JAC=fd forces the old
# finite-difference block Jacobian for an A/B.
const JACMODE = get(ENV, "RESEACT_JAC", "ad")
jac_cm!, mkjp = JACMODE == "fd" ? block_fd_jac(f_chem_cm!, P.NS, P.NC) :
                                  block_ad_jac(run.funcs[2], P.sm_of_cm, P.NS, P.NC)
say("    chem Jacobian: $(JACMODE == "fd" ? "finite difference" : "ForwardDiff (exact)")")
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

tgz(g, u, p, t) = (fill!(g, 0); nothing)   # chem tgrad is zero (autonomous over a step)
fc = SciMLBase.ODEFunction(f_chem_cm!; jac = jac_cm!, jac_prototype = mkjp(), tgrad = tgz)
inner_algs = (OrdinaryDiffEqSSPRK.SSPRK43(),
              OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false, linsolve = LU))
# Forcing refresh at cadence boundaries: rewrite the live buffers in
# run.merged_param from the discrete providers at time t, then rebuild the derived
# discrete caches -- the functional equivalent of run.cb's affect!, which the
# split solver cannot fire.
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
    sink = EA.build_zarr_sink(run.var_map, ZARR; output_times = out_times,
                              meta = EA.derive_output_meta(docs[1]))
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
# Interior and wall are reported separately. The open lateral walls prescribe
# inflow tracer mixing ratios (qbc_*) but take the AIR-MASS flux straight from the
# wind field, which is not constrained to close the interior column budget -- a
# different defect from an interior continuity bias, and one a pressure fixer
# would NOT correct. Which of the two moved is what decides where to look.
# --------------------------------------------------------------------------- #
const _PSk = "GEOSFP.GEOSFP_I3.PS"
const _dA = Float64.(ff.const_arrays["Transport3D.dA"])
const _dB = Float64.(ff.const_arrays["Transport3D.dB"])
const NLON_ = maximum(c[1] for c in P.cells); const NLAT_ = maximum(c[2] for c in P.cells)
"w_I3 for the I3 cadence: 3-hourly instantaneous anchored at 00:00Z."
w_I3(t) = (t - 10800.0 * floor(t / 10800.0)) / 10800.0
function continuity_drift(u, t)
    F = run.merged_param[_PSk]          # NATIVE [time, lat, lon], hPa
    w = w_I3(t)
    s2i = 0.0; ni = 0; s2w = 0.0; nw = 0; worsti = 0.0; wci = (0, 0, 0)
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
    end
    return (rms_int = sqrt(s2i / max(ni, 1)), rms_wall = sqrt(s2w / max(nw, 1)),
            worst_int = worsti, cell_int = wci)
end

rows = NamedTuple[]
push_row!(t, u, wall) = push!(rows, (t = t, hours = t / 3600,
    cos_sza = cos_sza(t, LAT_MID, LON_MID),
    o3_min = minimum(u[o3rng]), o3_mean = mean(u[o3rng]), o3_max = maximum(u[o3rng]),
    oh_max = maximum(u[ohrng]), no2_mean = mean(u[no2rng]),
    m_min = minimum(u[P.base_pos["Transport3D.m"]:P.NS:P.N]), wall = wall))
report(t, u, wall) = begin
    push_row!(t, u, wall); local r = rows[end]; local d = continuity_drift(u, t)
    say(@sprintf("   %6.2f  %7.4f  %9.5f  %9.5f  %9.5f  %10.3e  %9.3e  | %8.2e %9.2e %9.2e %-12s %6.1f",
        r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max, r.oh_max, r.no2_mean,
        d.rms_int, d.rms_wall, d.worst_int, string(d.cell_int), r.wall))
end

# --------------------------------------------------------------------------- #
# The macro-step loop.
# --------------------------------------------------------------------------- #
say("  t_hours  cos_sza     O3_min     O3_mean     O3_max      OH_max     NO2_mean  | rms_int  rms_wall  worst_int @cell_int    wall_s")
report(T0, u, 0.0); sink_put!(T0, u)

ts = time(); nT = nC = 0; ok = true; nstep = 0; nbisect = 0
t = T0
while t < T_END - 1e-9
    global nstep += 1
    tnext = min(t + MACRO_DT, T_END)
    res = lie_trotter_solve_bisect(f_trans_cm!, fc, u, (t, tnext), run.p, inner_algs;
        macro_dt = tnext - t, min_macro_dt = MIN_MACRO_DT,
        reltols = (RTOL, RTOL), abstols = (ATOL_T, ATOL),
        refresh = refresh_forcing, forcing_tstops = run.tstops, clamp_nonneg = true,
        on_bisect = (a, b) -> (global nbisect += 1;
                               say(@sprintf("    bisecting [%.0f, %.0f]", a, b))))
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
        # Dump the raw CELL-MAJOR state that failed. A failure reached after hours
        # of simulation is otherwise reproducible only by re-simulating to it.
        # Dump the PRE-STEP state `u`, not `res.u`: `lie_trotter_solve` returns
        # `copy(integ.u)` taken AFTER the failed `step!` and the clamp, so `res.u`
        # is where the integrator gave up -- past the transition, and restarting
        # from it does NOT reproduce the failure. `u` is the last state a macro
        # step accepted, so restarting there re-runs exactly the step that failed.
        # Both are written: `u` to restart from, `res.u` because it is what the
        # diagnostic above prints.
        if !isempty(STATEDUMP)
            open(STATEDUMP, "w") do io
                write(io, Int64(P.NS), Int64(P.NC), Float64(t))
                write(io, u)          # pre-step, RESTARTABLE
                write(io, res.u)      # post-failure, matches the printout above
            end
            open(STATEDUMP * ".names", "w") do io
                println(io, "# NS=$(P.NS) NC=$(P.NC) t=$t  layout: cell-major, u[(c-1)*NS+s]")
                for (nm, b) in sort(collect(P.base_pos), by = last)
                    println(io, b, "\t", nm)
                end
            end
            say("  !! failing state dumped -> $STATEDUMP")
        end
        say("  !! stopping at t=$t (see above); partial series below")
        break
    end
    global u = res.u; global nT += res.naT; global nC += res.naC
    global t = tnext
    report(t, u, time() - ts)
    if nstep % OUT_EVERY == 0
        sink_put!(t, u)
        sink === nothing || (nstep % (OUT_EVERY * FLUSH_EVERY) == 0 && EA.sink_flush!(sink))
    end
end
solve_s = time() - ts
if sink !== nothing
    EA.sink_flush!(sink); EA.sink_close!(sink)
    say("  gridded output committed: $ZARR")
end

o3m = [r.o3_mean for r in rows]
say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d window_h=%.2f macro_dt=%.0f nmacro=%d nT=%d nC=%d nbisect=%d solve_s=%.1f",
    LABEL, P.NC, P.N, P.NS, SOLVE_SECS / 3600, MACRO_DT, length(rows) - 1, nT, nC, nbisect, solve_s))
say(@sprintf("DIURNAL O3_mean: start=%.5f min=%.5f max=%.5f end=%.5f  peak-to-trough=%.5f ppb  ok=%s",
    o3m[1], minimum(o3m), maximum(o3m), o3m[end], maximum(o3m) - minimum(o3m),
    ok && all(isfinite, u)))

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
