#!/usr/bin/env julia
# ===========================================================================
# tools/continuity_residual.jl -- WHY does m still drift from dp after the fixer?
# ===========================================================================
# The pressure fixer is supposed to make the discrete air-mass equation
# reproduce the observed surface-pressure tendency exactly:
#
#     dm_k/dt  ==  d(dp_k)/dt  ==  dB[k] * dPSdt
#
# (dp_k = dA[k] + dB[k]*PS, and dA is constant, so only the dB term survives.)
# A residual ~1.9e-6/s survived, INDEPENDENT of step size -- so it is not time
# integration error, it is the RIGHT-HAND SIDE itself. This script evaluates the
# transport RHS ONCE, at t = T0, and compares dm/dt against dB[k]*dPSdt cell by
# cell. No solver, no stepping: whatever shows up here is pure discretisation.
#
# It also reports the residual RELATIVE to dp, per level. That ratio is the
# discriminator between the two candidate explanations:
#
#   * a boundary/stencil difference between the `divh` OBSERVED and the
#     divergence the m equation lowers on its own  ->  residual concentrated in
#     the boundary regions, with no clean vertical structure;
#   * the column correction surviving uncancelled  ->  residual EXACTLY
#     proportional to dp_k, i.e. residual/dp CONSTANT down the whole column,
#     because the fixer distributes the column imbalance by dp weight.
#
# Env: RESEACT_MODEL, RESEACT_NLON/NLAT/NLEV, RESEACT_T0
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)
haskey(ENV, "ESS_KERNEL_CLASS_MERGE_DISABLE") || (ENV["ESS_KERNEL_CLASS_MERGE_DISABLE"] = "1")
using Printf, Statistics, Logging
using SciMLBase, DiffEqCallbacks     # activates EarthSciASTDataRefreshExt

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
# Grid defaults follow the model's own metaparameters (CONUS 13x7x72). The
# slice ORIGIN is not spelled here at all: it comes from the .esm defaults
# (LON0/LAT0) and the matching degree-space twins `native_slice()` applies
# through `build_split_run`, so the two currencies cannot drift apart here.
const NLON  = parse(Int, get(ENV, "RESEACT_NLON", "13"))
const NLAT  = parse(Int, get(ENV, "RESEACT_NLAT", "7"))
const NLEV  = parse(Int, get(ENV, "RESEACT_NLEV", "72"))
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "64800"))

say("model  : $MODEL")
say("grid   : $(NLON)x$(NLAT)x$(NLEV)   t0 = $T0 s")

tb = time()
docs = Logging.with_logger(Logging.NullLogger()) do
    prepare_split_docs(MODEL; metaparameters = Dict("NLON" => NLON, "NLAT" => NLAT, "NLEV" => NLEV))
end
ff = reseact_forcing(CHEMDIR)
ff = merge(ff, (; const_arrays = GridResize.slice_hybrid_coefs(ff.const_arrays, NLEV)))
run = build_split_run(docs, (T0, T0 + 1.0); providers = ff.providers,
                      parameters = ff.parameters, const_arrays = ff.const_arrays)
say(@sprintf("build  : %.1f s", time() - tb))

P  = cellmajor_perm(run.var_map)
ft! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
u  = run.u0[P.sm_of_cm]
foreach(d -> d.materialize!(), run.dms)

# Seed m from the real hydrostatic thickness so the state is EXACTLY consistent
# at t0 -- the residual measured below is then the RHS defect alone.
dp0 = hydrostatic_dp(run.merged_param, ff.const_arrays, T0; slice = run.slice)
mb  = P.base_pos["Transport3D.m"]
for c in P.cells
    u[(P.cell_pos[c] - 1) * P.NS + mb] = dp0(c[1], c[2], c[3])
end

du = similar(u); ft!(du, u, run.p, T0)

# --- the target: d(dp_k)/dt = dB[k] * dPSdt, with dPSdt read the same way the
#     model reads it (record 2 minus record 1 of the bracketing I3 pair, hPa->Pa).
const dA = Float64.(ff.const_arrays["Transport3D.dA"])
const dB = Float64.(ff.const_arrays["Transport3D.dB"])
F = run.merged_param["GEOSFP.GEOSFP_I3.PS"]
# Native index base from the RUN's own slice, not literals: the origin is a
# metaparameter now, and a diagnostic that reads a different column than the
# model did would report the terrain difference as a continuity residual.
const LON0_, LAT0_ = run.slice.lon0, run.slice.lat0
dPSdt(i, j) = 100.0 * (F[2, LAT0_ + j, LON0_ + i] - F[1, LAT0_ + j, LON0_ + i]) / 10800.0

NL = maximum(c[1] for c in P.cells); NA = maximum(c[2] for c in P.cells)
isw(c) = c[1] == 1 || c[1] == NL || c[2] == 1 || c[2] == NA

# per-cell residual and its dp-relative form
res   = Dict{NTuple{3,Int},Float64}()
rel   = Dict{NTuple{3,Int},Float64}()
for c in P.cells
    i, j, k = c
    dmdt = du[(P.cell_pos[c] - 1) * P.NS + mb]
    tgt  = dB[k] * dPSdt(i, j)
    res[c] = dmdt - tgt
    rel[c] = res[c] / dp0(i, j, k)
end

allrel = [rel[c] for c in P.cells]
say(@sprintf("\nRHS residual  dm/dt - dB[k]*dPSdt      max|abs| = %.4e Pa/s", maximum(abs, values(res))))
say(@sprintf("RELATIVE (per dp)                     max|rel| = %.4e 1/s   rms = %.4e 1/s",
             maximum(abs, allrel), sqrt(mean(abs2, allrel))))

# --- Is the relative residual CONSTANT down each column? That is the signature
#     of the dp-weighted column correction surviving; anything else points at the
#     stencils. Report spread within a column, not just the extremes.
say("\nper-column relative residual (1/s): is it k-independent?")
say("   cell(i,j)   wall?    mean_k        min_k        max_k     spread/mean")
for j in (1, (NA + 1) ÷ 2, NA), i in (1, (NL + 1) ÷ 2, NL)
    col = [rel[(i, j, k)] for k in 1:NLEV]
    mu = mean(col); lo = minimum(col); hi = maximum(col)
    say(@sprintf("   (%2d,%2d)     %-5s  %12.5e %12.5e %12.5e   %10.3e",
                 i, j, isw((i, j, 1)), mu, lo, hi, mu == 0 ? 0.0 : (hi - lo) / abs(mu)))
end

# --- vertical profile through the domain centre, plus where the ABSOLUTE
#     residual sits (it should follow dp if the correction is dp-weighted).
ic, jc = (NL + 1) ÷ 2, (NA + 1) ÷ 2
say("\nprofile at centre column ($ic,$jc):")
say("     k       dp (Pa)     dm/dt (Pa/s)   target (Pa/s)   residual        rel (1/s)")
for k in unique(clamp.(round.(Int, range(1, NLEV, length = 12)), 1, NLEV))
    c = (ic, jc, k)
    say(@sprintf("   %3d  %12.4f  %14.6e  %14.6e  %13.5e  %12.5e", k, dp0(ic, jc, k),
        du[(P.cell_pos[c] - 1) * P.NS + mb] , dB[k] * dPSdt(ic, jc), res[c], rel[c]))
end

# --- interior vs wall: a stencil/BC explanation would separate these sharply.
ri = [rel[c] for c in P.cells if !isw(c)]
rw = [rel[c] for c in P.cells if isw(c)]
say(@sprintf("\ninterior rms = %.4e 1/s (n=%d)   wall rms = %.4e 1/s (n=%d)",
             sqrt(mean(abs2, ri)), length(ri), sqrt(mean(abs2, rw)), length(rw)))

# --- the predicted correction: (-dPSdt - divh_col)/dp_col, one number per column.
#     Recover divh_col from the residual itself is circular, so instead recover it
#     from the COLUMN SUM of dm/dt, which the no-flux vertical BC makes equal to
#     -divh_col regardless of Mz:  sum_k dm_k/dt = -(divh_col + Mz[top] - Mz[1]).
say("\ncolumn closure check:  sum_k dm/dt  vs  dPSdt  (they differ by exactly the")
say("column imbalance the fixer has to absorb somewhere)")
say("   cell(i,j)   sum_k dm/dt      dPSdt        imbalance     imbalance/dp_col")
for j in (1, (NA + 1) ÷ 2, NA), i in (1, (NL + 1) ÷ 2, NL)
    s = sum(du[(P.cell_pos[(i, j, k)] - 1) * P.NS + mb] for k in 1:NLEV)
    dpc = sum(dp0(i, j, k) for k in 1:NLEV)
    say(@sprintf("   (%2d,%2d)   %14.6e %14.6e %13.5e   %12.5e",
                 i, j, s, dPSdt(i, j), s - dPSdt(i, j), (s - dPSdt(i, j)) / dpc))
end
say("\nDONE continuity_residual")
