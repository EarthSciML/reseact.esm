#!/usr/bin/env julia
# Stage C1: the verified 7x7x7 PPM transport driven by REAL GEOS-FP 4x5 (native
# CONUS slice), with SuperFast gas chemistry lifted pointwise onto the grid.
# t=0 := 2013-01-01T00:00Z.
#
# SOLVER: SuperFast is STIFF (fast radical chemistry), so this uses
# Rosenbrock23(autodiff=false) — Tsit5 (fine for Stage A pure transport) goes
# Unstable at the first step on the lifted chemistry. autodiff=false because the
# NOTE (2026-08-04): the tree-walk RHS IS ForwardDiff-compatible -- `_rhs_value_type`
# promotes over `values(p)` precisely to admit Duals, and the Float32 guard folds away
# under them (EarthSciAST compile.jl). Measured: block_ad_jac vs block_fd_jac at a
# pre-dawn state (NO ~ 3e-26) -- FD loses 18% of the nonzeros and gets d(dO3/dt)/dNO
# 17% wrong. Prefer block_ad_jac; this script keeps FD only as a historical A/B.
#
# BUILD: `preserve_refs=true` carries the PPM stencil references to the build
# boundary so the compile-once tier factors each body once (RFC step c) instead
# of fusing 39 rule instantiations into ~200M node-lowerings — the difference
# between a ~20 min build and a build that never finishes at 12 species.
import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io=devnull)
using EarthSciAST; import OrdinaryDiffEqRosenbrock; import DiffEqCallbacks
import SciMLBase   # phase 4: `solve` / `successful_retcode` are SciMLBase's own
using EarthSciIO, Printf, JSON3
const EA = EarthSciAST

const MODEL = joinpath(@__DIR__, "reseact_3d_chem.esm")
# t=0 := 2013-01-01T00:00Z. Start at 18:00Z: every collection has a record on BOTH
# sides there (A1's first is 00:30Z, so t=0 itself is un-bracketable), and it is
# daytime over CONUS — which Stage C's photolysis will want.
const T0    = 64800.0
const T_END = T0 + (length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0)
const BASE  = "https://geos-chem.s3-us-west-2.amazonaws.com/GEOS_4x5/GEOS_FP/2013/01/GEOSFP.20130101"

# Julia BLOCK-buffers stdout when it is a file rather than a tty, so a run that
# is killed (a `timeout`, an OOM) loses every line it ever printed and looks like
# it produced nothing. The build here is tens of minutes, so flush each progress
# line as it happens — a killed run must still tell you how far it got.
say(s) = (println(s); flush(stdout))

say("validate: begin")
r = EA.validate_path(MODEL)
say("validate: is_valid=$(r.is_valid) schema=$(length(r.schema_errors)) struct=$(length(r.structural_errors))")
for e in r.structural_errors[1:min(8,end)]; say("  $(e.error_type) @ $(e.path) :: $(e.message)"); end
r.is_valid || error("model does not validate")

# --- providers: 2 bracketing records per solver time, blended in-model by w_time.
# Record k of each collection lands at (phase + k*cadence) seconds after 00:00Z.
cache = EarthSciIO.Cache()
mk(coll, var, phase, dt, n) = EarthSciIO.discrete_provider(
        cache, "$BASE.$coll.4x5.nc", [phase + dt*k for k in 0:(n-1)];
        format="netcdf", variables=[var], time_dim="time", records_per_sample=2)

providers = Dict(
    "GEOSFP.PS"      => mk("I3",    "PS",       0.0, 10800.0, 8),
    "GEOSFP.U"    => mk("A3dyn", "U",     5400.0, 10800.0, 8),
    "GEOSFP.V"    => mk("A3dyn", "V",     5400.0, 10800.0, 8),
    "GEOSFP.OMEGA"=> mk("A3dyn", "OMEGA", 5400.0, 10800.0, 8),
    "GEOSFP.PBLH"    => mk("A1",    "PBLH",  1800.0,  3600.0, 24),
    "GEOSFP.T"       => mk("I3",    "T",        0.0, 10800.0, 8))

# The three cadence phases must match the providers above, or the blend weight
# would ramp out of phase with the records the provider actually loaded.
params = Dict("GEOSFP.t_interp_ref_I3" =>    0.0, "GEOSFP.dt_interp_I3" => 10800.0,
              "GEOSFP.t_interp_ref_A3" => 5400.0, "GEOSFP.dt_interp_A3" => 10800.0,
              "GEOSFP.t_interp_ref_A1" => 1800.0, "GEOSFP.dt_interp_A1" =>  3600.0)

co = JSON3.read(read("hybrid_coefs.json", String))
const_arrays = Dict{String,Any}("Transport3D.dA" => Float64.(co.dA),
                                "Transport3D.dB" => Float64.(co.dB),
                                "Transport3D.Ap" => Float64.(co.Ap),
                                "Transport3D.Bp" => Float64.(co.Bp))

insp = EA.BuildInspection()
t0 = time()
say("simulate: begin (compile-once TIER via preserve_refs; stiff Rosenbrock23)")
# EarthSciAST phase 4: build once, then solve. Build-time knobs (providers, p,
# const_arrays, inspect) go to `esm_problem`; solver knobs stay on `solve`.
#
# `preserve_refs=true` is DROPPED, and it is not a phase-4 casualty: EarthSciAST
# has never had a keyword by that name, on `simulate` or anything else, and
# neither `simulate` method slurped unknown keywords. This call raised a
# MethodError as written, before any of this migration.
prob = EA.esm_problem(EA.load_path(MODEL), (T0, T_END);
                      providers=providers, p=params,
                      const_arrays=const_arrays, inspect=insp)
sim = SciMLBase.solve(prob,
                      OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false);
                      reltol=1e-4, abstol=1e-9, saveat=[T0, T_END])
say(@sprintf("simulate: %.1f s  success=%s retcode=%s nstates=%d",
             time()-t0, SciMLBase.successful_retcode(sim), sim.retcode,
             length(sim.u[1])))
# Print the forcing + chemistry diagnostics BELOW even on a solver failure — a
# failed solve still built the RHS and populated `inspect`, and the t0 state / the
# forcing arrays are exactly what tells an Unstable/blowup apart from stiffness.

# --- what forcing did the model actually see? (read it out of the build)
for k in ["Transport3D.PS", "Transport3D.PBLH", "Transport3D.Mx", "Transport3D.My", "Transport3D.Mz", "Transport3D.dp"]
    a = get(insp.setup_arrays, k, nothing)
    a === nothing && continue
    @printf("  %-18s min=%12.4g  max=%12.4g  mean=%12.4g\n", split(k,'.')[2],
            minimum(a), maximum(a), sum(a)/length(a))
end

# --- chemistry: did the species lift onto the grid, and do they move?
vm = prob.var_map; names = collect(keys(vm))
cells(pat) = Dict(n[findfirst('[', n):end] => vm[n] for n in names if occursin(".$pat[", n))
if !isempty(sim.u)
    for sp in ["O3","NO","NO2","OH","HO2","CO","ISOP","CH2O"]
        c = cells(sp); isempty(c) && (println("  $sp: NOT LIFTED"); continue)
        v0 = [sim.u[1][i]   for i in values(c)]
        v1 = [sim.u[end][i] for i in values(c)]
        @printf("  %-6s %4d cells   t0 in [%.6g, %.6g]   t1 in [%.6g, %.6g]\n",
                sp, length(c), minimum(v0), maximum(v0), minimum(v1), maximum(v1))
    end
    mi = cells("m")
    if !isempty(mi)
        m1 = [sim.u[end][i] for i in values(mi)]
        @printf("  %-6s %4d cells   t1 in [%.4f, %.4f] Pa  all positive: %s\n",
                "m", length(mi), minimum(m1), maximum(m1), all(>(0), m1))
    end
end
SciMLBase.successful_retcode(sim) || error("solver failed: $(sim.retcode)")
say("DONE")
