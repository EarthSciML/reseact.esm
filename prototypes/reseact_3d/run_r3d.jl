#!/usr/bin/env julia
# Stage A first light: the verified 7x7x7 PPM transport driven by REAL GEOS-FP 4x5,
# sliced natively over the central US. t=0 := 2013-01-01T00:00Z.
import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io=devnull)
using EarthSciAST; import OrdinaryDiffEqTsit5; import DiffEqCallbacks
import SciMLBase   # phase 4: `solve` / `successful_retcode` are SciMLBase's own
using EarthSciIO, Printf, JSON3
const EA = EarthSciAST

const MODEL = joinpath(@__DIR__, "reseact_3d.esm")
# t=0 := 2013-01-01T00:00Z. Start at 18:00Z: every collection has a record on BOTH
# sides there (A1's first is 00:30Z, so t=0 itself is un-bracketable), and it is
# daytime over CONUS — which Stage C's photolysis will want.
const T0    = 64800.0
const T_END = T0 + (length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0)
const BASE  = "https://geos-chem.s3-us-west-2.amazonaws.com/GEOS_4x5/GEOS_FP/2013/01/GEOSFP.20130101"

r = EA.validate(MODEL)
println("validate: is_valid=$(r.is_valid) schema=$(length(r.schema_errors)) struct=$(length(r.structural_errors))")
for e in r.structural_errors[1:min(8,end)]; println("  ", e.error_type, " @ ", e.path, " :: ", e.message); end
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
    "GEOSFP.PBLH"    => mk("A1",    "PBLH",  1800.0,  3600.0, 24))

# The three cadence phases must match the providers above, or the blend weight
# would ramp out of phase with the records the provider actually loaded.
params = Dict("GEOSFP.t_interp_ref_I3" =>    0.0, "GEOSFP.dt_interp_I3" => 10800.0,
              "GEOSFP.t_interp_ref_A3" => 5400.0, "GEOSFP.dt_interp_A3" => 10800.0,
              "GEOSFP.t_interp_ref_A1" => 1800.0, "GEOSFP.dt_interp_A1" =>  3600.0)

co = JSON3.read(read("hybrid_coefs.json", String))
const_arrays = Dict{String,Any}("Transport3D.dA" => Float64.(co.dA),
                                "Transport3D.dB" => Float64.(co.dB))

insp = EA.BuildInspection()
t0 = time()
# phase 4: build once, then solve.
prob = EA.esm_problem(EA.load_path(MODEL), (T0, T_END);
                      providers=providers, p=params,
                      const_arrays=const_arrays, inspect=insp)
sim = SciMLBase.solve(prob, OrdinaryDiffEqTsit5.Tsit5(); saveat=[T0, T_END])
@printf("simulate: %.1f s  success=%s retcode=%s nstates=%d\n",
        time()-t0, SciMLBase.successful_retcode(sim), sim.retcode,
        length(sim.u[1]))
SciMLBase.successful_retcode(sim) || error("solver failed: $(sim.retcode)")

# --- what forcing did the model actually see? (read it out of the build)
for k in ["Transport3D.PS", "Transport3D.PBLH", "Transport3D.Mx", "Transport3D.My", "Transport3D.Mz", "Transport3D.dp"]
    a = get(insp.setup_arrays, k, nothing)
    a === nothing && continue
    @printf("  %-18s min=%12.4g  max=%12.4g  mean=%12.4g\n", split(k,'.')[2],
            minimum(a), maximum(a), sum(a)/length(a))
end

# --- CWC gate: q = mq/m must stay EXACTLY 1, dev EXACTLY 0.
vm = prob.var_map; names = collect(keys(vm))
cells(pat) = Dict(n[findfirst('[', n):end] => vm[n] for n in names if occursin(".$pat[", n))
Mi, MQi, Di = cells("m"), cells("mq"), cells("dev")
for (lbl, uu) in (("t=18:00Z", sim.u[1]), ("+$(T_END-T0)s", sim.u[end]))
    ms = [uu[i] for i in values(Mi)]
    @printf("  %-8s m in [%.4f, %.4f] Pa   all positive: %s\n", lbl, minimum(ms), maximum(ms), all(>(0), ms))
end
uu = sim.u[end]
qerr = maximum(abs(uu[MQi[k]] / uu[Mi[k]] - 1) for k in keys(Mi))
derr = maximum(abs(uu[Di[k]]) for k in keys(Di))
@printf("CWC GATE  max|q-1| = %.3e   max|dev| = %.3e   -> %s\n",
        qerr, derr, (qerr == 0.0 && derr == 0.0) ? "PASS (bit-exact)" : "FAIL")
