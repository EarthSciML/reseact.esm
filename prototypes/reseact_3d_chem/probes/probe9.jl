import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "..", "run-model-jl"))); io=devnull)
using EarthSciAST; import OrdinaryDiffEqTsit5; import DiffEqCallbacks; using EarthSciIO, JSON3, Printf
import SciMLBase   # phase 4: `solve` / `successful_retcode` are SciMLBase's own
const EA = EarthSciAST
const T0 = 64800.0
const BASE = "https://geos-chem.s3-us-west-2.amazonaws.com/GEOS_4x5/GEOS_FP/2013/01/GEOSFP.20130101"
r = EA.validate_path("probe9.esm"); println("validate: is_valid=$(r.is_valid) struct=$(length(r.structural_errors))")
for e in r.structural_errors[1:min(3,end)]; println("  ", e.error_type, " :: ", e.message); end
cache = EarthSciIO.Cache()
prov = EarthSciIO.discrete_provider(cache, "$BASE.I3.4x5.nc", [10800.0*k for k in 0:7];
        format="netcdf", variables=["PS"], time_dim="time", records_per_sample=2)
co = JSON3.read(read("hybrid_coefs.json", String))
ca = Dict{String,Any}("Transport3D.dA"=>Float64.(co.dA), "Transport3D.dB"=>Float64.(co.dB))
t0 = time()
# phase 4: build once, then solve; the state map lives on the problem.
prob = EA.esm_problem(EA.load_path("probe9.esm"), (T0, T0+1.0);
        providers=Dict("GEOSFP.PS"=>prov),
        p=Dict("GEOSFP.t_interp_ref_I3"=>0.0, "GEOSFP.dt_interp_I3"=>10800.0),
        const_arrays=ca)
sim = SciMLBase.solve(prob, OrdinaryDiffEqTsit5.Tsit5(); saveat=[T0, T0+1.0])
@printf("simulate: %.1f s  success=%s nstates=%d\n", time()-t0, SciMLBase.successful_retcode(sim), length(sim.u[1]))
println("Chem.A cells: ", count(k->startswith(k, "Chem.A["), collect(keys(prob.var_map))))
