import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io=devnull)
using EarthSciAST; import OrdinaryDiffEqTsit5; import DiffEqCallbacks
using EarthSciIO, Printf, JSON3
const EA = EarthSciAST
const T0 = 64800.0
const BASE = "https://geos-chem.s3-us-west-2.amazonaws.com/GEOS_4x5/GEOS_FP/2013/01/GEOSFP.20130101"
r = EA.validate("probe3.esm"); println("validate: is_valid=$(r.is_valid) struct=$(length(r.structural_errors))")
for e in r.structural_errors[1:min(5,end)]; println("  ", e.error_type, " @ ", e.path, " :: ", e.message); end

cache = EarthSciIO.Cache()
mk(coll,var,phase,dt,n) = EarthSciIO.discrete_provider(cache, "$BASE.$coll.4x5.nc",
        [phase+dt*k for k in 0:(n-1)]; format="netcdf", variables=[var], time_dim="time", records_per_sample=2)
providers = Dict(
    "GEOSFP.PS"       => mk("I3","PS",0.0,10800.0,8),
    "GEOSFP.U"     => mk("A3dyn","U",5400.0,10800.0,8),
    "GEOSFP.V"     => mk("A3dyn","V",5400.0,10800.0,8),
    "GEOSFP.OMEGA" => mk("A3dyn","OMEGA",5400.0,10800.0,8),
    "GEOSFP.PBLH"     => mk("A1","PBLH",1800.0,3600.0,24))
params = Dict("GEOSFP.t_interp_ref_I3"=>0.0,"GEOSFP.dt_interp_I3"=>10800.0,
              "GEOSFP.t_interp_ref_A3"=>5400.0,"GEOSFP.dt_interp_A3"=>10800.0,
              "GEOSFP.t_interp_ref_A1"=>1800.0,"GEOSFP.dt_interp_A1"=>3600.0)
co = JSON3.read(read("hybrid_coefs.json", String))
ca = Dict{String,Any}("Transport3D.dA"=>Float64.(co.dA), "Transport3D.dB"=>Float64.(co.dB))

t0=time()
sim = EA.simulate(EA.load("probe3.esm"), (T0, T0+1.0); alg=OrdinaryDiffEqTsit5.Tsit5(),
                  saveat=[T0,T0+1.0], providers=providers, parameters=params, const_arrays=ca)
@printf("simulate: %.1f s  success=%s\n", time()-t0, sim.success)

NAMES = ["pMx","pMy","pMz","pPS","pdp","pBL","pMxw","pMys"]
ref = JSON3.read(read("probe3_ref.json", String))
vm = sim.var_map
println("\n  probe      model                 reference             rel.err")
worst = 0.0
for (n,nm) in enumerate(NAMES)
    k = "Transport3D.y[$n]"; haskey(vm,k) || (println("  $nm MISSING"); continue)
    got = sim.u[end][vm[k]] - sim.u[1][vm[k]]
    want = Float64(ref[Symbol(nm)])
    rel = want == 0 ? abs(got) : abs(got-want)/abs(want)
    global worst = max(worst, rel)
    @printf("  %-6s %20.10g  %20.10g  %9.2e %s\n", nm, got, want, rel, rel<1e-9 ? "OK" : "<-- MISMATCH")
end
@printf("\nworst rel err = %.3e  -> %s\n", worst, worst < 1e-9 ? "FORCING VERIFIED against the netCDF" : "FAIL")
