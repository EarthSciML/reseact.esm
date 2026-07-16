#!/usr/bin/env julia
# Stage A first light: the verified 7x7x7 PPM transport driven by REAL GEOS-FP 4x5,
# sliced natively over the central US. t=0 := 2013-01-01T00:00Z.
import Pkg; Pkg.activate("/Users/ctessum/code/earthsciml/reseact.esm/run-model-jl"; io=devnull)
using EarthSciAST; import OrdinaryDiffEqTsit5; import DiffEqCallbacks
using EarthSciIO, Printf, JSON3
const EA = EarthSciAST

const MODEL = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/reseact_3d_chem/reseact_3d_chem.esm"
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
    "GEOSFP.GEOSFP_I3.PS"      => mk("I3",    "PS",       0.0, 10800.0, 8),
    "GEOSFP.GEOSFP_A3dyn.U"    => mk("A3dyn", "U",     5400.0, 10800.0, 8),
    "GEOSFP.GEOSFP_A3dyn.V"    => mk("A3dyn", "V",     5400.0, 10800.0, 8),
    "GEOSFP.GEOSFP_A3dyn.OMEGA"=> mk("A3dyn", "OMEGA", 5400.0, 10800.0, 8),
    "GEOSFP.GEOSFP_A1.PBLH"    => mk("A1",    "PBLH",  1800.0,  3600.0, 24),
    "GEOSFP.GEOSFP_I3.T"       => mk("I3",    "T",        0.0, 10800.0, 8))

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
sim = EA.simulate(EA.load(MODEL), (T0, T_END);
                  alg=OrdinaryDiffEqTsit5.Tsit5(), saveat=[T0, T_END],
                  providers=providers, parameters=params,
                  const_arrays=const_arrays, inspect=insp)
@printf("simulate: %.1f s  success=%s retcode=%s nstates=%d\n",
        time()-t0, sim.success, sim.retcode, length(sim.u[1]))
sim.success || error("solver failed: $(sim.retcode) — $(sim.message)")

# --- what forcing did the model actually see? (read it out of the build)
for k in ["Transport3D.PS", "Transport3D.PBLH", "Transport3D.Mx", "Transport3D.My", "Transport3D.Mz", "Transport3D.dp"]
    a = get(insp.setup_arrays, k, nothing)
    a === nothing && continue
    @printf("  %-18s min=%12.4g  max=%12.4g  mean=%12.4g\n", split(k,'.')[2],
            minimum(a), maximum(a), sum(a)/length(a))
end

# --- chemistry: did the species lift onto the grid, and do they move?
vm = sim.var_map; names = collect(keys(vm))
cells(pat) = Dict(n[findfirst('[', n):end] => vm[n] for n in names if occursin(".$pat[", n))
for sp in ["O3","NO","NO2","OH","HO2","CO","ISOP","CH2O"]
    c = cells(sp); isempty(c) && (println("  $sp: NOT LIFTED"); continue)
    v0 = [sim.u[1][i]   for i in values(c)]
    v1 = [sim.u[end][i] for i in values(c)]
    @printf("  %-6s %4d cells   t0 = %-12.6g   t1 in [%.6g, %.6g]\n",
            sp, length(c), v0[1], minimum(v1), maximum(v1))
end
mi = cells("m")
if !isempty(mi)
    m1 = [sim.u[end][i] for i in values(mi)]
    @printf("  %-6s %4d cells   t1 in [%.4f, %.4f] Pa  all positive: %s\n",
            "m", length(mi), minimum(m1), maximum(m1), all(>(0), m1))
end
