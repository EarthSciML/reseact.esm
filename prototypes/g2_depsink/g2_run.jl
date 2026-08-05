# G2 de-risk: does an additive `couple` connector add -k*c to a reaction-system species ODE?
# Simulate SuperFast for 1 s with and without the deposition-sink coupling and compare.
import Pkg
ENV_DIR = get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "run-model-jl")))
Pkg.activate(ENV_DIR; io=devnull)
# stiff solver for gas-phase chemistry (radical chemistry is stiff)
try
    @eval import OrdinaryDiffEqRosenbrock
catch
    Pkg.add("OrdinaryDiffEqRosenbrock"; io=devnull)
    @eval import OrdinaryDiffEqRosenbrock
end
using EarthSciAST
const EA = EarthSciAST

HERE = @__DIR__
# NOTE (2026-08-04): the tree-walk RHS IS ForwardDiff-compatible; see block_jac.jl
# `block_ad_jac`. FD is kept here only as a historical A/B.
alg = OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false)

function run(path; params=Dict{String,Float64}())
    f = EA.load(path)
    # flatten with base_path = the assembly file's dir so refs resolve relative to
    # the file, not the process CWD (simulate() defaults base_path=".").
    fs = EA.flatten(f; base_path=dirname(path))
    sim = EA.simulate(fs, (0.0, 1.0); alg=alg, parameters=params,
                       reltol=1e-6, abstol=1e-9, saveat=[0.0, 1.0])
    return sim
end

# exaggerated deposition rates (1/s) for a visible 1 s effect, set on the ref'd sink model
DEP_PARAMS = Dict(
    "SuperFastDepositionSink.k_O3"   => 0.05,
    "SuperFastDepositionSink.k_NO2"  => 0.03,
    "SuperFastDepositionSink.k_HNO3" => 0.08,
    "SuperFastDepositionSink.k_H2O2" => 0.05,
    "SuperFastDepositionSink.k_HCHO" => 0.04,
    "SuperFastDepositionSink.k_wet"  => 0.02,
)

# map species name -> final value from a sim, tolerant of "SuperFast.O3", "O3", "SuperFast₊O3(t)" etc.
function finals(sim)
    d = Dict{String,Float64}()
    for (k, idx) in pairs(sim.var_map)
        ks = string(k)
        d[ks] = sim.u[end][idx]
    end
    return d
end
getsp(d, sp) = begin
    # find the key whose base name matches the species
    hits = filter(k -> occursin(sp, k), collect(keys(d)))
    isempty(hits) && return NaN
    # prefer an exact-ish match ending in the species token
    exact = filter(k -> endswith(replace(k, r"[\(\)t]"=>""), sp) || endswith(k, ".$sp") || k == sp || endswith(k, "₊$sp"), hits)
    d[isempty(exact) ? first(sort(hits; by=length)) : first(exact)]
end

println("== simulating g2_nodep.esm (baseline) ==")
snd = run(joinpath(HERE, "g2_nodep.esm"))
println("  success=", snd.success, " retcode=", snd.retcode, " nstates=", length(snd.u[end]))
println("  var_map keys (first 20): ", first(sort(string.(collect(keys(snd.var_map)))), 20))

println("== simulating g2_dep.esm (with deposition) ==")
sd = run(joinpath(HERE, "g2_dep.esm"); params=DEP_PARAMS)
println("  success=", sd.success, " retcode=", sd.retcode, " nstates=", length(sd.u[end]))

fnd = finals(snd); fd = finals(sd)
species = ["O3","NO2","NO","HNO3","H2O2","CH2O","CH3OOH","CO","ISOP","OH","HO2","CH3O2"]
println("\nspecies    nodep_final      dep_final        diff(dep-nodep)   dep_active?")
for sp in species
    a = getsp(fnd, sp); b = getsp(fd, sp)
    diff = b - a
    active = abs(diff) > 1e-9 ? "YES" : "-"
    println(rpad(sp,9), rpad(string(round(a,sigdigits=8)),16), rpad(string(round(b,sigdigits=8)),16),
            rpad(string(round(diff,sigdigits=6)),18), active)
end
println("\nExpected: species with a deposition sink (O3,NO2,NO,HNO3,H2O2,CH2O,CH3OOH) should be LOWER with dep (diff<0).")
println("Species with no dep sink (CO,ISOP,OH,HO2,CH3O2) should show ~0 direct diff (only indirect chemical feedback).")
