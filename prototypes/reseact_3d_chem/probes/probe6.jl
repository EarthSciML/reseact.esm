import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "..", "run-model-jl"))); io=devnull)
using EarthSciAST; import OrdinaryDiffEqTsit5; using JSON3, Printf
const EA = EarthSciAST
r = EA.validate("probe6.esm"); println("validate: is_valid=$(r.is_valid) struct=$(length(r.structural_errors))")
for e in r.structural_errors[1:min(5,end)]; println("  ", e.error_type, " @ ", e.path, " :: ", e.message); end
f = EA.load("probe6.esm"); fs = EA.flatten(f)
for n in ["Chem.A","Chem.B","Transport3D.m"]
    haskey(fs.state_variables,n) && println("  $n :: shape=", fs.state_variables[n].shape)
end
co = JSON3.read(read("hybrid_coefs.json", String))
ca = Dict{String,Any}("Transport3D.dA"=>Float64.(co.dA), "Transport3D.dB"=>Float64.(co.dB))
t0=time()
sim = EA.simulate(fs, (0.0,1.0); alg=OrdinaryDiffEqTsit5.Tsit5(), saveat=[0.0,1.0], const_arrays=ca)
@printf("simulate: %.1f s  success=%s nstates=%d\n", time()-t0, sim.success, length(sim.u[1]))
As = sort([k for k in keys(sim.var_map) if startswith(k,"Chem.A[")])
println("Chem.A cells: ", length(As), "  (expect 343 = 7*7*7 if LIFTED)")
if !isempty(As)
    a0 = [sim.u[1][sim.var_map[k]] for k in As]; a1 = [sim.u[end][sim.var_map[k]] for k in As]
    b1 = [sim.u[end][sim.var_map[replace(k,"Chem.A["=>"Chem.B[")]] for k in As]
    @printf("  A: t0 in [%.4f,%.4f] -> t1 in [%.4f,%.4f]\n", minimum(a0),maximum(a0),minimum(a1),maximum(a1))
    @printf("  B: t1 in [%.6f,%.6f]   (A->B at k=0.01/s over 1 s => B ~ 0.0995)\n", minimum(b1),maximum(b1))
end
