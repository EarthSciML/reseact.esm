import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "..", "run-model-jl"))); io=devnull)
using EarthSciAST; import OrdinaryDiffEqTsit5; using JSON3, Printf
import SciMLBase   # phase 4: `solve` / `successful_retcode` are SciMLBase's own
const EA = EarthSciAST
r = EA.validate("probe6.esm"); println("validate: is_valid=$(r.is_valid) struct=$(length(r.structural_errors))")
for e in r.structural_errors[1:min(5,end)]; println("  ", e.error_type, " @ ", e.path, " :: ", e.message); end
f = EA.load_path("probe6.esm"); fs = EA.flatten(f)
for n in ["Chem.A","Chem.B","Transport3D.m"]
    haskey(fs.state_variables,n) && println("  $n :: shape=", fs.state_variables[n].shape)
end
co = JSON3.read(read("hybrid_coefs.json", String))
ca = Dict{String,Any}("Transport3D.dA"=>Float64.(co.dA), "Transport3D.dB"=>Float64.(co.dB))
t0=time()
# phase 4: build once, then solve; the state map lives on the problem.
prob = EA.esm_problem(fs, (0.0,1.0); const_arrays=ca)
sim = SciMLBase.solve(prob, OrdinaryDiffEqTsit5.Tsit5(); saveat=[0.0,1.0])
@printf("simulate: %.1f s  success=%s nstates=%d\n", time()-t0, SciMLBase.successful_retcode(sim), length(sim.u[1]))
As = sort([k for k in keys(prob.var_map) if startswith(k,"Chem.A[")])
println("Chem.A cells: ", length(As), "  (expect 343 = 7*7*7 if LIFTED)")
if !isempty(As)
    a0 = [sim.u[1][prob.var_map[k]] for k in As]; a1 = [sim.u[end][prob.var_map[k]] for k in As]
    b1 = [sim.u[end][prob.var_map[replace(k,"Chem.A["=>"Chem.B[")]] for k in As]
    @printf("  A: t0 in [%.4f,%.4f] -> t1 in [%.4f,%.4f]\n", minimum(a0),maximum(a0),minimum(a1),maximum(a1))
    @printf("  B: t1 in [%.6f,%.6f]   (A->B at k=0.01/s over 1 s => B ~ 0.0995)\n", minimum(b1),maximum(b1))
end
