import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "..", "run-model-jl"))); io=devnull)
using EarthSciAST
const EA = EarthSciAST
f = EA.load_path("probe6.esm")
M = f.models["Transport3D"]
for eq in M.equations
    sp = EA.differential_lhs_variable(eq.lhs)
    (sp === nothing || !startswith(String(sp),"Chem.")) && continue
    println("=== equation for ", sp, " ===")
    ops = Dict{String,Int}(); leaves = Dict{String,Int}()
    function w(e, d)
        if e isa EA.VarExpr; leaves[e.name]=get(leaves,e.name,0)+1; return; end
        e isa EA.OpExpr || return
        ops[e.op]=get(ops,e.op,0)+1
        for a in e.args; w(a,d+1); end
        e.expr_body === nothing || w(e.expr_body,d+1)
        e.values === nothing || (for v in e.values; w(v,d+1); end)
    end
    w(eq.rhs, 0)
    println("ops present: ", sort(collect(ops), by=x->-x[2])[1:min(9,end)])
    println("RAW 'D' ops still unexpanded: ", get(ops,"D",0))
    println("makearray count: ", get(ops,"makearray",0))
    println("leaves mentioning Chem: ", [k for k in keys(leaves) if occursin("Chem",k)])
    println("top leaves: ", sort(collect(leaves), by=x->-x[2])[1:min(6,end)])
    break
end
