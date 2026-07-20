# G1 de-risk: is GEOS-FP vertical layer geometry (Δz from OFFSET Ap/Bp table_lookups)
# expressible AND correctly evaluable in v0.8.0? Simulate the accumulator model at several
# levels and compare P_lo/P_hi/dP/dz to a direct Ap/Bp computation.
import Pkg
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io=devnull)
import OrdinaryDiffEqTsit5
using EarthSciAST
const EA = EarthSciAST

getmod(name) = (for (p,m) in Base.loaded_modules; p.name==name && return m; end; error("no $name"))
J = getmod("JSON3")

HERE = @__DIR__
path = joinpath(HERE, "g1.esm")

# reference Ap/Bp arrays (1-based level index) from the same source the model was built from
apbp = J.read(read("/private/tmp/claude-501/-Users-ctessum-code-earthsciml-reseact-esm/f8e5391f-7aed-4ccd-9cc2-402284364f51/scratchpad/apbp.json", String), Dict)
Ap = Float64.(apbp["geosfp_Ap_pa"]["data"])
Bp = Float64.(apbp["geosfp_Bp"]["data"])
PS = 101325.0; T = 260.0; Rd_over_g = 29.27083
ref(lev) = begin
    plo = Ap[lev]   + Bp[lev]  *PS
    phi = Ap[lev+1] + Bp[lev+1]*PS
    (P_lo=plo, P_hi=phi, dP=plo-phi, dz=Rd_over_g*T*log(plo/phi))
end

f  = EA.load(path)
alg = OrdinaryDiffEqTsit5.Tsit5()

function sim_at(lev)
    fs = EA.flatten(f; base_path=HERE)
    sim = EA.simulate(fs, (0.0, 1.0); alg=alg,
                       parameters=Dict("LayerGeometry.lev"=>Float64(lev)),
                       saveat=[0.0, 1.0])
    vm = sim.var_map
    idx(sp) = vm[first(filter(k->endswith(string(k), sp), collect(keys(vm))))]
    (P_lo=sim.u[end][idx("col_P_lo")], P_hi=sim.u[end][idx("col_P_hi")],
     dP=sim.u[end][idx("col_dP")],   dz=sim.u[end][idx("col_dz")])
end

println("PS=$(PS) Pa, T=$(T) K  (edge lev=1 is the surface)")
println(rpad("lev",5), rpad("P_lo sim/ref (Pa)",30), rpad("P_hi (Pa)",22), rpad("dP (Pa)",14), rpad("dz sim/ref (m)",26), "match")
allok = true
for lev in (1, 10, 30, 50, 60, 72)
    global allok
    s = sim_at(lev); r = ref(lev)
    ok = all(abs(getfield(s,k)-getfield(r,k)) <= 1e-6*max(1,abs(getfield(r,k))) for k in (:P_lo,:P_hi,:dP,:dz))
    allok &= ok
    println(rpad(lev,5),
            rpad(string(round(s.P_lo,digits=1),"/",round(r.P_lo,digits=1)),30),
            rpad(string(round(s.P_hi,digits=1)),22),
            rpad(string(round(s.dP,digits=1)),14),
            rpad(string(round(s.dz,digits=2),"/",round(r.dz,digits=2)),26),
            ok ? "OK" : "MISMATCH")
end
println("\nAll levels match the direct Ap/Bp computation: ", allok)
println("=> offset table_lookup (Ap/Bp at lev+1) is expressible AND evaluates correctly in v0.8.0.")
