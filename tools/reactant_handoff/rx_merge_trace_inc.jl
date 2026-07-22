# Trace section for rx_merge_kernels.jl — include'd conditionally so `using`
# takes effect before the code that needs it (a `using` inside an if-block is
# invisible to the rest of the same top-level expression).
using Reactant
using Profile
try; Reactant.set_default_backend("cpu"); catch; end
include(joinpath(HERE, "rx_native_patch.jl"))
devnum(pp::NamedTuple) = NamedTuple{keys(pp)}(map(Reactant.ConcreteRNumber, values(pp)))
devnum(::Nothing) = nothing
ur = Reactant.ConcreteRArray(u0); pr = devnum(p0); tr = Reactant.ConcreteRNumber(T0)
dev = map(Reactant.ConcreteRArray, EA.forcing_buffers(fo1))
say("\n=== @compile MERGED transport ===")
tt0 = time()
xla_m = Reactant.@compile sync=true merged_rhs(ur, pr, tr, dev)
say("RESULT: merged transport @compile SUCCEEDED in $(round(time()-tt0, digits=1)) s")
report_patch_stats()
got = Array(xla_m(ur, pr, tr, dev))
say("RESULT: compiled-vs-host maxabs=$(maximum(abs, got .- du_ref)) " *
    "approx=$(isapprox(got, du_ref; rtol=1e-10, atol=1e-12))")
tt1 = @elapsed for _ in 1:10; xla_m(ur, pr, tr, dev); end
say("RESULT: compiled RHS eval=$(round(tt1/10*1000, digits=2))ms/call")
