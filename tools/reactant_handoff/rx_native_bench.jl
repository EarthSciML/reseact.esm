#!/usr/bin/env julia
# Microbenchmark: how does Reactant TRACE time scale with the number of
# broadcast sites, stock vs. the native-lowering patch (rx_native_patch.jl)?
# Isolates the throughput question from the 1.5 h full-model run.
#
# BENCH_PATCH=0 for stock lowering (one helper per site; keep K*2 < 10k cap),
# BENCH_PATCH=1 (default) for native. K list via BENCH_KS="250,500,1000,2000".
import Pkg
const HERE  = @__DIR__
const REPO  = normpath(joinpath(HERE, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io=devnull)
using Reactant
using Profile   # makes SIGUSR1 a live-profile peek instead of process death
try; Reactant.set_default_backend("cpu"); catch; end
const PATCHED = get(ENV, "BENCH_PATCH", "1") == "1"
PATCHED && include(joinpath(HERE, "rx_native_patch.jl"))
println("patched=", PATCHED); flush(stdout)

# K iterations * 2 broadcasts each, unrolled at trace time
function mkchain(K)
    return function (x, y)
        r = x
        for _ in 1:K
            r = r .* y .+ x
        end
        return r
    end
end

x = Reactant.ConcreteRArray(rand(637))
y = Reactant.ConcreteRArray(rand(637))
warm = Reactant.@compile sync=true mkchain(2)(x, y)   # warm up compile path

for K in parse.(Int, split(get(ENV, "BENCH_KS", "250,500,1000,2000"), ","))
    f = mkchain(K)
    t = @elapsed (xla = Reactant.@compile sync=true f(x, y))
    ok = isapprox(Array(xla(x, y)), f(Array(x), Array(y)); rtol=1e-10)
    println("K=$K sites=$(2K) trace+compile=$(round(t, digits=1))s per-site=$(round(1000t/2K, digits=2))ms correct=$ok")
    flush(stdout)
end
PATCHED && report_patch_stats()
println("BENCH DONE")
