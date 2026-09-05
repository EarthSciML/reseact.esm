#!/usr/bin/env julia
# p8_mof_probe.jl -- does XLA:CPU multi-output-fuse? The efficient adjoint of a
# stencil is face-major: compute the shared face chain ONCE and emit its 5-6
# partials as separate slabs. Under per-output-element loop fusion without
# multi-output fusion, those slabs are 5 fusions that each recompute the chain
# (5x), or one fusion per shifted consumer (6x) -- which is where the transport
# VJP's 17-22x over its primal comes from (tools/diag/p8_ppm_reverse.jl).
# Here: g(x) = an expensive elementwise chain; outputs (g, 2g, 3g, 4g, 5g).
# One MOF => ~1x the chain; no MOF => ~5x.   P8_N default 76440.
#
# RESULTS (slurm 10376932, N=76440): one_out 0.839 ms; five_out 1.320 ms
# (5 fusions: the 40-step chain is materialised ONCE and read by 4 cheap
# consumers -- 1.57x, not 5x, so an expensive shared producer is NOT
# duplicated); five_cat (one concatenate-root fusion) 8.368 ms = 10x the chain:
# concatenate-root fusions are pathological on XLA:CPU. xla_cpu_use_fusion_
# emitters is not assignable on Reactant 0.2.280. Consequence for the PPM
# adjoint: the per-face sub-chains are individually CHEAP, so XLA duplicates
# them into every shifted consumer; a face-major adjoint would need its 5
# partial slabs to come out of one (expensive-looking) producer.
using Reactant, Printf, Statistics
const N = parse(Int, get(ENV, "P8_N", "76440"))
const OUT = joinpath(@__DIR__, "..", "..", "logs", "p8-mof-" * get(ENV, "SLURM_JOB_ID", "local")); mkpath(OUT)
say(s) = (println(s); flush(stdout))
function chain(x)
    y = x
    for k in 1:40
        y = ifelse.(y .* (x .- 0.3k) .> 0, max.(-2 .* abs.(y), min.(2 .* abs.(y), 0.5 .* (x .- y))), 0.1 .* y .+ 0.01k)
    end
    y
end
one_out(x) = chain(x)
five_out(x) = (g = chain(x); (g, 2 .* g, 3 .* g, 4 .* g, 5 .* g))
five_cat(x) = (g = chain(x); vcat(g, 2 .* g, 3 .* g, 4 .* g, 5 .* g))
copts(dir; extra = (;)) = Reactant.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128, xla_dump_to = dir, xla_dump_hlo_as_text = true, extra...))
xR = Reactant.ConcreteRArray(rand(N))
function timeit(name, c, x; reps = 40)
    for _ in 1:3; c(x); end
    ts = [(t0 = time(); c(x); time() - t0) for _ in 1:reps]
    say(@sprintf("P8MOF_TIME %-14s N=%d  median %.3f ms  min %.3f ms", name, N, 1e3 * median(ts), 1e3 * minimum(ts)))
end
for (name, f, extra) in (("one_out", one_out, (;)), ("five_out", five_out, (;)), ("five_cat", five_cat, (;)),
                         ("five_out_fe", five_out, (; xla_cpu_use_fusion_emitters = true)))
    dir = joinpath(OUT, name); mkpath(dir)
    c = try Reactant.compile(f, (xR,); compile_options = copts(dir; extra)) catch e
        say("  $name compile FAILED: " * first(split(sprint(showerror, e), '\n'))[1:min(end, 200)]); continue end
    timeit(name, c, xR)
    fs = filter(f -> occursin("after_optimizations.txt", f), readdir(dir))
    if !isempty(fs)
        txt = read(joinpath(dir, fs[argmax([filesize(joinpath(dir, f)) for f in fs])]), String)
        ent = split(split(txt, "ENTRY")[end], "\n\n")[1]
        say(@sprintf("   fusions in ENTRY: %d   (multi-output = tuple-shaped fusion: %s)",
                     count(r"= .* fusion\(", ent), occursin(r"= \(f64", ent) ? "yes" : "no"))
    end
end
say("P8MOF_DONE")
