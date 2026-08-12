# j09 -- poison XLA's OWN allocator, not Julia's.
#
# j07 poisoned the Julia heap between calls and moved the rate not at all. But
# XLA's temporaries do not come from Julia's heap; they come from the PJRT CPU
# allocator. This poisons THAT: allocate a pile of ConcreteRArrays full of a
# poison bit pattern, drop them, and let PJRT recycle those buffers into the
# next execution's temp arena. If the faulting kernel is reading memory it never
# wrote, the value it reads is the poison, and the rate should move.
#
# Also runs the complementary control: poison with the value the kernel WOULD
# have read if it were reading a stale copy of its own previous output.
const J9N  = parse(Int, get(ENV, "J9N", "20000"))
const J9K  = parse(Int, get(ENV, "J9K", "24"))     # arrays per poisoning round
const J9SZ = parse(Int, get(ENV, "J9SZ", "65536")) # doubles per array

callrhs9(c, TH, u, t) = Array(c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t)))

function xlapoison(v::Float64)
    keep = Vector{Any}(undef, J9K)
    for i in 1:J9K
        keep[i] = RX.ConcreteRArray(fill(v, J9SZ))
    end
    keep = nothing
    GC.gc(false)
    return nothing
end

function cell9(nm, K; between = nothing)
    ref = callrhs9(CRHS, THC, USTEP, T0 + DT0C)
    nbad = 0; t0 = time(); l0 = loadavg()
    for k in 1:K
        between === nothing || between()
        out = callrhs9(CRHS, THC, USTEP, T0 + DT0C)
        any(i -> !(out[i] === ref[i]), eachindex(out)) && (nbad += 1)
    end
    say("  RATE9 $nm K=$K faults=$nbad rate=$(round(nbad/K, sigdigits=3)) " *
        "load=$l0..$(loadavg()) s_per_call=$(round((time()-t0)/K, digits=6))")
    flush(stdout)
end

say("---- j09: poison the PJRT allocator, N=$J9N, $(J9K)x$(J9SZ) doubles per round ----")
cell9("base_1", J9N)
cell9("xlapoison_NaN", J9N; between = () -> xlapoison(NaN))
cell9("base_2", J9N)
cell9("xlapoison_big", J9N; between = () -> xlapoison(1.0e300))
cell9("base_3", J9N)
cell9("xlapoison_zero", J9N; between = () -> xlapoison(0.0))
cell9("base_4", J9N)
