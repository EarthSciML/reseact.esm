# j07 -- is the faulting kernel READING MEMORY IT NEVER WROTE?
#
# What j01-j06b have established: the chemistry RHS is a pure StableHLO dataflow
# graph (no while, no rng, no custom_call), it is called with bit-identical
# inputs, and its output is bit-identical on ~999 calls in 1000 -- and on the
# thousandth, ONE lane (cell (1,1,1)) of the six dry-deposition species is NaN.
# There is never anything in between: no ULP drift, ever, in >200k calls.
#
# "All-correct or one-lane-garbage, nothing in between" is the signature of a
# read of memory the kernel did not write, in a process where the allocator
# almost always hands back the SAME buffer holding the SAME previous (correct)
# value. That hypothesis makes a testable prediction the alternatives do not:
# disturbing what is in recently-freed memory between calls should change the
# rate. Three ways of disturbing it here:
#   A  poison   allocate a block, fill it with a bit pattern, free it.
#   B  interleave  run a DIFFERENT compiled program between calls, so XLA's own
#      allocator recycles its buffers.
#   C  churn    lots of small Julia allocations, then GC.
# Each is measured against a baseline cell run immediately before it.
const J7N  = parse(Int, get(ENV, "J7N", "20000"))
const J7MB = parse(Int, get(ENV, "J7MB", "8"))

callrhs7(c, TH, u, t) = Array(c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t)))

function poison!(v::Float64, mb::Int)
    b = Vector{Float64}(undef, (mb * 1024 * 1024) ÷ 8)
    fill!(b, v)
    b = nothing
    GC.gc(false)
    return nothing
end

function cell(nm, K; between = nothing)
    ref = callrhs7(CRHS, THC, USTEP, T0 + DT0C)
    nbad = 0; t0 = time(); l0 = loadavg()
    for k in 1:K
        between === nothing || between()
        out = callrhs7(CRHS, THC, USTEP, T0 + DT0C)
        any(i -> !(out[i] === ref[i]), eachindex(out)) && (nbad += 1)
    end
    say("  RATE7 $nm K=$K faults=$nbad rate=$(round(nbad/K, sigdigits=3)) " *
        "load=$l0..$(loadavg()) s_per_call=$(round((time()-t0)/K, digits=6))")
    flush(stdout)
    return nbad
end

say("---- j07: does disturbing recently-freed memory change the rate? N=$J7N ----")
cell("baseline_A", J7N)
cell("poison_NaN_$(J7MB)MB", J7N; between = () -> poison!(NaN, J7MB))
cell("baseline_B", J7N)
cell("poison_big_$(J7MB)MB", J7N; between = () -> poison!(1.0e300, J7MB))
cell("baseline_C", J7N)
cell("interleave_tranRHS", J7N;
     between = () -> callrhs7(CRHST, THT, USTEP, T0 + DT0C))
cell("baseline_D", J7N)
cell("julia_churn", J7N; between = function ()
         s = 0.0
         for _ in 1:200; v = rand(1024); s += v[1]; end
         s
     end)
cell("baseline_E", J7N)
