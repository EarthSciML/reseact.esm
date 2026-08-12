# j02 -- soak, and the HEAP-POISON control.
#
# If the compiled program reads a temp buffer it never wrote, its output is a
# function of whatever the allocator handed it -- usually the previous call's
# nearly-identical data (so: a tiny difference), occasionally something wild
# (so: a NaN). That hypothesis makes a prediction no other one does: deliberately
# filling recently-freed memory with a poison value should change the output
# distribution. Phase B does exactly that, with 0x7ff8... (NaN) and with a huge
# finite value, and compares the fault rate against the un-poisoned Phase A run
# interleaved with it.
const J2N = parse(Int, get(ENV, "J2N", "6000"))
const J2MB = parse(Int, get(ENV, "J2MB", "192"))

poisonbuf = Float64[]
function poison!(v::Float64, mb::Int)
    global poisonbuf
    n = (mb * 1024 * 1024) ÷ 8
    poisonbuf = Vector{Float64}(undef, n)
    fill!(poisonbuf, v)
    poisonbuf = Float64[]          # drop the reference
    GC.gc(false)                   # return it to the allocator
    return nothing
end

function soak(nm, c, TH, dt, u::Vector{Float64}, t::Float64, K::Int; poison::Float64=NaN,
              dopoison::Bool=false)
    ref = nothing; ndiff = 0; nbad = 0; maxrel = 0.0; t0 = time()
    for k in 1:K
        dopoison && poison!(poison, J2MB)
        out = Array(callstep(c, u, TH, t, dt)[1])
        if count(!isfinite, out) > 0
            nbad += 1
            say("  !! $nm k=$k NON-FINITE ($(count(!isfinite,out))) poison=$(dopoison ? poison : nothing) load=$(loadavg())")
            report_bad(nm, out, u, t, dt, c, TH)
            for j in 1:3
                o = Array(callstep(c, u, TH, t, dt)[1])
                say("     re-issue $j nonfinite=$(count(!isfinite,o)) bitdiff=$(length(bitdiff(out,o)))")
            end
        end
        if ref === nothing
            ref = out
        else
            idx = bitdiff(ref, out)
            if !isempty(idx)
                ndiff += 1
                maxrel = max(maxrel, maximum(i -> abs(out[i]-ref[i])/max(abs(ref[i]),1e-300), idx))
            end
        end
        k % 1000 == 0 && (say("    $nm $k/$K bad=$nbad differing=$ndiff $(round(time()-t0,digits=1))s"); flush(stdout))
    end
    say("  RESULT soak $nm poison=$(dopoison ? string(poison) : "none") K=$K nonfinite=$nbad " *
        "bit_differing=$ndiff maxrel=$maxrel s_per_call=$(round((time()-t0)/K, digits=5))")
    return nbad
end

say("---- j02: soak + heap poison, N=$J2N per cell, poison block $(J2MB) MB ----")
soak("ros/clean", CROS, THC, DT0C, USTEP, T0 + DT0C, J2N)
soak("ros/poisonNaN", CROS, THC, DT0C, USTEP, T0 + DT0C, J2N; poison=NaN, dopoison=true)
soak("ros/poisonBig", CROS, THC, DT0C, USTEP, T0 + DT0C, J2N; poison=1.0e300, dopoison=true)
soak("ssp/clean", CSSP, THT, DT0T, USTEP, T0 + DT0C, J2N)
soak("ssp/poisonNaN", CSSP, THT, DT0T, USTEP, T0 + DT0C, J2N; poison=NaN, dopoison=true)
