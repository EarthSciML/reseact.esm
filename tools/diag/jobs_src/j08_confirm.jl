# j08 -- confirm the two knobs that zeroed the RHS fault rate ALSO fix the whole
# ROS23 step program (the object the driver actually calls), and check what they
# cost: does the "fixed" program return the same numbers as the faulty one on the
# calls where the faulty one is correct?
#
# j06b, 40000 calls each, chemistry RHS:
#   baseline 19,28   eigen_mt_off 22,24   xnnpack_off 19   isa_avx2 15   isa_sse4 19
#   codegen_split1 24                                        <- still faults
#   fusion_emit_off 0   ynn_off 0   vecwidth128 0   backend_opt0 0   <- clean
const J8N = parse(Int, get(ENV, "J8N", "20000"))
CO8(nt; kw...) = RX.CompileOptions(; sync = true, xla_debug_options = nt, kw...)

const V8 = [
    ("baseline",        CO8((;))),
    ("fusion_emit_off", CO8((; xla_cpu_use_fusion_emitters = false))),
    ("vecwidth128",     CO8((; xla_cpu_prefer_vector_width = 128))),
]

say("---- j08: does the fix hold on the full ros23 step? N=$J8N ----")
ROSV = Dict{String,Any}()
for (nm, co) in V8
    try
        ROSV[nm] = timed_compile("ros:$nm", () -> RX.@compile compile_options=co ros_step(U_R, THC, T_R, DTC_R))
    catch e
        say("  ros:$nm COMPILE FAILED: $(sprint(showerror, e))")
    end
end

# do the variants agree with each other numerically on a clean call?
let refs = Dict(nm => Array(c(RX.ConcreteRArray(USTEP), THC, RX.ConcreteRNumber(T0 + DT0C),
                              RX.ConcreteRNumber(DT0C))[1]) for (nm, c) in ROSV)
    b = refs["baseline"]
    for (nm, v) in sort(collect(refs))
        d = maximum(i -> abs(v[i] - b[i]) / max(abs(b[i]), 1e-300), eachindex(b))
        say("  agreement $nm vs baseline: bitdiff=$(count(i -> !(v[i] === b[i]), eachindex(b)))/$(length(b)) maxrel=$d")
    end
end

function cell8(nm, c, K)
    ref = Array(c(RX.ConcreteRArray(USTEP), THC, RX.ConcreteRNumber(T0 + DT0C), RX.ConcreteRNumber(DT0C))[1])
    nbad = 0; nnf = 0; t0 = time(); l0 = loadavg()
    for k in 1:K
        out = Array(c(RX.ConcreteRArray(USTEP), THC, RX.ConcreteRNumber(T0 + DT0C), RX.ConcreteRNumber(DT0C))[1])
        any(i -> !(out[i] === ref[i]), eachindex(out)) && (nbad += 1)
        any(!isfinite, out) && (nnf += 1)
    end
    say("  RATE8 ros/$nm K=$K deviating=$nbad nonfinite=$nnf rate=$(round(nbad/K, sigdigits=3)) " *
        "load=$l0..$(loadavg()) s_per_call=$(round((time()-t0)/K, digits=5))")
    flush(stdout)
end

for pass in 1:2, (nm, _) in V8
    haskey(ROSV, nm) && cell8("$nm/p$pass", ROSV[nm], J8N)
end
