# j06b -- THE DISCRIMINATOR. Recompile the faulting object (the chemistry RHS)
# under different XLA:CPU backend options and measure each one's fault rate.
# Reactant threads a `xla_debug_options` NamedTuple into the executable's
# DebugOptions proto, so every variant is a ~6 s compile inside the process that
# is already built -- no rebuild, and the machine load is shared across variants
# because they are interleaved.
#
# What each knob tests:
#   eigen_mt_off     XLA:CPU's multi-threaded Eigen mode. The threading hypothesis.
#   xnnpack_off      XNNPACK kernels (on by default here).
#   fusion_emit_off  the new CPU fusion emitters (on by default here).
#   ynn_off          the experimental ynn fusion types (DOT/CONV/REDUCE, on here).
#   vecwidth128 / avx2 / sse4   vector width and ISA -- the fault is confined to
#                    LANE 1 of a 288-lane axis, which is what a vectorised kernel
#                    getting its first vector wrong would look like.
#   opt0             XLA backend optimisation off.
#   no_reactant_passes  Reactant's own MLIR pipeline off.
const J6N = parse(Int, get(ENV, "J6N", "40000"))

callrhs6(c, TH, u, t) = Array(c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t)))

function rate(nm, c, TH, u, t, K)
    ref = callrhs6(c, TH, u, t)
    nbad = 0; lanes = Set{Int}(); t0 = time(); l0 = loadavg()
    for k in 1:K
        out = callrhs6(c, TH, u, t)
        b = findall(i -> !(out[i] === ref[i]), eachindex(out))
        isempty(b) && continue
        nbad += 1
        for i in b; push!(lanes, ((i - 1) % NC) + 1); end
    end
    el = time() - t0
    say("  RATE $nm K=$K faults=$nbad rate=$(round(nbad/K, sigdigits=3)) " *
        "lanes=$(sort(collect(lanes))) load=$l0..$(loadavg()) s_per_call=$(round(el/K, digits=6))")
    flush(stdout)
    return nbad
end

CO(nt; kw...) = RX.CompileOptions(; sync = true, xla_debug_options = nt, kw...)

const VARIANTS = [
    ("baseline",        () -> CO((;))),
    ("eigen_mt_off",    () -> CO((; xla_cpu_multi_thread_eigen = false))),
    ("xnnpack_off",     () -> CO((; xla_cpu_use_xnnpack = false))),
    ("fusion_emit_off", () -> CO((; xla_cpu_use_fusion_emitters = false))),
    ("ynn_off",         () -> CO((; xla_cpu_experimental_ynn_fusion_type = RX.Proto.xla.var"DebugOptions.LibraryFusionType".T[]))),
    ("vecwidth128",     () -> CO((; xla_cpu_prefer_vector_width = 128))),
    ("isa_avx2",        () -> CO((; xla_cpu_max_isa = "AVX2"))),
    ("isa_sse4",        () -> CO((; xla_cpu_max_isa = "SSE4_2"))),
    ("backend_opt0",    () -> CO((; xla_backend_optimization_level = 0))),
    ("no_rx_passes",    () -> CO((;); optimization_passes = :none)),
    ("codegen_split1",  () -> CO((; xla_cpu_parallel_codegen_split_count = 1))),
]

say("---- j06b: fault rate vs XLA:CPU backend options, N=$J6N each ----")
COMPILED = Dict{String,Any}()
for (nm, mkco) in VARIANTS
    try
        co = mkco()
        COMPILED[nm] = timed_compile("v:$nm", () -> RX.@compile compile_options=co rhs_only(U_R, THC, T_R))
    catch e
        say("  $nm COMPILE FAILED: $(sprint(showerror, e))")
    end
end

# two interleaved passes so a drifting machine load cannot manufacture a difference
for pass in 1:2
    say("  == pass $pass ==")
    for (nm, _) in VARIANTS
        haskey(COMPILED, nm) || continue
        rate("$nm/p$pass", COMPILED[nm], THC, USTEP, T0 + DT0C, J6N)
    end
end
