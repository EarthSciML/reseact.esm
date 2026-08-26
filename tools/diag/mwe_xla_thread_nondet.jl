#!/usr/bin/env julia
# ===========================================================================
# mwe_xla_thread_nondet.jl -- TARGET C: a model-free hammer for the claimed
# XLA:CPU multi-thread wrong-answer fault.
# ===========================================================================
# CLAIM UNDER TEST (tools/diag/README-nondet.md):
#   "The compiled ReSEACT chemistry RHS is a pure StableHLO dataflow graph, and
#    XLA:CPU executes it WRONGLY on about 1 call in 2,000 when it has more than
#    one intra-op thread. The corruption is always the same shape: NaN in the
#    six dry-deposition species at one grid cell (lane 1). With ONE XLA thread
#    it never happens. ~5e-4 per call at 20 threads; ~1e-2 per call pinned to
#    4 CPUs."
#
# This is the SYNTHETIC half of the investigation: no model, no EarthSciAST, no
# autodiff. It builds a program with the same STRUCTURAL character as the
# chemistry RHS -- a few thousand lanes, a long chain of fused elementwise ops
# including transcendentals (exp/log/pow/divide), compare+select clamps, a
# couple of cross-lane reductions to force a real fusion boundary, and a
# "deposition"-shaped tail whose outputs are reciprocals of a sum of
# exponentials (the exact algebraic shape of the six species that went NaN) --
# and then calls it in a tight loop, checking every result BIT-FOR-BIT against
# the first call and against a golden vector.
#
# XLA:CPU sizes its intra-op pool from `tsl::port::MaxParallelism()`, which
# reads the process CPU affinity mask, so the thread count is set by running
# under `taskset -c ...`, NOT by an environment variable. Run it both ways:
#
#   taskset -c 0-19 julia --project=$RESEACT_RXENV tools/diag/mwe_xla_thread_nondet.jl
#   taskset -c 0    julia --project=$RESEACT_RXENV tools/diag/mwe_xla_thread_nondet.jl
#
# On the first run pass MWE_GOLDEN=write while pinned to ONE cpu to record the
# single-thread answer; later runs compare against it.
#
# Env:
#   MWE_LANES   lanes (default 3744 == ReSEACT N at 6x6x8)
#   MWE_DEPTH   elementwise chain depth (default 60)
#   MWE_CALLS   calls in the hammer (default 20000)
#   MWE_GOLDEN  write | check | off        (default check; file below)
#   MWE_VECW    1 = also compile with xla_cpu_prefer_vector_width=128
#
#   Fault rates to beat: 1e-2/call would show ~200 hits in 20,000.
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf, Serialization
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end
say(s) = (println(s); flush(stdout))

const LANES  = parse(Int, get(ENV, "MWE_LANES", "3744"))
const DEPTH  = parse(Int, get(ENV, "MWE_DEPTH", "60"))
const CALLS  = parse(Int, get(ENV, "MWE_CALLS", "20000"))
const GOLDEN = get(ENV, "MWE_GOLDEN", "check")
const VECW   = get(ENV, "MWE_VECW", "0") == "1"
const GFILE  = joinpath(@__DIR__, "logs", "mwe_xla_thread_nondet.golden")

cpuset() = try
    strip(first(split(split(read("/proc/self/status", String), "Cpus_allowed_list:")[2], '\n')))
catch; "?" end
ncpu() = try; length(Base.Sys.cpu_info()); catch; -1 end

# --------------------------------------------------------------------------
# The program. Deliberately NOT vectorisation-friendly in a uniform way: the
# lanes carry very different magnitudes (as chemistry species do, 1e-12 to
# 1e12), so denormals, cancellation and reciprocals of near-zero sums all
# occur, and the compare/select clamps break the chain into fusion regions.
# --------------------------------------------------------------------------
function chem_like(u, k)
    x = u
    s = zero(eltype(u))
    for j in 1:DEPTH
        a = exp.(clamp.(-x .* k, -60.0, 60.0))          # Arrhenius-shaped
        b = log.(abs.(x) .+ 1.0e-30)                    # log of a clamped magnitude
        c = (x .* a) ./ (a .+ abs.(b) .+ 1.0e-25)       # divide by a small sum
        d = (abs.(x) .+ 1.0e-20) .^ 1.37                 # non-integer pow
        e = ifelse.(x .> 0.0, c .+ d, c .- d)            # compare + select
        x = e .- 1.0e-3 .* (e .- x)
        if j % 17 == 0
            s = sum(x)                                   # a real reduction: fusion boundary
            x = x .- (s / LANES) .* 1.0e-6
        end
        x = clamp.(x, 1.0e-30, 1.0e14)                   # the production clamp,
                                                         # and what keeps the map bounded
    end
    # The "dry deposition" tail: v_d = 1/(r_a + r_b + r_c), each r a reciprocal
    # of an exponential -- the algebraic shape of the six species that went NaN.
    ra = 1.0 ./ (exp.(clamp.(x .* 1.0e-3, -60.0, 60.0)) .+ 1.0e-30)
    rb = 1.0 ./ (exp.(clamp.(sqrt.(x) .* 1.0e-2, -60.0, 60.0)) .+ 1.0e-30)
    rc = 1.0 ./ (x .^ 0.61 .+ 1.0e-30)
    vd = 1.0 ./ (ra .+ rb .+ rc)
    return x .* (1.0 .- vd .* 1.0e-4)
end

# lane magnitudes spanning 24 decades, like a chemistry state vector
const U0 = [10.0^(-12 + 24 * ((i - 1) / (LANES - 1))) * (1 + 1.0e-3 * sin(i)) for i in 1:LANES]
const K0 = 0.37

const COPTS = VECW ?
    RX.CompileOptions(; sync = true, xla_debug_options = (; xla_cpu_prefer_vector_width = 128)) :
    RX.CompileOptions(; sync = true)

say(@sprintf("mwe_xla_thread_nondet  Reactant %s  julia %s", pkgversion(Reactant), VERSION))
say(@sprintf("  lanes=%d depth=%d calls=%d vecw128=%s", LANES, DEPTH, CALLS, VECW))
say(@sprintf("  host cpus=%d  Cpus_allowed_list=%s  julia threads=%d",
             ncpu(), cpuset(), Threads.nthreads()))

UR = RX.ConcreteRArray(U0); KR = RX.ConcreteRNumber(K0)
t0 = time()
C = RX.@compile compile_options=COPTS chem_like(UR, KR)
say(@sprintf("  compiled in %.1f s", time() - t0))

# op census, so the reader can see this really is a long fused dataflow graph
try
    h = repr(RX.@code_hlo optimize=false chem_like(UR, KR))
    cnt(op) = length(collect(eachmatch(Regex("stablehlo\\.$op\\b"), h)))
    say(@sprintf("  HLO ops: exp=%d log=%d power=%d divide=%d select=%d compare=%d reduce=%d",
                 cnt("exponential"), cnt("log"), cnt("power"), cnt("divide"),
                 cnt("select"), cnt("compare"), cnt("reduce")))
catch e
    say("  HLO census failed: " * first(split(sprint(showerror, e), '\n')))
end

ref = Array(C(UR, KR))
say(@sprintf("  reference: |x|max=%.6g  nonfinite=%d  x[1]=%.17g",
             maximum(abs, ref), count(!isfinite, ref), ref[1]))

if count(!isfinite, ref) > 0
    say("  !! the REFERENCE itself is non-finite -- the synthetic program overflows on its")
    say("     own and this probe cannot distinguish that from the fault. ABORTING.")
    exit(2)
end

golden = nothing
if GOLDEN == "write"
    mkpath(dirname(GFILE)); open(GFILE, "w") do io; serialize(io, ref); end
    say("  golden WRITTEN to $GFILE  (do this pinned to ONE cpu)")
elseif GOLDEN == "check" && isfile(GFILE)
    golden = open(deserialize, GFILE)
    nd = count(i -> !(golden[i] === ref[i]), eachindex(ref))
    say(@sprintf("  golden loaded: %d/%d lanes differ from this run's first call", nd, LANES))
else
    say("  no golden file; bit-exactness is measured against THIS process's first call")
end

# --------------------------------------------------------------------------
# The hammer. Fresh input buffer per call, exactly as the driver issues it.
# --------------------------------------------------------------------------
say("\n---- hammer: $CALLS calls ----")
nbad = 0; ndiff = 0; ngold = 0; firstdiff = Int[]
th = time()
for k in 1:CALLS
    out = Array(C(RX.ConcreteRArray(U0), RX.ConcreteRNumber(K0)))
    nf = count(!isfinite, out)
    if nf > 0
        global nbad += 1
        say(@sprintf("  !! call %d: %d/%d NON-FINITE, lanes %s", k, nf, LANES,
                     join(findall(!isfinite, out)[1:min(end, 12)], ",")))
    end
    idx = findall(i -> !(ref[i] === out[i]), eachindex(ref))
    if !isempty(idx)
        global ndiff += 1
        isempty(firstdiff) && (global firstdiff = idx)
        ndiff <= 5 && say(@sprintf("  !! call %d: %d/%d lanes BIT-DIFFER from call 1, first %s",
                                   k, length(idx), LANES, join(idx[1:min(end, 12)], ",")))
    end
    if golden !== nothing && any(i -> !(golden[i] === out[i]), eachindex(out))
        global ngold += 1
    end
    k % 2000 == 0 && say(@sprintf("     %d/%d  nonfinite=%d bitdiffering=%d vs_golden=%d  %.1f s",
                                  k, CALLS, nbad, ndiff, ngold, time() - th))
end
el = time() - th
say(@sprintf("\nRESULT lanes=%d depth=%d calls=%d cpus=%s vecw128=%s", LANES, DEPTH, CALLS,
             cpuset(), VECW) *
    @sprintf("  nonfinite=%d rate=%.3g bitdiffering=%d rate=%.3g vs_golden=%d %.5f s/call",
             nbad, nbad / CALLS, ndiff, ndiff / CALLS, ngold, el / CALLS))
say(nbad == 0 && ndiff == 0 ?
    "VERDICT: PASS -- bit-identical on every call; the fault did NOT fire here." :
    "VERDICT: FAIL -- the program is not a function of its inputs.")
say("MWE_XLA_THREAD_NONDET_DONE")
