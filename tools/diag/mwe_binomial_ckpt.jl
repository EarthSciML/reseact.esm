#!/usr/bin/env julia
# ===========================================================================
# mwe_binomial_ckpt.jl -- TARGET B: `checkpointing=Binomial(n)` on a
# `Reactant.@trace for` under `Enzyme.gradient(Reverse, ...)`.
# ===========================================================================
# TWO CLAIMS, from reseact.esm HELPERS.md ~line 526:
#   B1  Binomial(n) on a loop whose trip count is a RUNTIME ARGUMENT compiles
#       and returns a SILENTLY WRONG gradient (exact at n=2,4; 1.0e-3 rel at
#       n=5; 2.8e-3 at n=10).
#   B2  Binomial SEGFAULTS in `BinomialProgressConstProp` on a STATIC-bound
#       loop.
#
# B2 kills the process, so this script runs exactly ONE configuration per
# process and the verdict is recorded by the PARENT (see mwe_binomial_sweep.sh;
# exit code 139 is SIGSEGV). B1's error curve is assembled by that driver too.
#
# THE MODEL. u_{j+1} = u_j * (1 + k*H), N steps, objective sum(u_N). Closed
# form: f(k) = S0*(1+kH)^N, so df/dk = S0*N*H*(1+kH)^(N-1) EXACTLY -- an
# analytic reference, not a finite difference and not ForwardDiff, so a
# silently wrong gradient cannot hide behind reference error.
#
# THE LABEL TRAP (HELPERS.md's own lesson: a probe's label is not its
# semantics). "Static" here means STATIC IN MLIR, not "the caller always passes
# the same number". This script therefore prints the `stablehlo.while` census
# AND the loop's condition region from `@code_hlo optimize=false`, so the
# reader can see whether the bound is a `stablehlo.constant` or a block
# argument. `MWE_BOUND=traced` passes a ConcreteRNumber{Int}; `MWE_BOUND=static`
# writes the count as a literal in the Julia range.
#
# Env (all optional):
#   MWE_BOUND  static | traced        (default traced)
#   MWE_CKPT   none | periodic | binomial | auto   (default none)
#   MWE_BUDGET checkpoint budget n    (default 5)
#   MWE_TRIPS  loop trip count        (default 20)
#   MWE_HLO    1 = dump the while region                (default 1)
#
#   julia --project=$RESEACT_RXENV tools/diag/mwe_binomial_ckpt.jl
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf
const RX = Reactant
const Enzyme = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end

say(s) = (println(s); flush(stdout))
function mlirerr(e)
    s = sprint(showerror, e)
    m = match(r"error: ([^\n]+)", s)
    return m === nothing ? first(split(s, '\n')) : "MLIR: " * strip(m.captures[1])
end

const BOUND  = get(ENV, "MWE_BOUND", "traced")
const CKPT   = get(ENV, "MWE_CKPT", "none")
const BUDGET = parse(Int, get(ENV, "MWE_BUDGET", "5"))
const TRIPS  = parse(Int, get(ENV, "MWE_TRIPS", "20"))
const DOHLO  = get(ENV, "MWE_HLO", "1") == "1"

const NU = 6
const H  = 0.01
const U0 = [0.5 + 0.1i for i in 1:NU]
const K0 = 0.7
const S0 = sum(U0)
# EXACT analytic derivative of sum(u0*(1+kH)^N) wrt k.
const GREF = S0 * TRIPS * H * (1 + K0 * H)^(TRIPS - 1)

# --------------------------------------------------------------------------
# The eight loop bodies. They are written out rather than generated because
# `@trace`'s checkpointing keyword must be a literal at macroexpansion time.
# Every one of them computes exactly the same thing.
# --------------------------------------------------------------------------
step(u, k) = u .+ k .* u .* H

# --- static bound: the trip count is a Julia literal, so `1:TRIPS` is folded
#     and MLIR sees a constant bound.
f_static_none(u, k)     = (RX.@trace track_numbers=false for _ in 1:TRIPS; u = step(u, k); end; sum(u))
f_static_periodic(u, k) = (RX.@trace track_numbers=false checkpointing=RX.Periodic(BUDGET) for _ in 1:TRIPS; u = step(u, k); end; sum(u))
f_static_binomial(u, k) = (RX.@trace track_numbers=false checkpointing=RX.Binomial(BUDGET) for _ in 1:TRIPS; u = step(u, k); end; sum(u))
f_static_auto(u, k)     = (RX.@trace track_numbers=false checkpointing=true for _ in 1:TRIPS; u = step(u, k); end; sum(u))

# --- traced bound: `n` arrives as a ConcreteRNumber{Int} argument. Fixed in
#     the Julia source, statically UNKNOWN in MLIR. This is the shape the
#     HELPERS.md retraction is about.
f_traced_none(u, k, n)     = (RX.@trace track_numbers=false for _ in 1:n; u = step(u, k); end; sum(u))
f_traced_periodic(u, k, n) = (RX.@trace track_numbers=false checkpointing=RX.Periodic(BUDGET) for _ in 1:n; u = step(u, k); end; sum(u))
f_traced_binomial(u, k, n) = (RX.@trace track_numbers=false checkpointing=RX.Binomial(BUDGET) for _ in 1:n; u = step(u, k); end; sum(u))
f_traced_auto(u, k, n)     = (RX.@trace track_numbers=false checkpointing=true for _ in 1:n; u = step(u, k); end; sum(u))

const TABLE = Dict(
    ("static", "none")     => f_static_none,   ("static", "periodic") => f_static_periodic,
    ("static", "binomial") => f_static_binomial, ("static", "auto")   => f_static_auto,
    ("traced", "none")     => f_traced_none,   ("traced", "periodic") => f_traced_periodic,
    ("traced", "binomial") => f_traced_binomial, ("traced", "auto")   => f_traced_auto,
)
haskey(TABLE, (BOUND, CKPT)) || error("MWE_BOUND must be static|traced, MWE_CKPT none|periodic|binomial|auto")
const F = TABLE[(BOUND, CKPT)]

const UR = RX.ConcreteRArray(U0)
const KR = RX.ConcreteRNumber(K0)
const EXTRA = BOUND == "traced" ? (RX.ConcreteRNumber(TRIPS),) : ()

say(@sprintf("mwe_binomial_ckpt  bound=%s ckpt=%s budget=%d trips=%d",
             BOUND, CKPT, BUDGET, TRIPS))
say(@sprintf("  Reactant %s  Enzyme %s  julia %s", pkgversion(Reactant),
             pkgversion(Enzyme), VERSION))
say(@sprintf("  analytic d/dk sum(u_N) = %.17g", GREF))

# ---- what does the compiler actually see? --------------------------------
# This is the label-vs-semantics check. `stablehlo.while` must be present (else
# the loop was unrolled and the probe tests nothing), and the condition region
# tells us whether the bound is a constant.
if DOHLO
    try
        h = repr(RX.@code_hlo optimize=false F(UR, KR, EXTRA...))
        nwhile = length(collect(eachmatch(r"stablehlo\.while\b", h)))
        say(@sprintf("  PRIMAL HLO: stablehlo.while=%d", nwhile))
        i = findfirst("stablehlo.while", h)
        if i !== nothing
            ln = split(h[first(i):min(end, first(i) + 900)], '\n')
            pre = split(h[1:first(i)], '\n')
            say("  --- the 8 lines BEFORE the while (they define its operands) ---")
            for l in pre[max(1, end - 8):(end - 1)]; say("    " * l); end
            say("  --- while op head (cond region tells you if the bound is static) ---")
            for l in ln[1:min(end, 12)]; say("    " * l); end
            say("  --- end ---")
        end
    catch e
        say("  PRIMAL HLO FAILED: " * mlirerr(e))
    end
end

# ---- the gradient --------------------------------------------------------
g = (u, k, e...) -> Enzyme.gradient(Enzyme.Reverse, F, Enzyme.Const(u), k,
                                    map(Enzyme.Const, e)...)[2]

function run_gradient()
    try
        t0 = time()
        xg = RX.@compile sync=true g(UR, KR, EXTRA...)
        tc = time() - t0
        got = Float64(xg(UR, KR, EXTRA...))
        rel = abs(got - GREF) / abs(GREF)
        say(@sprintf("  compiled in %.1f s", tc))
        return (rel <= 1e-12 ? "EXACT" : "WRONG"), got, rel
    catch e
        say("  " * mlirerr(e))
        return "COMPILE_FAIL", NaN, NaN
    end
end
const status, got, rel = run_gradient()

say(@sprintf("RESULT bound=%s ckpt=%s budget=%d trips=%d status=%s got=%.17g want=%.17g rel=%.6g",
             BOUND, CKPT, BUDGET, TRIPS, status, got, GREF, rel))
say("MWE_BINOMIAL_CKPT_DONE")
