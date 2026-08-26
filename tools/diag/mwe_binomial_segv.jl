#!/usr/bin/env julia
# ===========================================================================
# mwe_binomial_segv.jl -- TARGET B, claim 2: does `checkpointing=Binomial(n)`
# on a STATIC-bound `@trace for` still SEGFAULT in BinomialProgressConstProp?
# ===========================================================================
# The original observation (Reactant 0.2.274, EarthSciAST/bench/whilerev_ckpt.jl):
#
#   @trace for, ckpt=Binomial(3)   *** SEGFAULT *** in
#     mlir::enzyme::CheckedOpRewritePattern<BinomialProgressOp,
#     BinomialProgressConstProp>::matchAndRewrite -> stablehlo::ConstantOp::build
#
# This is a BYTE-FOR-BYTE re-spelling of that probe, because a negative result
# is only worth something if the spelling matches: NSTEP=8, the same
# step(x)=0.99x+0.01x^2 body, the same 6-element X0, the same in-place
# `Enzyme.autodiff(Reverse, f, Active, Duplicated(x,dx))` (NOT `Enzyme.gradient`)
# and the same `raise=true raise_first=true sync=true` compile options -- the
# raising pipeline is where the crashing pattern runs.
#
# A segfault kills the process, so this runs ONE configuration per process and
# the verdict is taken from the parent shell (mwe_binomial_sweep.sh; 139=SIGSEGV).
#
# Env:
#   MWE_VARIANT  none | true | periodic | binomial | binomial_mincut
#                                                        (default binomial)
#   MWE_BUDGET   checkpoint budget      (default 3)
#   MWE_NSTEP    static trip count      (default 8)
#   MWE_RAISE    1 = raise=true raise_first=true  (default 1, as the original)
#
#   julia --project=$RESEACT_RXENV tools/diag/mwe_binomial_segv.jl
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf
const RX = Reactant
const Enzyme = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end
say(s) = (println(s); flush(stdout))

const VARIANT = get(ENV, "MWE_VARIANT", "binomial")
const BUDGET  = parse(Int, get(ENV, "MWE_BUDGET", "3"))
const NSTEP   = parse(Int, get(ENV, "MWE_NSTEP", "8"))
const RAISE   = get(ENV, "MWE_RAISE", "1") == "1"

step(x) = 0.99 .* x .+ 0.01 .* x .* x

loss_host(x)     = (for _ in 1:NSTEP; x = step(x); end; sum(x))
loss_none(x)     = (RX.@trace track_numbers=false for _ in 1:NSTEP; x = step(x); end; sum(x))
loss_true(x)     = (RX.@trace track_numbers=false checkpointing=true for _ in 1:NSTEP; x = step(x); end; sum(x))
loss_periodic(x) = (RX.@trace track_numbers=false checkpointing=RX.Periodic(BUDGET) for _ in 1:NSTEP; x = step(x); end; sum(x))
loss_binomial(x) = (RX.@trace track_numbers=false checkpointing=RX.Binomial(BUDGET) for _ in 1:NSTEP; x = step(x); end; sum(x))
loss_binomial_mincut(x) = (RX.@trace track_numbers=false checkpointing=RX.Binomial(BUDGET) mincut=true for _ in 1:NSTEP; x = step(x); end; sum(x))

const F = VARIANT == "none" ? loss_none :
          VARIANT == "true" ? loss_true :
          VARIANT == "periodic" ? loss_periodic :
          VARIANT == "binomial" ? loss_binomial :
          VARIANT == "binomial_mincut" ? loss_binomial_mincut :
          VARIANT == "host" ? loss_host :
          error("MWE_VARIANT?")

const X0 = collect(range(0.5, 1.5; length = 6))
# ANALYTIC reference: x_{j+1} = 0.99 x_j + 0.01 x_j^2, so
# dx_{j+1}/dx_j = 0.99 + 0.02 x_j, and d sum(x_N)/d x_0 is the product along
# each lane. Exact, closed form, no autodiff and no finite difference.
function analytic(x0)
    g = ones(length(x0)); x = copy(x0)
    for _ in 1:NSTEP
        g .*= (0.99 .+ 0.02 .* x)
        x = 0.99 .* x .+ 0.01 .* x .* x
    end
    g
end
const GREF = analytic(X0)

grad!(f, x, dx) = (Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                                   Enzyme.Duplicated(x, dx)); nothing)

say(@sprintf("mwe_binomial_segv  variant=%s budget=%d nstep=%d raise=%s",
             VARIANT, BUDGET, NSTEP, RAISE))
say(@sprintf("  Reactant %s  Enzyme %s  julia %s", pkgversion(Reactant),
             pkgversion(Enzyme), VERSION))
say(@sprintf("  analytic grad[1..3] = %.15g %.15g %.15g", GREF[1], GREF[2], GREF[3]))

# The primal first: if this fails the AD result below means nothing.
let x = RX.to_rarray(copy(X0))
    try
        c = RAISE ? (RX.@compile raise=true sync=true F(x)) : (RX.@compile sync=true F(x))
        say(@sprintf("  primal OK  %.12f", Float64(c(x))))
    catch e
        say("  primal FAILED: " * first(split(sprint(showerror, e), '\n')))
    end
end

say("  --> entering the reverse compile (this is where 0.2.274 died)")
function run_reverse()
    x  = RX.to_rarray(copy(X0))
    dx = RX.to_rarray(zeros(length(X0)))
    try
        cg = RAISE ? (RX.@compile raise=true raise_first=true sync=true grad!(F, x, dx)) :
                     (RX.@compile sync=true grad!(F, x, dx))
        cg(F, x, dx)
        g = Array(dx)
        r = maximum(abs.(g .- GREF)) / maximum(abs.(GREF))
        say(@sprintf("  reverse returned g[1..3] = %.15g %.15g %.15g", g[1], g[2], g[3]))
        return (r <= 1e-12 ? "EXACT" : "WRONG"), r
    catch e
        str = sprint(showerror, e)
        m = match(r"error: ([^\n]+)", str)
        say("  " * (m === nothing ? first(split(str, '\n')) : "MLIR: " * strip(m.captures[1])))
        return "COMPILE_FAIL", NaN
    end
end
const status, rel = run_reverse()

say(@sprintf("RESULT variant=%s budget=%d nstep=%d raise=%s status=%s rel=%.6g",
             VARIANT, BUDGET, NSTEP, RAISE, status, rel))
say("MWE_BINOMIAL_SEGV_DONE")
