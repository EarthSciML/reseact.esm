#!/usr/bin/env julia
# ===========================================================================
# mwe_case_reverse.jl -- TARGET A: does `stablehlo.case` have a reverse rule?
# ===========================================================================
# CLAIM UNDER TEST (reseact.esm HELPERS.md ~line 527):
#   "`stablehlo.case` has no reverse rule at all -- keep it out of
#    differentiated code."
#
# Two separate questions, and the first one turns out to matter most:
#
#   Q1  WHAT EMITS `stablehlo.case` AT ALL?  The natural assumption is that a
#       `Reactant.@trace if` on a traced condition does. It does NOT:
#       `ReactantCore.traced_if` -> `Ops.if_condition` -> `stablehlo.if`, in
#       every Reactant from 0.2.274 to 0.2.280 (`grep -c Ops.case src/ControlFlow.jl`
#       is 0 in all of them). The ONLY user-reachable emitter of `stablehlo.case`
#       is an explicit `Reactant.Ops.case(index, branch_fns, args...)`.
#       Section 1 MEASURES this rather than assuming it.
#
#   Q2  Given a module that really does contain `stablehlo.case`, does
#       `Enzyme.gradient(Reverse, ...)` cross it, and does it get the RIGHT
#       answer? Section 3 differentiates an explicit `Ops.case` against an
#       ANALYTIC gradient, so a SILENTLY WRONG answer is caught and not just a
#       hard failure. Every branch is exercised, because a rule that always
#       differentiates branch 0 would pass one index and fail the others.
#
# Section 2 is the `@trace if` / `ifelse` control pair: identical mathematics,
# one with a region and one with a `select`.
#
#   julia --project=$RESEACT_RXENV tools/diag/mwe_case_reverse.jl
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf
const RX = Reactant
const Enzyme = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end

say(s) = (println(s); flush(stdout))
ok(t, g, m) = say(@sprintf("  [%s] %-32s %s", g ? "PASS" : "FAIL", t, m))
function mlirerr(e)
    s = sprint(showerror, e)
    m = match(r"error: ([^\n]+)", s)
    return m === nothing ? first(split(s, '\n')) : "MLIR: " * strip(m.captures[1])
end
countop(txt, op) = length(collect(eachmatch(Regex("stablehlo\\.$op\\b"), txt)))
function opcensus(f, args...)
    h = repr(RX.@code_hlo optimize = false f(args...))
    (case = countop(h, "case"), iff = countop(h, "if"), select = countop(h, "select"))
end

const NX = 6
const X0 = [0.5 + 0.1i for i in 1:NX]

# ---------------------------------------------------------------------------
# The functions under test. `s` / `idx` select the arm; `x` is differentiated.
#   arm 0: sum(x.^2)   d/dx = 2x
#   arm 1: sum(x.^3)   d/dx = 3x^2
#   arm 2: sum(x.^4)   d/dx = 4x^3
# ---------------------------------------------------------------------------
arm(x, b) = b == 0 ? sum(x .^ 2) : b == 1 ? sum(x .^ 3) : sum(x .^ 4)
danalytic(x, b) = b == 0 ? 2 .* x : b == 1 ? 3 .* x .^ 2 : 4 .* x .^ 3

function f_traceif(x, s)                     # 2-arm @trace if
    y = zero(eltype(x))
    RX.@trace if s > 0.5
        y = sum(x .^ 2)
    else
        y = sum(x .^ 3)
    end
    y
end

function f_traceif3(x, s)                    # 3-arm @trace if/elseif/else
    y = zero(eltype(x))
    RX.@trace if s > 1.5
        y = sum(x .^ 4)
    elseif s > 0.5
        y = sum(x .^ 3)
    else
        y = sum(x .^ 2)
    end
    y
end

f_select(x, s) = ifelse(s > 0.5, sum(x .^ 2), sum(x .^ 3))

# THE construct that actually emits `stablehlo.case`.
f_case(x, idx) = RX.Ops.case(idx,
                             [a -> sum(a .^ 2),      # branch 0
                              a -> sum(a .^ 3),      # branch 1
                              a -> sum(a .^ 4)],     # branch 2 (also the default)
                             x; track_numbers = Number)

grad(f) = (x, s) -> Enzyme.gradient(Enzyme.Reverse, f, x, Enzyme.Const(s))[1]

say("mwe_case_reverse  Reactant $(pkgversion(Reactant))  Enzyme $(pkgversion(Enzyme))  julia $(VERSION)")
say("")

# ---- 1/1b. what emits stablehlo.case? -------------------------------------
say("=== 1. construct census (@code_hlo optimize=false on the PRIMAL) ===")
const NCASE = Ref(-1)
for (nm, f, s) in (("@trace if   (2 arm)", f_traceif,  RX.ConcreteRNumber(1.0)),
                   ("@trace if   (3 arm)", f_traceif3, RX.ConcreteRNumber(2.0)),
                   ("ifelse / select    ", f_select,   RX.ConcreteRNumber(1.0)),
                   ("Ops.case    (3 arm)", f_case,     RX.ConcreteRNumber(1)))
    XR = RX.ConcreteRArray(X0)
    try
        c = opcensus(f, XR, s)
        nm == "Ops.case    (3 arm)" && (NCASE[] = c.case)
        say(@sprintf("  %-20s case=%d if=%d select=%d", nm, c.case, c.iff, c.select))
    catch e
        say(@sprintf("  %-20s HLO FAILED: %s", nm, mlirerr(e)))
    end
end
if NCASE[] < 1
    say("")
    say("  !! Even `Ops.case` did not emit stablehlo.case. Nothing below tests the")
    say("     claim; the probe is broken, not the compiler.")
end
say("")

# ---- 2/3. reverse mode ----------------------------------------------------
function revtest(tag, f, x0, s, want; rtol = 1e-12)
    XR = RX.ConcreteRArray(x0)
    g = grad(f)
    nc = -1
    try; nc = opcensus(g, XR, s).case; catch; end
    try
        t0 = time()
        xg = RX.@compile sync = true g(XR, s)
        tc = time() - t0
        got = Array(xg(XR, s))
        rel = maximum(abs.(got .- want) ./ max.(abs.(want), 1e-300))
        ok(tag, rel <= rtol,
           @sprintf("rel err %.3g  (grad HLO case=%d, compile %.1f s)", rel, nc, tc))
        rel > rtol && say("        got  $got\n        want $want")
    catch e
        ok(tag, false, mlirerr(e))
    end
end

say("=== 2. control pair: `@trace if` (stablehlo.if) vs `ifelse` (select) ===")
for (nm, f) in (("@trace if", f_traceif), ("ifelse   ", f_select))
    for s in (1.0, 0.0)
        revtest(@sprintf("%s s=%.0f", nm, s), f, X0, RX.ConcreteRNumber(s),
                danalytic(X0, s > 0.5 ? 0 : 1))
    end
end
say("")

say("=== 3. THE TEST: reverse mode over an explicit `Ops.case` ===")
for b in 0:2
    revtest(@sprintf("Ops.case branch %d", b), f_case, X0, RX.ConcreteRNumber(b),
            danalytic(X0, b))
end
# index out of range -> stablehlo.case takes the LAST branch; the gradient must too.
revtest("Ops.case branch 7 (=> 2)", f_case, X0, RX.ConcreteRNumber(7), danalytic(X0, 2))
say("")

# ---- 4. is it reverse SPECIFICALLY, or `case` and autodiff generally? ------
say("=== 4. the same `Ops.case`, FORWARD mode (JVP against the analytic) ===")
for b in 0:2
    XR = RX.ConcreteRArray(X0)
    V  = ones(NX)                       # seed direction; JVP = dot(danalytic, V)
    want = sum(danalytic(X0, b) .* V)
    fwd = (x, dx, s) -> Enzyme.autodiff(Enzyme.Forward, f_case,
                                        Enzyme.Duplicated(x, dx), Enzyme.Const(s))[1]
    try
        xg = RX.@compile sync = true fwd(XR, RX.ConcreteRArray(V), RX.ConcreteRNumber(b))
        got = Float64(xg(XR, RX.ConcreteRArray(V), RX.ConcreteRNumber(b)))
        rel = abs(got - want) / abs(want)
        ok(@sprintf("Ops.case fwd branch %d", b), rel <= 1e-12,
           @sprintf("got %.12g want %.12g rel %.3g", got, want, rel))
    catch e
        ok(@sprintf("Ops.case fwd branch %d", b), false, mlirerr(e))
    end
end
say("")
say("MWE_CASE_REVERSE_DONE")
