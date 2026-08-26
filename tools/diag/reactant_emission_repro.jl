#!/usr/bin/env julia
# ===========================================================================
# reactant_emission_repro.jl -- two Reactant EMISSION defects, minimal form.
# ===========================================================================
# Both were found while cutting EarthSciAST's traced op count and are
# independent of that project. Neither involves autodiff.
#
#   A. `Ops.constant(::Number)` is NOT memoized, while `Ops.constant(::Array)`
#      IS. Every scalar literal mints a fresh `stablehlo.constant`. That is not
#      merely op bloat: two uses of the same scalar get DIFFERENT SSA values, so
#      any structural CSE above them fails to see `k*x` and `k*x` as the same
#      expression.
#
#   B. `broadcast_to_size` emits a `stablehlo.broadcast_in_dim` even when the
#      operand shape ALREADY matches the target, and `materialize` is
#      `copyto!(similar(bc), bc)` whose `similar` is a dead zero fill. So one
#      `a .+ b` costs several ops where one is wanted.
#
# Counts raw (`optimize=false`) ops, so it measures what Reactant HANDS to XLA,
# not what XLA keeps. XLA's own CSE cleans most of it up later -- the cost is
# borne in trace time, module size, and lost CSE opportunities upstream.
#
#   julia --project=$RESEACT_RXENV tools/diag/reactant_emission_repro.jl
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

say(s) = (println(s); flush(stdout))
function census(txt)
    h = Dict{String,Int}()
    for m in eachmatch(r"=\s+\"?(stablehlo\.[\w.]+)\"?", txt)
        h[m.captures[1]] = get(h, m.captures[1], 0) + 1
    end
    h
end
hlo(f, args...) = sprint(show, @code_hlo optimize = false f(args...))

say("Reactant $(pkgversion(Reactant))  julia $(VERSION)")

# ---- A. scalar constant memoization ---------------------------------------
# The SAME scalar literal, used k times. If scalar constants were memoized the
# way array constants are, this would emit ONE stablehlo.constant regardless.
for k in (1, 4, 16, 64)
    f = let k = k
        x -> begin
            acc = x
            for _ in 1:k
                acc = acc .+ 3.5           # the same Float64 literal, every time
            end
            acc
        end
    end
    X = RX.ConcreteRArray(collect(1.0:8.0))
    h = census(hlo(f, X))
    say(@sprintf("  A. scalar 3.5 used %3d times -> %3d stablehlo.constant  (%3d add)",
                 k, get(h, "stablehlo.constant", 0), get(h, "stablehlo.add", 0)))
end

# Control: the same test with an ARRAY constant, which Reactant DOES memoize.
for k in (1, 4, 16, 64)
    v = collect(1.0:8.0)
    f = let k = k, v = v
        x -> begin
            acc = x
            for _ in 1:k
                acc = acc .+ v             # the same Vector, every time
            end
            acc
        end
    end
    X = RX.ConcreteRArray(collect(1.0:8.0))
    h = census(hlo(f, X))
    say(@sprintf("  A-control. array const used %3d times -> %3d stablehlo.constant",
                 k, get(h, "stablehlo.constant", 0)))
end

# ---- B. broadcast scaffolding ---------------------------------------------
# One elementwise add of two SAME-SHAPE arrays. Ideal: 1 stablehlo.add.
let X = RX.ConcreteRArray(collect(1.0:8.0)), Y = RX.ConcreteRArray(collect(1.0:8.0))
    h = census(hlo((a, b) -> a .+ b, X, Y))
    say("\n  B. one same-shape `a .+ b` emits:")
    for (op, n) in sort(collect(h), by = x -> -x[2])
        say(@sprintf("       %-34s %d", op, n))
    end
    say(@sprintf("     total %d ops for what should be 1", sum(values(h))))
end
