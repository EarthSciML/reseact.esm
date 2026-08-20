#!/usr/bin/env julia
# ===========================================================================
# frozen_grid_probe.jl -- can the differentiated time loop live ON THE DEVICE?
# ===========================================================================
# The host-lifted adjoint exists because reverse mode needs a statically known
# trip count and the adaptive controller's is data-dependent. Two candidate
# ways out, measured here against a ForwardDiff host reference:
#
#   K1  reverse over `@trace while` (data-dependent)          -- the baseline
#   K2  ... + checkpointing=Binomial(budget)                  -- what Reactant's
#       OWN error hint (Reactant.jl:381) tells you to do, and what Diffrax does
#       (RecursiveCheckpointAdjoint / equinox bounded_while_loop)
#   K3  THE FROZEN GRID: static-trip-count `@trace for` over a compile-time CAP
#       with the accepted dt sequence handed in as a RUNTIME TENSOR and dead
#       iterations masked by `ifelse` on a traced live-count. Trip count is the
#       literal CAP, so this needs no checkpointing at all.
#       This is what adjoint_gradient.jl's (t,dt) recording ALREADY produces --
#       it just replays it on the host, one XLA call per step. jaxdae does the
#       same thing ("freezing the accepted step grid").
#   K4  Binomial correctness on a STATIC-bound loop -- the Reactant.jl#1895 /
#       bench/maskedloop.jl Y4 shape, which returned a SILENTLY WRONG gradient
#       on 0.2.274 (exact at n=2,4; 1.0e-3 rel at n=5; 2.8e-3 at n=10).
#
# Run against both Reactant versions and compare.
# ===========================================================================
using Pkg
using Reactant, ForwardDiff, Printf
const RX = Reactant
const EZ = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end
say(s) = (println(s); flush(stdout))
function err1(e)
    s = sprint(showerror, e)
    m = match(r"error: ([^\n]+)", s)
    return m === nothing ? first(split(s, '\n')) : "MLIR: " * m.captures[1]
end

say("="^75)
say("frozen_grid_probe   Reactant v$(pkgversion(Reactant))   julia $(VERSION)")
say("="^75)

const NS   = 6
const U0   = [0.5 + 0.1i for i in 1:NS]
const K0   = 0.7
const TEND = 1.0
const H    = 0.02                     # so the exact trip count is 50
const CAP  = 64                       # compile-time cap >= trip count

# the vector field, identical in every arm
@inline vf(u, k) = k .* u .* (1 .+ 0.1 .* u)

# ---- host reference: fixed grid of NSTEP explicit-Euler steps ---------------
const NSTEP = Int(round(TEND / H))
function host_fixed(k)
    u = copy(U0)
    for _ in 1:NSTEP; u = u .+ H .* vf(u, k); end
    sum(u)
end
const GREF = ForwardDiff.derivative(host_fixed, K0)
say(@sprintf("host reference: NSTEP=%d  J=%.12g  dJ/dk=%.12g", NSTEP, host_fixed(K0), GREF))

function check(tag, thunk; gref = GREF, rtol = 1e-8)
    try
        # NB: `thunk` returns a ConcreteRNumber (a device value). Converting to
        # Float64 must happen HERE, on the host -- doing it INSIDE the compiled
        # function raises `MethodError: no method matching
        # Float64(::TracedRNumber{Float64})`, because during tracing there is no
        # value yet to convert.
        t0 = time(); got = Float64(thunk()); el = time() - t0
        rel = abs(got - gref) / abs(gref)
        pass = rel <= rtol
        say(@sprintf("  [%s] %-34s got %.12g  rel %.2e  (%.1f s)%s",
                     pass ? "PASS" : "FAIL", tag, got, rel, el,
                     pass ? "" : "   <-- WRONG ANSWER, not an error"))
        return pass
    catch e
        say(@sprintf("  [ERR ] %-34s %s", tag, err1(e)))
        return false
    end
end

# stays fully traced: no host conversion anywhere inside the compiled function
gradk(f) = (u, k, rest...) -> EZ.gradient(EZ.Reverse, f, EZ.Const(u), k,
                                          map(EZ.Const, rest)...)[2]

# ===========================================================================
say("\n---- K1/K2  reverse over a DATA-DEPENDENT `@trace while` ----")
# stop when t >= TEND: trip count is decided by the data, not the compiler
function while_body(u, k, ckpt)
    t = zero(k)
    if ckpt === nothing
        RX.@trace track_numbers=false while t < TEND - 1e-9
            u = u .+ H .* vf(u, k); t = t + H
        end
    else
        RX.@trace track_numbers=false checkpointing=ckpt while t < TEND - 1e-9
            u = u .+ H .* vf(u, k); t = t + H
        end
    end
    return sum(u)
end
let ur = RX.ConcreteRArray(U0), kr = RX.ConcreteRNumber(K0)
    f1 = (u, k) -> while_body(u, k, nothing)
    check("K1 while, no checkpointing", () -> begin
        g = gradk(f1); c = RX.@compile sync=true g(ur, kr); c(ur, kr)
    end)
    for b in (4, 8, 16)
        fb = (u, k) -> while_body(u, k, RX.Binomial(b))
        check("K2 while + Binomial($b)", () -> begin
            g = gradk(fb); c = RX.@compile sync=true g(ur, kr); c(ur, kr)
        end)
    end
end

# ===========================================================================
say("\n---- K3  THE FROZEN GRID: static CAP, dt as a runtime tensor, masked ----")
# `dts` is padded to CAP with a DUMMY POSITIVE value (never 0 -- a real
# Rosenbrock step forms W = I/(gamma*dt) and would divide by zero); the padded
# iterations are neutralised by `ifelse` on the carry, exactly as
# rx_traced_integrator.adaptive_solve already writes every update.
const DTS_PAD = Float64[i <= NSTEP ? H : 1.0 for i in 1:CAP]

# (a) index the runtime dt vector by the loop-carried counter
function frozen_a(u, k, dts, nlive)
    # `i` is the TRACED induction variable, so `dts[i]` lowers to a dynamic_slice.
    # A Float64-typed counter cannot index at all -- `ArgumentError: invalid
    # index ::TracedRNumber{Float64}` -- which is the trap here. This body has no
    # reshape dims to protect, so it does NOT need track_numbers=false (that is
    # only why fd_block_jac uses it).
    RX.@trace for i in 1:CAP
        h = dts[i]
        live = i <= nlive
        un = u .+ h .* vf(u, k)
        u = ifelse.(live, un, u)
    end
    return sum(u)
end
# (b) no indexing at all: dt rides as a loop-carried scalar, mask by counter.
#     (the real driver's inner steps share one dt per macro half often enough
#      that this is a useful fallback if dynamic_slice misbehaves)
function frozen_b(u, k, h, nlive)
    c = zero(k) + 1
    RX.@trace track_numbers=false for _ in 1:CAP
        live = c <= nlive
        un = u .+ h .* vf(u, k)
        u = ifelse.(live, un, u)
        c = c + 1
    end
    return sum(u)
end

let ur = RX.ConcreteRArray(U0), kr = RX.ConcreteRNumber(K0),
    dr = RX.ConcreteRArray(DTS_PAD), nr = RX.ConcreteRNumber(NSTEP),
    nrf = RX.ConcreteRNumber(Float64(NSTEP)), hr = RX.ConcreteRNumber(H)
    check("K3a for(CAP) + dts[c] runtime tensor", () -> begin
        g = gradk(frozen_a); c = RX.@compile sync=true g(ur, kr, dr, nr); c(ur, kr, dr, nr)
    end)
    check("K3b for(CAP) + scalar dt, masked", () -> begin
        g = gradk(frozen_b); c = RX.@compile sync=true g(ur, kr, hr, nrf); c(ur, kr, hr, nrf)
    end)
    # does the differentiated module still contain a while region (i.e. NOT unrolled)?
    try
        g = gradk(frozen_b)
        s = repr(RX.@code_hlo optimize=false g(ur, kr, hr, nrf))
        nw = length(collect(eachmatch(r"stablehlo\.while\b", s)))
        say(@sprintf("       differentiated module: %d stablehlo.while, %d lines  (%s)",
                     nw, count(==('\n'), s) + 1,
                     nw > 0 ? "NOT unrolled -- program size O(1) in CAP" : "UNROLLED -- size O(CAP)"))
    catch e
        say("       code_hlo failed: " * err1(e))
    end
end

# ===========================================================================
say("\n---- K4  Binomial on a STATIC-bound loop (Reactant.jl#1895 shape) ----")
function static_loop(u, k, ckpt)
    if ckpt === nothing
        RX.@trace track_numbers=false for _ in 1:NSTEP
            u = u .+ H .* vf(u, k)
        end
    else
        RX.@trace track_numbers=false checkpointing=ckpt for _ in 1:NSTEP
            u = u .+ H .* vf(u, k)
        end
    end
    return sum(u)
end
let ur = RX.ConcreteRArray(U0), kr = RX.ConcreteRNumber(K0)
    for (nm, ck) in (("none", nothing), ("true", true), ("Periodic(5)", RX.Periodic(5)),
                     ("Binomial(2)", RX.Binomial(2)), ("Binomial(5)", RX.Binomial(5)),
                     ("Binomial(10)", RX.Binomial(10)))
        f = (u, k) -> static_loop(u, k, ck)
        check("K4 static for, ckpt=$nm", () -> begin
            g = gradk(f); c = RX.@compile sync=true g(ur, kr); c(ur, kr)
        end)
    end
end

say("\nFROZEN_GRID_PROBE_DONE")
