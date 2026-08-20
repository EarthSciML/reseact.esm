#!/usr/bin/env julia
# ===========================================================================
# frozen_grid_probe2.jl -- the two open questions from frozen_grid_probe.jl
# ===========================================================================
# probe 1 established: a static-trip-count `@trace for` over a compile-time CAP
# with dead iterations masked by `ifelse` differentiates EXACTLY (rel 0.0) on
# BOTH 0.2.274 and 0.2.280, keeps a `stablehlo.while` in the differentiated
# module (so program size is O(1) in CAP), and needs NO checkpointing -- which
# matters because every checkpointing mode returns a silently wrong gradient.
#
# Two things it did not settle, and both decide whether this works on ReSEACT:
#
#  L. PER-ITERATION dt. The real frozen grid has a DIFFERENT dt each step.
#     `dts[i]` with a traced `i` raises "Scalar indexing is disallowed".
#     DIFFERENTIABILITY_PLAN.md section 1 notes a size-1 SLICE read works where
#     a scalar read does not -- test that and two alternatives.
#
#  M. TAPE COST AT REAL STATE SIZE. Enzyme tapes the trajectory as a dense
#     [CAP, state] tensor. At CONUS state = 85,176, so CAP=256 is 175 MB for the
#     state alone -- but the tape also holds the body's live intermediates, and
#     the toy body here is far smaller than a ROS23 step. This measures the
#     FLOOR, not the ReSEACT number: if the floor is already bad, stop.
# ===========================================================================
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
anon_gb() = try
    cg = split(readline("/proc/self/cgroup"), ':')[3]
    for l in eachline("/sys/fs/cgroup$cg/memory.stat")
        startswith(l, "anon ") && return parse(Int, split(l)[2]) / 2^30
    end; NaN
catch; NaN end

say("="^78)
say("frozen_grid_probe2   Reactant v$(pkgversion(Reactant))   julia $(VERSION)")
say("="^78)

@inline vf(u, k) = k .* u .* (1 .+ 0.1 .* u)

# ---------------------------------------------------------------- L ---------
# A genuinely VARYING dt sequence, so a formulation that silently reads the
# wrong element cannot accidentally agree with the reference.
const NS_L  = 6
const U0_L  = [0.5 + 0.1i for i in 1:NS_L]
const K0    = 0.7
const CAP_L = 16
const NLIVE = 10
const DTS_L = Float64[i <= NLIVE ? 0.005 * i : 1.0 for i in 1:CAP_L]  # varying, padded

function host_L(k)
    u = copy(U0_L)
    for i in 1:NLIVE; u = u .+ DTS_L[i] .* vf(u, k); end
    sum(u)
end
const GREF_L = ForwardDiff.derivative(host_L, K0)
say(@sprintf("\nL reference (varying dt, %d live of CAP %d): dJ/dk = %.12g", NLIVE, CAP_L, GREF_L))

# (a) size-1 SLICE instead of a scalar read
function L_slice(u, k, dts, nlive)
    RX.@trace for i in 1:CAP_L
        h = sum(dts[i:i])                    # size-1 slice -> scalar by reduction
        un = u .+ h .* vf(u, k)
        u = ifelse.(i <= nlive, un, u)
    end
    sum(u)
end
# (b) explicit dynamic_slice
function L_dynslice(u, k, dts, nlive)
    RX.@trace for i in 1:CAP_L
        h = sum(RX.Ops.dynamic_slice(dts, [i], [1]))
        un = u .+ h .* vf(u, k)
        u = ifelse.(i <= nlive, un, u)
    end
    sum(u)
end
# (c) no indexing at all: select by a one-hot mask. O(CAP) per iteration, but
#     CAP is small and this uses only ops already known to differentiate.
function L_onehot(u, k, dts, nlive)
    idx = RX.Ops.constant(collect(1:CAP_L))
    RX.@trace for i in 1:CAP_L
        h = sum(ifelse.(idx .== i, dts, zero(dts)))
        un = u .+ h .* vf(u, k)
        u = ifelse.(i <= nlive, un, u)
    end
    sum(u)
end

gradk(f) = (u, k, rest...) -> EZ.gradient(EZ.Reverse, f, EZ.Const(u), k,
                                          map(EZ.Const, rest)...)[2]

function check(tag, thunk, gref; rtol = 1e-10)
    try
        t0 = time(); got = Float64(thunk()); el = time() - t0
        rel = abs(got - gref) / abs(gref)
        say(@sprintf("  [%s] %-32s got %.12g  rel %.2e  (%.1f s)%s",
                     rel <= rtol ? "PASS" : "FAIL", tag, got, rel, el,
                     rel <= rtol ? "" : "   <-- WRONG, not an error"))
        return rel <= rtol
    catch e
        say(@sprintf("  [ERR ] %-32s %s", tag, err1(e)))
        return false
    end
end

let ur = RX.ConcreteRArray(U0_L), kr = RX.ConcreteRNumber(K0),
    dr = RX.ConcreteRArray(DTS_L), nr = RX.ConcreteRNumber(NLIVE)
    for (tag, f) in (("L-a dts[i:i] size-1 slice", L_slice),
                     ("L-b Ops.dynamic_slice",     L_dynslice),
                     ("L-c one-hot mask select",   L_onehot))
        check(tag, () -> begin
            g = gradk(f); c = RX.@compile sync=true g(ur, kr, dr, nr); c(ur, kr, dr, nr)
        end, GREF_L)
    end
end

# ---------------------------------------------------------------- M ---------
say("\nM tape cost at real state size -- dense [CAP, state] is the design's one cost")
say("  (this is the FLOOR: the toy body has ~4 live intermediates, a ROS23 step has many more)")
const NLIVE_M = 24
for nstate in (3_744, 85_176)          # 6x6x8 demo, then CONUS
    uv = [0.5 + 1e-5 * i for i in 1:nstate]
    for cap in (32, 128, 256)
        dts = Float64[i <= NLIVE_M ? 0.01 : 1.0 for i in 1:cap]
        href = kk -> (u = copy(uv); for i in 1:NLIVE_M; u = u .+ dts[i] .* vf(u, kk); end; sum(u))
        gref = ForwardDiff.derivative(href, K0)
        f = (u, k, h, n) -> begin
            c = zero(k) + 1
            RX.@trace track_numbers=false for _ in 1:cap
                un = u .+ h .* vf(u, k)
                u = ifelse.(c <= n, un, u)
                c = c + 1
            end
            sum(u)
        end
        ur = RX.ConcreteRArray(uv); kr = RX.ConcreteRNumber(K0)
        hr = RX.ConcreteRNumber(0.01); nr = RX.ConcreteRNumber(Float64(NLIVE_M))
        m0 = anon_gb()
        say(@sprintf("  state=%-7d cap=%-4d  dense[cap,state] = %6.1f MB", nstate, cap,
                     nstate * cap * 8 / 2^20))
        check(@sprintf("    ns=%d cap=%d", nstate, cap), () -> begin
            g = gradk(f); c = RX.@compile sync=true g(ur, kr, hr, nr); r = c(ur, kr, hr, nr)
            say(@sprintf("       anon RSS %.2f -> %.2f GB (+%.2f)", m0, anon_gb(), anon_gb() - m0))
            r
        end, gref)
    end
end
say("\nFROZEN_GRID_PROBE2_DONE")
