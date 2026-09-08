#!/usr/bin/env julia
# ===========================================================================
# frozen_loop_smoke.jl -- RxTracedIntegrator's FROZEN GRID, on a toy RHS.
# ===========================================================================
# `frozen_grid_probe.jl` / `frozen_grid_probe2.jl` established that a masked
# static-CAP `@trace for` differentiates exactly and that `Ops.dynamic_slice` is
# the only working per-iteration dt read. This file checks the thing those
# probes do NOT: that `RxTracedIntegrator.frozen_run` / `frozen_vjp` -- the
# actual entry points `tools/frozen_device_loop.jl` calls, with the real
# `ssprk43_step_unrolled` and `ros23_step` in the loop body and a real
# `(p = ..., bufs = ...)` theta -- compile and differentiate exactly.
#
# It runs in ~2 minutes on a toy 3-species / 8-cell RHS, which is what makes it
# the right first gate: the ReSEACT driver takes ~7 minutes to BUILD before it
# reaches the same code path.
#
# Four things it can catch, each of which costs a full driver cycle to find the
# slow way:
#   1. a traced value hiding inside a closure the loop body calls -- the
#      `stablehlo.while` verifier rejects the module ("expect operands to be
#      compatible with body block arguments");
#   2. `track_numbers` promoting the step's host Ints (N, NS, NC) to traced
#      scalars and breaking `reshape`;
#   3. the masked (dead) iterations poisoning the reverse pass;
#   4. the parameter cotangent not being summed over the live steps.
#
# THE ONE THING THIS FILE HAS TO GET RIGHT ITSELF -- AND DID NOT, FIRST TIME.
# `RX.@compile` DONATES the input buffers it does not return. Reusing one
# `ConcreteRArray` across two calls therefore hands the second call a CORRUPTED
# array, and it does so silently: the first measurement in a fresh process is
# right and every one after it drifts. That is exactly the house failure mode --
# it compiles, it runs, and the number is wrong -- and an earlier version of this
# probe reported a 1.7e-1 "gradient error" in `frozen_vjp` that was entirely its
# own reuse of `UD`. Every call below therefore uploads FRESH inputs, which is
# also what the adjoint driver does (`step_call` / `vjp_call` / `devloop_*`
# build a new `ConcreteRArray` per call), and which is why the driver never sees
# this.
#
# The reference is a HOST loop over the same fixed (t, dt) grid, differentiated
# by ForwardDiff -- an entirely separate implementation of the same map.
# ===========================================================================
using Printf
include(joinpath(@__DIR__, "_env.jl"))
using Reactant, ForwardDiff, LinearAlgebra
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end
say(s) = (println(s); flush(stdout))

const RXDIR = normpath(joinpath(@__DIR__, "..", "reactant_handoff"))
include(joinpath(RXDIR, "rx_native_patch.jl"))
include(joinpath(RXDIR, "rx_traced_integrator.jl"))
const RTI = RxTracedIntegrator

say("="^78)
say("frozen_loop_smoke   Reactant v$(pkgversion(Reactant))   julia $(VERSION)")
say("="^78)

const NS = 3
const NC = 8
const N  = NS * NC
const ATOL, RTOL = 1e-6, 1e-4

# A toy chemistry-shaped RHS: cell-local, species-coupled, parameterised by two
# scalars, and reading a "forcing buffer" so `theta` has the same
# (p = NamedTuple of scalars, bufs = NamedTuple of arrays) shape the driver
# passes. `t` enters through a diurnal factor, so the per-iteration `t` read is
# exercised too.
function toyrhs(u, theta, t)
    k1 = theta.p.k1; k2 = theta.p.k2
    w = theta.bufs.temp
    a = u[1:NC]; b = u[(NC + 1):(2NC)]; c = u[(2NC + 1):(3NC)]
    s = 1.0 + 0.5 * sin(t / 600.0)
    r1 = (k1 * s) .* a .* w
    r2 = k2 .* b .* b
    return vcat(-r1, r1 .- r2, r2 .- (0.001 .* c))
end

const HB = (temp = [0.8 + 0.05i for i in 1:NC],)
const HP = (k1 = 0.02, k2 = 0.01)
const U0 = [0.5 + 0.01i for i in 1:N]
const T0 = 5400.0
const LAM = Float64[sin(0.7i) for i in 1:N]     # a dense, arbitrary adjoint seed

# A genuinely VARYING (t, dt) grid with PADDING past `nlive` -- exactly what the
# driver hands the loop, and the case a probe that only tests cap == nlive would
# miss. The padding is the LAST LIVE (t, dt): never zero, because a Rosenbrock
# attempt forms W = I/(gamma*dt).
const NLIVE = 7
const CAP = 16
const SEQ = [(T0 + 3.0 * (i - 1)^1.3, 0.4 + 0.05i) for i in 1:NLIVE]
const TS = Float64[SEQ[min(i, NLIVE)][1] for i in 1:CAP]
const DTS = Float64[SEQ[min(i, NLIVE)][2] for i in 1:CAP]

const MASKS = [Float64[(s - 1) * NC < i <= s * NC ? 1.0 : 0.0 for i in 1:N] for s in 1:NS]

# ---- the HOST reference: the same map, in plain Julia --------------------
function host_map(pv, which::Symbol, clamp::Bool, u0)
    th = (p = (k1 = pv[1], k2 = pv[2]), bufs = (temp = HB.temp,))
    u = u0
    for i in 1:NLIVE
        t, dt = SEQ[i]
        un = which === :T ?
            first(RTI.ssprk43_step_unrolled((uu, ss) -> toyrhs(uu, th, ss), u, t, dt, ATOL, RTOL)) :
            first(RTI.ros23_step((uu, ss) -> toyrhs(uu, th, ss), u, t, dt, NS, NC, MASKS,
                                 ATOL, RTOL; unrolled = true, jac = :fd))
        u = clamp ? ifelse.(un .> 0.0, un, zero(un)) : un
    end
    return u
end

# FRESH device inputs for every single call -- see the header.
devu() = RX.ConcreteRArray(copy(U0))
devth() = (p = (k1 = RX.ConcreteRNumber(HP.k1), k2 = RX.ConcreteRNumber(HP.k2)),
           bufs = (temp = RX.ConcreteRArray(copy(HB.temp)),))
devlam() = RX.ConcreteRArray(copy(LAM))
devts() = RX.ConcreteRArray(copy(TS))
devdts() = RX.ConcreteRArray(copy(DTS))
devnl() = RX.ConcreteRNumber(NLIVE)

fail = 0
for which in (:T, :C), clamp in (false, true)
    tag = "$(which === :T ? "SSPRK43" : "ROS23") clamp=$(clamp)"
    pv = [HP.k1, HP.k2]
    uref = host_map(pv, which, clamp, U0)
    gref = ForwardDiff.gradient(v -> dot(LAM, host_map(v, which, clamp, eltype(v).(U0))), pv)
    lref = ForwardDiff.gradient(uu -> dot(LAM, host_map(pv, which, clamp, uu)), U0)

    bodyf = which === :T ?
        RTI.frozen_ssprk43_body(toyrhs, ATOL, RTOL; clamp_nonneg = clamp) :
        RTI.frozen_ros23_body(toyrhs, NS, NC, MASKS, ATOL, RTOL;
                              jac = :fd, clamp_nonneg = clamp)

    try
        fo_ = (u, th, ts, dts, nl) -> RTI.frozen_run(bodyf, u, th, ts, dts, nl, CAP)
        co = RX.@compile sync=true fo_(devu(), devth(), devts(), devdts(), devnl())
        ud = Array(co(devu(), devth(), devts(), devdts(), devnl()))
        du = maximum(abs.(ud .- uref) ./ max.(abs.(uref), 1e-30))
        ok = du <= 1e-12
        ok || (global fail += 1)
        @printf("  [%s] %-18s primal   max rel %.3e\n", ok ? "PASS" : "FAIL", tag, du)
    catch e
        global fail += 1
        @printf("  [ERR ] %-18s primal   %s\n", tag, first(split(sprint(showerror, e), '\n')))
        continue
    end

    try
        fv_ = (u, th, l, ts, dts, nl) ->
            RTI.frozen_vjp(bodyf, u, th, l, ts, dts, nl, CAP; active_bufs = false)
        cv = RX.@compile sync=true fv_(devu(), devth(), devlam(), devts(), devdts(), devnl())
        r = cv(devu(), devth(), devlam(), devts(), devdts(), devnl())
        gp = [Float64(r[2].p.k1), Float64(r[2].p.k2)]
        lin = Array(r[1])
        rp = maximum(abs.(gp .- gref) ./ max.(abs.(gref), 1e-30))
        rl = maximum(abs.(lin .- lref) ./ max.(abs.(lref), 1e-30))
        ok = rp <= 1e-10 && rl <= 1e-10
        ok || (global fail += 1)
        @printf("  [%s] %-18s dJ/dp    rel %.3e   lambda_in rel %.3e   [% .10e % .10e]\n",
                ok ? "PASS" : "FAIL", tag, rp, rl, gp[1], gp[2])
    catch e
        global fail += 1
        @printf("  [ERR ] %-18s vjp      %s\n", tag, first(split(sprint(showerror, e), '\n')))
    end
end

# ---- the structure claim: a while region, and O(1) in cap -----------------
let bodyf = RTI.frozen_ssprk43_body(toyrhs, ATOL, RTOL; clamp_nonneg = true)
    say("\n  differentiated module (SSPRK43 half):")
    for cap in (16, 128)
        fv_ = (u, th, l, ts, dts, nl) ->
            RTI.frozen_vjp(bodyf, u, th, l, ts, dts, nl, cap; active_bufs = false)
        s = repr(RX.@code_hlo optimize=false fv_(devu(), devth(), devlam(),
                                                 RX.ConcreteRArray(zeros(cap)),
                                                 RX.ConcreteRArray(ones(cap)),
                                                 RX.ConcreteRNumber(cap)))
        nw = length(collect(eachmatch(r"stablehlo\.while\b", s)))
        @printf("    cap=%-5d %2d stablehlo.while  %7d lines   %s\n", cap, nw,
                count(==('\n'), s) + 1, nw > 0 ? "NOT unrolled" : "UNROLLED -- O(cap)")
    end
end

say(fail == 0 ? "\nFROZEN_LOOP_SMOKE_OK" : "\nFROZEN_LOOP_SMOKE_FAILED ($fail)")
exit(fail == 0 ? 0 : 1)
