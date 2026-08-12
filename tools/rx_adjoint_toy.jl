#!/usr/bin/env julia
# ===========================================================================
# rx_adjoint_toy.jl -- the MODEL-FREE half of the Phase 3 adjoint check.
# ===========================================================================
# `tools/rx_adjoint_check.jl` validates the step VJPs on ReSEACT, and costs a
# ~580 s build per run. This one runs the same checks on a 3-species / 5-cell
# toy in about a minute, which makes it the thing you actually run while
# changing `rx_traced_integrator.jl`. The stage algebra is model-independent,
# so a break here is a break everywhere.
#
# It answers two questions:
#
#  DEFAULT MODE -- is the step VJP right?
#    * dot-product identity <lam, J v> == <J' lam, v>       (exact arithmetic)
#    * central finite differences of the same compiled step
#    * a `stablehlo.while` census of the compiled VJP modules
#    for ROS23 with both Jacobians and for SSPRK43. Reference numbers, measured:
#      ROS23 jac=:ad   dot 1.8e-16, FD 5.9e-12
#      ROS23 jac=:fd   dot 6.1e-14  -- 350x worse, and that is not noise: an FD
#                      Jacobian puts a sqrt(eps)-scale difference quotient
#                      inside the step, so the step map is ill-conditioned in u
#                      by construction. On ReSEACT the same effect costs ~1e-6.
#      SSPRK43         dot 1.8e-16, FD 1.8e-13
#      all four modules: stablehlo.while = 0
#
#  VARIANT MODE (`julia tools/rx_adjoint_toy.jl <variant>`) -- WHY does reverse
#    mode crash on the real model? ReSEACT's RHS mints helper functions that the
#    toy does not (`rx_native_patch.jl`'s heartbeat lists them:
#    TypeCast_broadcast_scalar, reduce_fnadd_sum, log10_broadcast_scalar,
#    floor_broadcast_scalar, update_computation). Each variant adds ONE of them
#    and re-runs the VJPs. Measured: log10, floor and sumreduce ALL survive,
#    under both Jacobians -- so the ReSEACT segfault
#    (AutoDiffCallRev -> func::CallOp::build -> getAttr on a null FuncOp) is NOT
#    explained by any single helper-minting construct, and the report of it
#    upstream should say so.
#      variants: base log10 floor sumreduce
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io=devnull)
using Reactant, Printf, LinearAlgebra, Random
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end
const EZ = Reactant.Enzyme
include(joinpath(REPO, "tools", "reactant_handoff", "rx_native_patch.jl"))
include(joinpath(REPO, "tools", "reactant_handoff", "rx_traced_integrator.jl"))
const RTI = RxTracedIntegrator

const VARIANT = length(ARGS) >= 1 ? ARGS[1] : "base"
const NS = 3; const NC = 5; const N = NS * NC
const MASKS = [begin m = zeros(N); m[((s-1)*NC+1):(s*NC)] .= 1.0; m end for s in 1:NS]
const ATOL = 1e-9; const RTOL = 1e-4

# the extra term that mints the named helper function
function extra(C)
    VARIANT == "base"      ? 0.0 .* C :
    VARIANT == "log10"     ? 1e-3 .* log10.(C .+ 1.0) :
    VARIANT == "floor"     ? 1e-3 .* floor.(C) :
    VARIANT == "sumreduce" ? 1e-3 .* (sum(C) .* (C ./ C)) :
    error("unknown variant '$VARIANT' (base log10 floor sumreduce)")
end

# cell-local nonlinear chemistry: A+B -> C, C -> A+B, 2C -> ...
function grhs(u, p, t)
    A = u[1:NC]; B = u[(NC+1):(2NC)]; C = u[(2NC+1):(3NC)]
    p1 = p[1:1]; p2 = p[2:2]; p3 = p[3:3]
    ab = A .* B; r1 = p1 .* ab; r2 = p2 .* C; cc = C .* C; r3 = p3 .* cc
    s = 1.0 + 0.1 * sin(t); r1s = s .* r1
    dA = r2 .- r1s
    dB = dA .+ r3
    dC0 = r1s .- r2 .- r3
    dC = dC0 .+ extra(C)
    return vcat(dA, dB, dC)
end

Random.seed!(11)
uh = 0.5 .+ rand(N); ph = [0.7, 0.3, 0.15]
lamh = randn(N); vuh = randn(N); vph = randn(3)
U = RX.ConcreteRArray(uh); P = RX.ConcreteRArray(ph)
LAM = RX.ConcreteRArray(lamh); VU = RX.ConcreteRArray(vuh); VP = RX.ConcreteRArray(vph)
T = RX.ConcreteRNumber(100.0); DT = RX.ConcreteRNumber(0.05)

rout(u, p, t, dt)          = RTI.ros23_step_out(grhs, u, p, t, dt, NS, NC, MASKS, ATOL, RTOL; jac=:ad)
rvjp(u, p, lam, t, dt)     = RTI.ros23_step_vjp(grhs, u, p, t, dt, lam, NS, NC, MASKS, ATOL, RTOL; jac=:ad)
rjvp(u, du, p, dp, t, dt)  = RTI.ros23_step_jvp(grhs, u, du, p, dp, t, dt, NS, NC, MASKS, ATOL, RTOL; jac=:ad)
foutf(u, p, t, dt)         = RTI.ros23_step_out(grhs, u, p, t, dt, NS, NC, MASKS, ATOL, RTOL; jac=:fd)
fvjp(u, p, lam, t, dt)     = RTI.ros23_step_vjp(grhs, u, p, t, dt, lam, NS, NC, MASKS, ATOL, RTOL; jac=:fd)
fjvp(u, du, p, dp, t, dt)  = RTI.ros23_step_jvp(grhs, u, du, p, dp, t, dt, NS, NC, MASKS, ATOL, RTOL; jac=:fd)
sout(u, p, t, dt)          = RTI.ssprk43_step_out(grhs, u, p, t, dt, ATOL, RTOL)
svjp(u, p, lam, t, dt)     = RTI.ssprk43_step_vjp(grhs, u, p, t, dt, lam, ATOL, RTOL)
sjvp(u, du, p, dp, t, dt)  = RTI.ssprk43_step_jvp(grhs, u, du, p, dp, t, dt, ATOL, RTOL)

function check(nm, outf, vjpf, jvpf)
    println("\n==== $nm ====")
    cout = @compile outf(U, P, T, DT)
    cvjp = @compile vjpf(U, P, LAM, T, DT)
    cjvp = @compile jvpf(U, VU, P, VP, T, DT)
    g = cvjp(U, P, LAM, T, DT); lam_in = Array(g[1]); gp = Array(g[2])
    jv = Array(cjvp(U, VU, P, VP, T, DT))
    lhs = dot(lamh, jv); rhs = dot(lam_in, vuh) + dot(gp, vph)
    @printf("  dot   <lam,Jv>=%.16e  <J'lam,v>=%.16e  rel=%.3e %s\n", lhs, rhs,
            abs(lhs - rhs) / max(abs(lhs), abs(rhs)),
            abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < 1e-10 ? "PASS" : "FAIL")
    best = Inf
    for e in (1e-5, 1e-6, 1e-7)
        Up = RX.ConcreteRArray(uh .+ e .* vuh); Um = RX.ConcreteRArray(uh .- e .* vuh)
        Pp = RX.ConcreteRArray(ph .+ e .* vph); Pm = RX.ConcreteRArray(ph .- e .* vph)
        fdv = (Array(cout(Up, Pp, T, DT)) .- Array(cout(Um, Pm, T, DT))) ./ (2e)
        r = abs(dot(lamh, fdv) - rhs) / abs(rhs)
        best = min(best, r)
        @printf("  FD eps=%.0e  rel=%.3e  ||FD-Jv||/||Jv||=%.3e\n", e, r, norm(fdv .- jv) / norm(jv))
    end
    @printf("  FD best %.3e  %s\n", best, best < 1e-6 ? "PASS" : "FAIL")
    s = sprint(show, @code_hlo optimize=false vjpf(U, P, LAM, T, DT))
    nw = count(_ -> true, eachmatch(r"stablehlo\.while", s))
    @printf("  VJP module: lines=%d  stablehlo.while=%d  %s\n",
            count(==('\n'), s), nw, nw == 0 ? "PASS" : "FAIL")
end

println("=== rx_adjoint_toy: variant=$VARIANT, NS=$NS NC=$NC N=$N ===")
_ = @compile sout(U, P, T, DT)     # force the RHS through the tracer once
println("helper functions minted by this RHS: ", Reactant.TracedUtils._CAP_CALLS[], " ",
        sort(collect(Reactant.TracedUtils._NAME_COUNTS); by=last, rev=true))
check("ROS23 jac=:ad", rout, rvjp, rjvp)
check("ROS23 jac=:fd", foutf, fvjp, fjvp)
check("SSPRK43", sout, svjp, sjvp)
println("\nDONE variant=$VARIANT")
