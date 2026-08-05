# ===========================================================================
# op_split.jl -- drive the ReSEACT Lie-Trotter split with the REAL SciML
# operator-splitting solver, OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov.
# ===========================================================================
# This replaces the hand-rolled transport-then-chemistry step the runners used
# to carry. It is now usable because `blockdiag_similar.jl` (included below)
# stops LieTrotterGodunov's plain-`ODEProblem` wrapping from densifying the
# block-diagonal chemistry Jacobian at `init` (see that file for the root cause).
#
#   f_full(u) = f_transport(u) + f_chemistry(u)
#     transport (non-stiff, stencil)  -> SSPRK43            (explicit, adaptive)
#     chemistry (stiff, cell-local)   -> Rosenbrock23/BD    (implicit, adaptive,
#                                        block-diagonal FD Jacobian preserved)
#
# GenericSplitFunction((f_trans, f_chem), (1:N, 1:N)) : both operators act on the
# whole state; LieTrotterGodunov steps transport by the macro dt, writes back,
# then steps chemistry from that state -- a first-order Lie-Trotter step.
#
# Why a macro-step driver loop instead of a bare `solve(prob, alg)`:
# OrdinaryDiffEqOperatorSplitting (v0.3.2) does NOT apply discrete callbacks
# during its step loop and does NOT forward reltol/abstol to the inner solvers.
# So the two things the ReSEACT solve needs -- positivity and forcing refresh at
# GEOS-FP cadence boundaries -- have to be driven here, between macro steps:
#   * `adaptive=true` + `verbose=false` : the OUTER algorithm is not adaptive
#     (LieTrotter is fixed-macro-dt), but this flag flows down to the INNER
#     integrators so SSPRK43/Rosenbrock23 sub-step adaptively inside each macro
#     step (without it they take one fixed step of size macro_dt and blow up).
#   * clamp to >= 0 after each macro step  -- PositiveDomain substitute.
#   * `refresh!(t)` at cadence boundaries  -- rewrites the shared forcing buffers
#     (the runner supplies a closure; boundaries come from `forcing_tstops`).
# Inner tolerances take OrdinaryDiffEq defaults (reltol 1e-3, abstol 1e-6); the
# package gives no hook to set them per operator. `macro_dt` is the Lie-Trotter
# splitting interval (first-order splitting error grows ~linearly with it).
#
# Requires `BlockDiagonal` in scope (runner does `using .BlockDiag` first) so the
# similar-patch can attach.

import OrdinaryDiffEqOperatorSplitting as OS
import OrdinaryDiffEqCore

include(joinpath(@__DIR__, "blockdiag_similar.jl"))   # the densification fix

# --------------------------------------------------------------------------- #
# UPSTREAM BUG SHIM: a REJECTED outer step crashes OrdinaryDiffEqOperatorSplitting.
#
# `step_footer!` (integrator.jl:635) takes the rejection branch and calls
#   OrdinaryDiffEqCore.post_newton_controller!(integrator, integrator.alg)
# whose 2-arg method (OrdinaryDiffEqCore controllers.jl:123) forwards through
#   integrator.opts.controller
# -- but this package's own `IntegratorOptions` has no `controller` field (only
# adaptive / dtmin / dtmax / failfactor / verbose / isoutofdomain), so the call
# dies with
#   FieldError: type IntegratorOptions has no field `controller`
#
# The 3-arg method it is trying to reach IGNORES the controller entirely:
#   post_newton_controller!(integrator, controller, alg) = (integrator.dt /= failfactor)
# so the 2-arg indirection is pure overhead on this integrator, and skipping
# straight to that body is exact, not an approximation.
#
# This is latent for a short run and fatal for a long one: nothing rejects during
# a 60 s validation window, but a multi-hour run hits a rejected step eventually
# and dies mid-solve (first seen at t = 3.6 h of a 19.5 h diurnal run). It is NOT
# specific to the diurnal driver -- run_reseact.jl is exposed to the same crash.
#
# Type piracy, like blockdiag_similar.jl next door; durable home is upstream
# OrdinaryDiffEqOperatorSplitting (either give IntegratorOptions a `controller`
# field or define this method there).
OrdinaryDiffEqCore.post_newton_controller!(integrator::OS.AnySplitIntegrator, alg) =
    (integrator.dt = integrator.dt / integrator.opts.failfactor; nothing)

module OpSplit
import ..OS
import SciMLBase
# `step!` is CommonSolve's generic, re-exported by SciMLBase; the operator-
# splitting package defines its `step!(::AnySplitIntegrator, dt, stop)` method on
# that same function, so SciMLBase.step! dispatches to it (DiffEqBase is not a
# direct dep of this env, so we cannot `import` it here).
const step! = SciMLBase.step!

"""
    lie_trotter_solve(f_trans!, fc, u0, tspan, p, inner_algs; kw...) -> NamedTuple

Advance `u0` over `tspan` with LieTrotterGodunov((alg_transport, alg_chem)).

- `f_trans!(du,u,p,t)` : in-place transport RHS (wrapped in an `ODEFunction`).
- `fc::ODEFunction`    : chemistry RHS carrying `jac` + BlockDiagonal `jac_prototype`.
- `p`                  : parameters passed to BOTH operators (as `(p, p)`).
- `inner_algs`         : `(alg_transport, alg_chem)`, e.g. `(SSPRK43(), Rosenbrock23(...))`.

Keywords: `macro_dt` (Lie-Trotter interval, required); `reltols`/`abstols` (tuples,
one entry per operator, applied to the inner integrators after init -- the package
forwards no tolerances, so without these the inner solvers use OrdinaryDiffEq
defaults reltol 1e-3 / abstol 1e-6); `refresh(t)` (forcing refresh at cadence
boundaries; default no-op), `forcing_tstops` (cadence boundary times), `clamp_nonneg`
(clamp state to >= 0 after each macro step; default true).

Returns `(; u, retcode, naT, nrT, naC, nrC, nmacro)` -- end state, outer retcode,
and accepted/rejected inner step counts for transport (T) and chemistry (C).
"""
function lie_trotter_solve(f_trans!, fc, u0, tspan, p, inner_algs;
                           macro_dt::Real,
                           reltols = nothing, abstols = nothing,
                           refresh = (_t -> nothing),
                           forcing_tstops = Float64[],
                           clamp_nonneg::Bool = true)
    N   = length(u0)
    idx = Base.OneTo(N)                          # both operators see the full state
    gsf = OS.GenericSplitFunction(
        (SciMLBase.ODEFunction(f_trans!), fc), (idx, idx))
    t0  = Float64(tspan[1]); tf = Float64(tspan[2])
    prob = OS.OperatorSplittingProblem(gsf, collect(float(u0)), (t0, tf), (p, p))
    integ = SciMLBase.init(prob, OS.LieTrotterGodunov(inner_algs);
                           dt = Float64(macro_dt), adaptive = true, verbose = false)

    # The operator-splitting package forwards no reltol/abstol to the inner
    # integrators; set them directly on the (mutable) child DEOptions post-init.
    for (i, ch) in enumerate(integ.child_subintegrators)
        reltols === nothing || (ch.opts.reltol = reltols[i])
        abstols === nothing || (ch.opts.abstol = abstols[i])
    end

    # Stop grid: macro-dt lattice ∪ forcing cadence boundaries, then the window end.
    # Stopping AT each cadence boundary lets `refresh` swap the forcing so it is
    # never stale across a boundary.
    #
    # The interval is HALF-OPEN ON THE RIGHT — `t0 < x <= tf`, not `t0 < x < tf`.
    # Strict interiority looks like the safe choice and is in fact a silent
    # forcing-freeze: callers drive this ONE MACRO STEP AT A TIME, so a boundary
    # that lands on a macro-step edge is excluded from the step that ENDS there
    # (not < tf) and again from the step that STARTS there (not > t0), and is
    # never refreshed at all. Every GEOS-FP tstop is a multiple of 1800 (I3 at
    # 10800k, A3 at 5400+10800k, A1 at 1800+3600k), so with the usual macro_dt=300
    # and a T0 on the cadence, EVERY boundary was skipped: measured 0 refreshes
    # over a 9 h run, i.e. U/V/PS/T/PBLH frozen at t0 for the whole simulation.
    # (macro_dt=250, which does not divide 1800, fired 9 times — which is what made
    # it invisible in ad-hoc testing.)
    #
    # Refreshing AT tf is correct, not a fencepost fudge: at a boundary time both
    # brackets are valid, since the old bracket's right record and the new
    # bracket's left record are the same record. Swapping there hands the next
    # macro step fresh forcing. It cannot double-fire — the next call has that
    # instant as its t0, which the strict left bound excludes.
    fstops = Set(round(t; digits = 6) for t in forcing_tstops if t0 + 1e-6 < t <= tf + 1e-6)
    grid   = collect((t0 + macro_dt):macro_dt:(tf - 1e-9))
    stops  = sort!(unique!(vcat(grid, collect(fstops), Float64[tf])))

    tcur = t0
    nmacro = 0
    for tnext in stops
        tnext <= tcur + 1e-9 && continue
        step!(integ, tnext - tcur, true)                 # inner solvers sub-step adaptively
        clamp_nonneg && (integ.u .= max.(integ.u, 0.0))
        tcur = integ.t
        nmacro += 1
        round(tcur; digits = 6) in fstops && refresh(tcur)
    end

    subT, subC = integ.child_subintegrators
    return (; u = copy(integ.u),
            retcode = integ.sol.retcode,
            naT = Int(subT.stats.naccept), nrT = Int(subT.stats.nreject),
            naC = Int(subC.stats.naccept), nrC = Int(subC.stats.nreject),
            nmacro = nmacro)
end

_ltg_ok(r) = r.retcode == SciMLBase.ReturnCode.Success ||
             r.retcode == SciMLBase.ReturnCode.Default

"""
    lie_trotter_solve_bisect(f_trans!, fc, u0, tspan, p, inner_algs;
                             macro_dt, min_macro_dt = 9.0, kwargs...)

`lie_trotter_solve`, but a FAILED macro step is retried over two half-intervals
instead of aborting the run. Recurses until the sub-interval would fall below
`min_macro_dt`, then propagates the failure.

WHY THIS EXISTS, and why it is not a clamp. The chemistry dies at the pre-dawn
NO minimum: the inner Rosenbrock23 proposes TRIAL sub-steps that take species
negative, the RHS is evaluated there (rate = k * [negative]) and convergence
collapses. MEASURED on the reproduced 11.33 h failure -- `clamp_nonneg` is INERT
against this (turning it off is bit-identical, and both arms end `nneg=0`),
because a macro-boundary clamp only ever inspects ACCEPTED states and what goes
negative is a trial state inside the step. Only shortening the step can prevent
it, which is what `PositiveDomain`/`isoutofdomain` do -- they reject the step and
retry smaller, they do NOT modify values. Measured: the failing step succeeds at
macro_dt 60 and at 30.

Doing it HERE rather than with the real callback is forced. OrdinaryDiffEq-
OperatorSplitting v0.3.2 accepts a `callback` at both levels and applies it at
NEITHER: the outer integrator stores, initializes and finalizes a CallbackSet but
never applies it in its step loop (no `apply_discrete_callback!` anywhere in the
package), and the leaf builder takes `callback` as a parameter and then omits it
from its `SciMLBase.__init` call. So `callback = PositiveDomain(...)` is silently
IGNORED today -- worse than an error, since it looks like it worked. See
HELPERS.md for the one-line upstream fix and how to validate it.

Bisecting only pays on the steps that actually need it, unlike lowering
`macro_dt` globally (which would cost 5-10x more macro steps across the whole run
to rescue a handful of pre-dawn ones). A retry is not free -- the failed attempt
is wasted work -- but it is bounded by `min_macro_dt`.

Each retry calls `lie_trotter_solve`, which builds a FRESH integrator, so no
failed-integrator state has to be reset. Note the splitting error grows ~linearly
with the interval, so sub-stepping is also the more accurate arm.

LIMITATION -- CALL THIS ONE MACRO STEP AT A TIME. Bisection halves the whole
`tspan`, not the individual macro step that failed, because a failed integrator
cannot be resumed mid-window. `tools/diurnal_run.jl` drives it correctly: it
calls once per macro step with `tspan = (t, tnext)` and `macro_dt = tnext - t`,
so the halved interval IS the failed step. Handed a MULTI-STEP window it is still
correct but wasteful -- it discards every macro step already completed in that
window and redoes them at half `macro_dt`. `run_reseact.jl` passes a whole
`(T0, T_END)` window and is deliberately NOT switched over: its windows are short
(a validation run), and for a long one this would be the wrong shape. Making it
resumable needs the per-step retry to live inside the macro loop, which in turn
needs `reinit!` of a failed split integrator -- or, better, the upstream callback
fix that makes all of this unnecessary (HELPERS.md §3).
"""
function lie_trotter_solve_bisect(f_trans!, fc, u0, tspan, p, inner_algs;
                                  macro_dt, min_macro_dt::Real = 9.0,
                                  on_bisect = (_t0, _tf) -> nothing, kwargs...)
    t0 = Float64(tspan[1]); tf = Float64(tspan[2])
    r = lie_trotter_solve(f_trans!, fc, u0, (t0, tf), p, inner_algs;
                          macro_dt = macro_dt, kwargs...)
    (_ltg_ok(r) || (tf - t0) <= min_macro_dt + 1e-9) && return r
    tm = 0.5 * (t0 + tf)
    on_bisect(t0, tf)
    half = min(Float64(macro_dt), tm - t0)
    r1 = lie_trotter_solve_bisect(f_trans!, fc, u0, (t0, tm), p, inner_algs;
                                  macro_dt = half, min_macro_dt, on_bisect, kwargs...)
    _ltg_ok(r1) || return r1
    r2 = lie_trotter_solve_bisect(f_trans!, fc, r1.u, (tm, tf), p, inner_algs;
                                  macro_dt = min(Float64(macro_dt), tf - tm),
                                  min_macro_dt, on_bisect, kwargs...)
    _ltg_ok(r2) || return r2
    # Counts are summed across the halves; the wasted failed attempt is NOT
    # subtracted, so naC reflects the true work done.
    return (; u = r2.u, retcode = r2.retcode,
            naT = r1.naT + r2.naT, nrT = r1.nrT + r2.nrT,
            naC = r1.naC + r2.naC, nrC = r1.nrC + r2.nrC,
            nmacro = r1.nmacro + r2.nmacro)
end

end # module OpSplit
using .OpSplit: lie_trotter_solve, lie_trotter_solve_bisect
