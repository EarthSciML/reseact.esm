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

include(joinpath(@__DIR__, "blockdiag_similar.jl"))   # the densification fix

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

    # Stop grid: macro-dt lattice ∪ forcing cadence boundaries (strictly interior),
    # then the window end. Stopping AT each cadence boundary lets `refresh!` swap
    # the forcing so it is never stale across a boundary within a macro step.
    fstops = Set(round(t; digits = 6) for t in forcing_tstops if t0 + 1e-6 < t < tf - 1e-6)
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
        (tcur < tf - 1e-6 && round(tcur; digits = 6) in fstops) && refresh(tcur)
    end

    subT, subC = integ.child_subintegrators
    return (; u = copy(integ.u),
            retcode = integ.sol.retcode,
            naT = Int(subT.stats.naccept), nrT = Int(subT.stats.nreject),
            naC = Int(subC.stats.naccept), nrC = Int(subC.stats.nreject),
            nmacro = nmacro)
end

end # module OpSplit
using .OpSplit: lie_trotter_solve
