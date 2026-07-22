#!/usr/bin/env julia
# ===========================================================================
# run_reseact.jl -- run the full ReSEACT model with the NATIVE in-place runner.
# ===========================================================================
# Builds the operator-split model as two in-place `f!(du,u,p,t)` closures and
# drives the Lie-Trotter scheme entirely on the CPU:
#
#   transport (non-stiff, stencil terms)   -> SSPRK43            (explicit)
#   chemistry (stiff, cell-local pointwise) -> Rosenbrock23      (implicit,
#                                              with a block-diagonal FD Jacobian:
#                                              343 blocks of NS x NS, one per cell)
#
# The two halves share ONE state vector and ONE set of live GEOS-FP forcing
# buffers, so f_full(u) = f_transport(u) + f_chemistry(u) exactly and a single
# refresh callback drives both. This is the reference runner: it is what the
# Reactant runner (run_reseact_reactant.jl) is validated against.
#
# Why a HAND-ROLLED Lie-Trotter loop and not OrdinaryDiffEqOperatorSplitting.jl or
# a single SplitODEProblem(transport, chemistry)? Both were tried and measured
# (prototypes/reseact_3d_chem/SPLIT_SOLVER.md); the block-diagonal chemistry
# Jacobian is what rules them out:
#   * OrdinaryDiffEqOperatorSplitting.jl wraps each sub-operator in a *plain*
#     ODEProblem, whose init() silently replaces our BlockDiagonal jac_prototype
#     with a dense 45864x45864 matrix -> O(N^3) LU, the exact cost the split exists
#     to avoid. (Still true on the package's current main, verified 2026-07-22.)
#   * A true SplitODEProblem(f_chem[implicit,BlockDiagonal], f_transport[explicit])
#     solved by an IMEX method (KenCarp47) DOES work and has no splitting error,
#     but is ~1.7x slower (1235 s vs 710 s over a 3600 s window): one additive-RK
#     integrator whose step is paced by the stiff chemistry, so transport can't take
#     its own large explicit steps. It is the accuracy-insurance fallback, not the
#     default. See SPLIT_SOLVER.md "Option B".
# So we drive each operator with the integrator it wants, coupled by a first-order
# Lie-Trotter step over the window.
#
# Env:
#   RESEACT_MODEL      .esm to run             (default: repo-root reseact.esm)
#   RESEACT_LABEL      tag for the RESULT line (default: basename of the model)
#   RESEACT_SOLVE_SECS solve window, seconds   (default 60, one Lie-Trotter step)
#   RESEACT_RUN_ENV    Julia env to activate   (default: <repo>/run-model-jl)
#
# Helper code it pulls in (see HELPERS.md for the migration plan):
#   prototypes/reseact_3d_chem/split_common.jl    build_split_run, reseact_forcing
#   prototypes/reseact_3d_chem/blockdiag_local.jl  BlockDiagonal (from EarthSciMLBase)
#   prototypes/reseact_3d_chem/block_jac.jl        cellmajor_perm/rhs, block_fd_jac
# ===========================================================================
import Pkg
const REPO = @__DIR__
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)

# The kernel-class merge is default-ON in EarthSciAST (since the IIP hoist,
# EarthSciAST 2fb930d6). It exists to shrink the IR for Reactant TRACING; the
# native CPU runner does not want it. With the merge on, the :inplace RHS goes
# through the codegen tier, whose first-call compile of the (large, merged)
# transport kernels is expensive. That compile is now MEMORY-bounded and the
# codegen RHS is zero-alloc (EarthSciAST codegen function-barrier fix,
# codegen_kernel.jl), so it is correct and no longer OOMs — but the first-call
# compile still dominates a single short CPU solve (hundreds of seconds) versus
# ~25 s for the unmerged INTERPRETED RHS. So disable the merge here unless the
# user explicitly sets the flag (ESS_KERNEL_CLASS_MERGE_DISABLE=0 forces it on).
haskey(ENV, "ESS_KERNEL_CLASS_MERGE_DISABLE") || (ENV["ESS_KERNEL_CLASS_MERGE_DISABLE"] = "1")
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))                    # prepare_split_docs, reseact_forcing, build_split_run
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))                       # cellmajor_perm/rhs, block_fd_jac
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", basename(MODEL))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "60"))
const T0    = 64800.0
const T_END = T0 + SOLVE_SECS
const RTOL  = 1e-4
const ATOL  = 1e-9          # chemistry abstol
const ATOL_T = 1e-6         # transport abstol
const LU    = LinearSolve.LUFactorization()
const PS_REF = 101325.0
rc_ok(rc) = (rc == SciMLBase.ReturnCode.Success || rc == SciMLBase.ReturnCode.Default)

# --------------------------------------------------------------------------- #
# 1. Split + build both in-place halves over shared live GEOS-FP forcing.
# --------------------------------------------------------------------------- #
say("=== $LABEL : validate + build split ($MODEL) ===")
let r = EA.validate(MODEL)
    r.is_valid || (for e in r.structural_errors[1:min(6, end)]
                       say("  $(e.error_type)@$(e.path): $(e.message)")
                   end; error("invalid model"))
end
docs = prepare_split_docs(MODEL)
ff = reseact_forcing(CHEMDIR)               # 72-level hybrid coefs + native GEOS-FP providers
tb = time()
run = build_split_run(docs, (T0, T_END);
    providers = ff.providers, parameters = ff.parameters, const_arrays = ff.const_arrays)
say(@sprintf("BUILD %s: %.2f s   nstates=%d", LABEL, time() - tb, length(run.u0)))

# --------------------------------------------------------------------------- #
# 2. Cell-major reorder + block-diagonal chem Jacobian (the block-diagonal
#    structure is only contiguous once the state is permuted species->cell major).
# --------------------------------------------------------------------------- #
P = cellmajor_perm(run.var_map)
f_trans_cm! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)

u0 = run.u0[P.sm_of_cm]
mb = P.base_pos["Transport3D.m"]
mrng() = mb:P.NS:P.N
foreach(d -> d.materialize!(), run.dms)       # prime discrete (forcing-derived) caches
# Seed air mass m(0) = dA + dB*ps_ref per level (hydrostatic column mass).
let dA = Float64.(ff.const_arrays["Transport3D.dA"]), dB = Float64.(ff.const_arrays["Transport3D.dB"])
    for c in P.cells
        u0[(P.cell_pos[c] - 1) * P.NS + mb] = dA[c[3]] + dB[c[3]] * PS_REF
    end
end

# --------------------------------------------------------------------------- #
# 3. Lie-Trotter step: transport (SSPRK43) then chemistry (Rosenbrock23/BlockDiag).
#    PositiveDomain keeps concentrations >= 0; the shared refresh callback (run.cb)
#    rewrites forcing buffers at each GEOS-FP cadence boundary.
#
#    The two sub-steps below TOGETHER realize f_full = f_transport + f_chemistry
#    over the window: SSPRK43 advances transport, then Rosenbrock23 advances
#    chemistry FROM that transported state. The chemistry sub-step is wrapped in a
#    SplitODEProblem(fc, zerof!) NOT to add a second operator but as a plumbing
#    trick: a jac_prototype (our BlockDiagonal) is honored only inside a
#    SplitODEProblem -- a plain ODEProblem's init() densifies it. The `zerof!` is
#    the empty second slot; transport is not dropped, it was already applied above.
# --------------------------------------------------------------------------- #
zerof!(du, u, p, t) = (fill!(du, 0); nothing)
tgz(g, u, p, t)     = (fill!(g, 0); nothing)   # chem tgrad is zero (autonomous over a step)
ts = time()
u = copy(u0)
pT = SciMLBase.ODEProblem(f_trans_cm!, u, (T0, T_END), run.p)
sT = SciMLBase.solve(pT, OrdinaryDiffEqSSPRK.SSPRK43(); reltol = RTOL, abstol = ATOL_T,
    callback = SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u))),
    tstops = run.tstops, save_everystep = false, maxiters = 500000)
u = sT.u[end]
fc = SciMLBase.ODEFunction(f_chem_cm!; jac = jac_cm!, jac_prototype = mkjp(), tgrad = tgz)
pC = SciMLBase.SplitODEProblem(fc, zerof!, u, (T0, T_END), run.p)   # SplitODEProblem so jac_prototype reaches jac!
sC = SciMLBase.solve(pC, OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false, linsolve = LU);
    reltol = RTOL, abstol = ATOL, callback = DiffEqCallbacks.PositiveDomain(copy(u)),
    save_everystep = false, maxiters = 500000)
u = sC.u[end]
solve_s = time() - ts

# --------------------------------------------------------------------------- #
# 4. Report.
# --------------------------------------------------------------------------- #
mf = u[mrng()]
o3 = u[P.base_pos["SuperFast.O3"]:P.NS:P.N]
ok = rc_ok(sT.retcode) && rc_ok(sC.retcode) && all(isfinite, u) && all(>(0), mf)
say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d solve_s=%.2f solve_secs=%.0f nT=%d nC=%d rcT=%s rcC=%s m=[%.3e,%.3e] O3=[%.4e,%.4e] ok=%s",
    LABEL, P.NC, P.N, P.NS, solve_s, SOLVE_SECS, sT.stats.naccept, sC.stats.naccept,
    sT.retcode, sC.retcode, minimum(mf), maximum(mf), minimum(o3), maximum(o3), ok))
say("DONE $LABEL")
