#!/usr/bin/env julia
# ===========================================================================
# run_reseact.jl -- run the full ReSEACT model with the NATIVE in-place runner.
# ===========================================================================
# Builds the operator-split model as two in-place `f!(du,u,p,t)` closures and
# drives a Lie-Trotter operator split entirely on the CPU:
#
#   transport (non-stiff, stencil terms)   -> SSPRK43            (explicit)
#   chemistry (stiff, cell-local pointwise) -> Rosenbrock23      (implicit,
#                                              with a block-diagonal FD Jacobian:
#                                              343 blocks of NS x NS, one per cell)
#
# The two halves share ONE state vector and ONE set of live GEOS-FP forcing
# buffers, so f_full(u) = f_transport(u) + f_chemistry(u) exactly. This is the
# reference runner: it is what the Reactant runner (run_reseact_reactant.jl) is
# validated against.
#
# The Lie-Trotter coupling is done by the REAL SciML operator-splitting solver,
# OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov, via the thin `lie_trotter_solve`
# driver in tools/reactant_handoff/op_split.jl. That package USED to be unusable
# here: it wraps each sub-operator in a plain ODEProblem, whose init() densified
# our BlockDiagonal jac_prototype into a 45864x45864 dense matrix (O(N^3) LU, the
# exact cost the split exists to avoid). The root cause was a single missing
# `similar(::BlockDiagonal, ::Type)` method; tools/reactant_handoff/blockdiag_similar.jl
# supplies it, so the block-diagonal chemistry Jacobian now survives inside the
# solver. See SPLIT_SOLVER.md and op_split.jl for the full story.
#
# (Alternative considered: a single SplitODEProblem(f_chem[implicit,BlockDiagonal],
# f_transport[explicit]) solved by an IMEX method (KenCarp47). It works and has no
# splitting error, but is ~1.7x slower over a 3600 s window because one additive-RK
# integrator's step is paced by the stiff chemistry. It is the accuracy-insurance
# fallback, not the default. See SPLIT_SOLVER.md "Option B".)
#
# Env:
#   RESEACT_MODEL      .esm to run             (default: repo-root reseact.esm)
#   RESEACT_LABEL      tag for the RESULT line (default: basename of the model)
#   RESEACT_SOLVE_SECS solve window, seconds   (default 60)
#   RESEACT_T0         start time, s since 2016-01-01T00:00:00Z (default 64800 =
#                      18:00Z). Also the model's solar epoch, so t IS the UTC clock.
#   RESEACT_MACRO_DT   Lie-Trotter interval, s (default 300, capped at the window)
#   RESEACT_NLON/NLAT/NLEV  grid dims          (bound as .esm metaparameters;
#                           defaults = the model's own: 7 / 7 / 72. Native GEOS-FP
#                           4x5 slice bounds: NLON 1..57, NLAT 1..17, NLEV 1..72.)
#   RESEACT_RUN_ENV    Julia env to activate   (default: <repo>/run-model-jl)
#
# Helper code it pulls in (see HELPERS.md for the migration plan):
#   prototypes/reseact_3d_chem/split_common.jl    build_split_run, reseact_forcing
#   prototypes/reseact_3d_chem/blockdiag_local.jl  BlockDiagonal (from EarthSciMLBase)
#   prototypes/reseact_3d_chem/block_jac.jl        cellmajor_perm/rhs, block_fd_jac
#   tools/reactant_handoff/op_split.jl             lie_trotter_solve (+ BlockDiagonal fix)
# ===========================================================================
import Pkg
const REPO = @__DIR__
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)

# The kernel-class merge is default-ON in EarthSciAST (since the IIP hoist,
# EarthSciAST 2fb930d6): it collapses per-cell-fragmented kernels into
# lane-batched classes, which is what keeps the IR from growing with the grid.
#
# This runner USED to force it off. That was measured on a 7x7x7 box, where the
# merged codegen tier's first-call compile (hundreds of seconds) swamped a ~25 s
# interpreted solve — a real result, but for a grid small enough that the
# unmerged IR was affordable in the first place. At CONUS 13x7x72 that trade
# reverses: the unmerged per-cell IR is the thing that does not fit, and paying a
# fixed compile to keep the kernel count grid-independent is the whole point of
# the merge. Leave it ON and let the runner's own bench counters show the cost;
# ESS_KERNEL_CLASS_MERGE_DISABLE=1 still forces the old behaviour for a
# small-grid A/B.
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))                    # prepare_split_docs, reseact_forcing, build_split_run
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))                       # cellmajor_perm/rhs, block_fd_jac
include(joinpath(RXDIR, "op_split.jl"))                          # lie_trotter_solve (+ BlockDiagonal similar fix)
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize   # slice_hybrid_coefs (NLEV<72)
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", basename(MODEL))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "60"))
const MACRO_DT   = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
# Grid dims as .esm metaparameters (empty dict -> the model's own defaults 7/7/72).
const GRID_MP = Dict{String,Int}(k => parse(Int, ENV["RESEACT_$k"])
                                 for k in ("NLON", "NLAT", "NLEV") if haskey(ENV, "RESEACT_$k"))
const NLEV_EFF = get(GRID_MP, "NLEV", 72)
# Solver time is seconds since 2016-01-01T00:00:00Z -- the GEOS-FP day the forcing
# is read from AND the epoch the model's solar chain (Transport3D.doy0/hour0/...)
# is anchored to, so t IS the UTC clock. Default 64800 = 18:00Z (near local solar
# noon over the slice). A diurnal run wants T0=0; the forcing is only valid while
# both bracketing records exist -- see reseact_forcing for the usable span.
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "64800"))
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
say("=== $LABEL : validate + build split ($MODEL) grid=$(get(GRID_MP,"NLON",7))x$(get(GRID_MP,"NLAT",7))x$(NLEV_EFF) ===")
# ONE known-false diagnostic is exempted, by exact shape, and nothing else.
#
# `manifold` is a SCALAR-FIELD substitution site (esm-spec §9.6.1): a geometry
# kernel's manifold is bound to the LITERAL "planar"/"spherical"/"geodesic", and
# §9.6.9 discharges its admissibility on the EXPANDED form. But validate's
# reference walk descends `apply_expression_template` bindings as expressions, so
# it parses the literal as a bare variable and reports it undefined. This is not
# reseact's: it reproduces on a four-variable fixture that contains one
# conservative_overlap call and nothing else, and wildlandfire.esm's Era5Regrid
# spells its manifold binding identically. The tree-walk BUILD is unaffected --
# it expands the template and sees the scalar field, which is why NEIRegrid
# builds and runs. Exempted narrowly (this error_type, this literal set) so a
# genuinely undefined variable anywhere else still stops the run.
const _MANIFOLDS = ("planar", "spherical", "geodesic")
_is_manifold_false_positive(e) =
    e.error_type == "undefined_variable" &&
    any(m -> occursin("Variable '$m' referenced", e.message), _MANIFOLDS)

let r = isempty(GRID_MP) ? EA.validate(MODEL) : EA.validate(EA.load(MODEL; metaparameters = GRID_MP))
    real_errors = filter(!_is_manifold_false_positive, r.structural_errors)
    exempted = length(r.structural_errors) - length(real_errors)
    exempted == 0 || say("  ($exempted manifold scalar-field diagnostic(s) exempted; see note above)")
    isempty(real_errors) || (for e in real_errors[1:min(6, end)]
                                 say("  $(e.error_type)@$(e.path): $(e.message)")
                             end; error("invalid model"))
end
docs = prepare_split_docs(MODEL; metaparameters = GRID_MP)
ff = reseact_forcing(CHEMDIR)               # 72-level hybrid coefs + native GEOS-FP providers
# When NLEV<72 the `lev` axis is shorter than the 72-entry hybrid table; slice the
# vertical coefs to match (a truncated column, k=1 = surface).
ff = merge(ff, (; const_arrays = GridResize.slice_hybrid_coefs(ff.const_arrays, NLEV_EFF)))
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
# Seed air mass m(0) = dA[k] + dB[k]*PS(i,j,T0) -- the REAL GEOS-FP surface
# pressure, not a constant. Seeding from a constant PS_REF (as this did) starts m
# 16% RMS / 48% worst-column away from the dp it is supposed to equal, purely
# from terrain. See hydrostatic_dp in split_common.jl.
let dp0 = hydrostatic_dp(run.merged_param, ff.const_arrays, T0; slice = run.slice)
    for c in P.cells
        u0[(P.cell_pos[c] - 1) * P.NS + mb] = dp0(c[1], c[2], c[3])
    end
end

# --------------------------------------------------------------------------- #
# 3. Lie-Trotter operator split via LieTrotterGodunov((SSPRK43, Rosenbrock23/BD)).
#    `lie_trotter_solve` steps the split by MACRO_DT: SSPRK43 advances transport
#    over the interval, then Rosenbrock23 advances chemistry FROM that state with
#    the block-diagonal Jacobian (preserved by the blockdiag_similar fix). The
#    driver clamps to >= 0 after each macro step (PositiveDomain substitute) and
#    refreshes the shared GEOS-FP forcing buffers at each cadence boundary
#    (run.tstops) -- the operator-splitting package applies neither callbacks nor
#    inner tolerances itself, so both are driven from the macro loop.
# --------------------------------------------------------------------------- #
tgz(g, u, p, t) = (fill!(g, 0); nothing)   # chem tgrad is zero (autonomous over a step)
fc = SciMLBase.ODEFunction(f_chem_cm!; jac = jac_cm!, jac_prototype = mkjp(), tgrad = tgz)
inner_algs = (OrdinaryDiffEqSSPRK.SSPRK43(),
              OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false, linsolve = LU))

# Forcing refresh at cadence boundaries: rewrite the live buffers in run.merged_param
# from the discrete providers at time t, then rebuild the derived discrete caches --
# the functional equivalent of run.cb's affect!, which the split solver cannot fire.
disc = Dict(String(k) => prov for (k, prov) in ff.providers if !EA.provider_is_const(prov))
refresh_forcing = t -> begin
    for (k, prov) in disc
        run.merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), run.dms)
end

ts = time()
res = lie_trotter_solve(f_trans_cm!, fc, u0, (T0, T_END), run.p, inner_algs;
    macro_dt = min(MACRO_DT, SOLVE_SECS),
    reltols = (RTOL, RTOL), abstols = (ATOL_T, ATOL),   # transport / chemistry
    refresh = refresh_forcing, forcing_tstops = run.tstops, clamp_nonneg = true)
u = res.u
solve_s = time() - ts

# --------------------------------------------------------------------------- #
# 4. Report.
# --------------------------------------------------------------------------- #
mf = u[mrng()]
o3 = u[P.base_pos["SuperFast.O3"]:P.NS:P.N]
ok = rc_ok(res.retcode) && all(isfinite, u) && all(>(0), mf)
say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d solve_s=%.2f solve_secs=%.0f macro_dt=%.0f nmacro=%d nT=%d nC=%d rc=%s m=[%.3e,%.3e] O3=[%.4e,%.4e] ok=%s",
    LABEL, P.NC, P.N, P.NS, solve_s, SOLVE_SECS, min(MACRO_DT, SOLVE_SECS), res.nmacro,
    res.naT, res.naC, res.retcode, minimum(mf), maximum(mf), minimum(o3), maximum(o3), ok))
say("DONE $LABEL")
