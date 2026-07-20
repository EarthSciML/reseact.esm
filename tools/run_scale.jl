#!/usr/bin/env julia
# Build + short op-split (Option A) solve of the full ReSEACT model at one grid.
# Reuses the Stage-C split machinery (prototypes/reseact_3d_chem/{split_common,
# block_jac,blockdiag_local}.jl). Emits one machine-readable RESULT line.
#
# Env:
#   RESEACT_MODEL      path to the full .esm to build (default: repo-root reseact.esm)
#   RESEACT_LABEL      tag for the RESULT line (default: basename of the model)
#   RESEACT_SOLVE_SECS solve window in seconds (default 60), single Lie-Trotter step
#   RESEACT_RUN_ENV    Julia env to activate (default: <repo>/run-model-jl)
import Pkg
const TOOLS = @__DIR__
const REPO  = dirname(TOOLS)
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io=devnull)
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL = get(ENV, "RESEACT_LABEL", basename(MODEL))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "60"))
const T0 = 64800.0
const T_END = T0 + SOLVE_SECS
const RTOL = 1e-4; const ATOL = 1e-9
const LU = LinearSolve.LUFactorization()
const PS_REF = 101325.0
rc_ok(rc) = (rc == SciMLBase.ReturnCode.Success || rc == SciMLBase.ReturnCode.Default)

say("=== $LABEL : validate $MODEL ===")
let r = EA.validate(MODEL)
    r.is_valid || (for e in r.structural_errors[1:min(6,end)]; say("  $(e.error_type)@$(e.path): $(e.message)"); end; error("invalid"))
end

docs = prepare_split_docs(MODEL)
fo = reseact_forcing(CHEMDIR)   # 72-level hybrid coefs + native GEOS-FP providers (grid-independent)
tb = time()
run = build_split_run(docs, (T0, T_END);
    providers = fo.providers, parameters = fo.parameters, const_arrays = fo.const_arrays)
build_s = time() - tb
say(@sprintf("BUILD %s: %.2f s   nstates=%d", LABEL, build_s, length(run.u0)))

P = cellmajor_perm(run.var_map)
f_trans_cm! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)
u0 = run.u0[P.sm_of_cm]
mb = P.base_pos["Transport3D.m"]
mrng() = mb:P.NS:P.N
foreach(d -> d.materialize!(), run.dms)
let dA = Float64.(fo.const_arrays["Transport3D.dA"]), dB = Float64.(fo.const_arrays["Transport3D.dB"])
    for c in P.cells
        u0[(P.cell_pos[c]-1)*P.NS + mb] = dA[c[3]] + dB[c[3]]*PS_REF
    end
end

zerof!(du,u,p,t) = (fill!(du,0); nothing)
tgz(g,u,p,t)     = (fill!(g,0); nothing)
ts = time()
u = copy(u0)
pT = SciMLBase.ODEProblem(f_trans_cm!, u, (T0, T_END), run.p)
sT = SciMLBase.solve(pT, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
    callback=SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u))),
    tstops=run.tstops, save_everystep=false, maxiters=500000)
u = sT.u[end]
f1c = SciMLBase.ODEFunction(f_chem_cm!; jac=jac_cm!, jac_prototype=mkjp(), tgrad=tgz)
pC = SciMLBase.SplitODEProblem(f1c, zerof!, u, (T0, T_END), run.p)
sC = SciMLBase.solve(pC, OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
    reltol=RTOL, abstol=ATOL, callback=DiffEqCallbacks.PositiveDomain(copy(u)),
    save_everystep=false, maxiters=500000)
u = sC.u[end]
solve_s = time() - ts

mf = u[mrng()]
o3b = P.base_pos["SuperFast.O3"]; o3 = u[o3b:P.NS:P.N]
ok = rc_ok(sT.retcode) && rc_ok(sC.retcode) && all(isfinite, u) && all(>(0), mf)
say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d build_s=%.2f solve_s=%.2f solve_secs=%.0f nT=%d nC=%d rcT=%s rcC=%s m=[%.3e,%.3e] O3=[%.4e,%.4e] ok=%s",
    LABEL, P.NC, P.N, P.NS, build_s, solve_s, SOLVE_SECS, sT.stats.naccept, sC.stats.naccept,
    sT.retcode, sC.retcode, minimum(mf), maximum(mf), minimum(o3), maximum(o3), ok))
say("DONE $LABEL")
