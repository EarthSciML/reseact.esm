#!/usr/bin/env julia
# Horizontal-grid scaling sweep for the full ReSEACT model, all in ONE warm process
# so Julia precompilation + build-machinery JIT are paid once (a warmup build excludes
# the latter from every timed point). Each grid: timed split-build + a short op-split
# (Option A) solve over a fixed window. Emits one RESULT line per grid.
#
# Args: one "label=/abs/path/to/model.esm" per grid (smallest first — it is warmed up).
# Env:  RESEACT_SOLVE_SECS (default 60), RESEACT_RUN_ENV (default <repo>/run-model-jl).
import Pkg
const TOOLS = @__DIR__
const REPO  = dirname(TOOLS)
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io=devnull)
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf, Logging

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
say(s) = (println(s); flush(stdout))

const T0 = 64800.0
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "60"))
const T_END = T0 + SOLVE_SECS
const RTOL = 1e-4; const ATOL = 1e-9
const LU = LinearSolve.LUFactorization()
const PS_REF = 101325.0
rc_ok(rc) = (rc == SciMLBase.ReturnCode.Success || rc == SciMLBase.ReturnCode.Default)

grids = [(String(split(a, '=', limit=2)[1]), String(split(a, '=', limit=2)[2])) for a in ARGS]
isempty(grids) && error("usage: run_sweep.jl label1=/path1.esm label2=/path2.esm ...")

const FO = reseact_forcing(CHEMDIR)   # grid-independent GEOS-FP providers + 72-lev coefs

function build_one(model)
    docs = prepare_split_docs(model)
    tb = time()
    run = build_split_run(docs, (T0, T_END);
        providers = FO.providers, parameters = FO.parameters, const_arrays = FO.const_arrays)
    return run, time() - tb
end

function solve_one(run)
    P = cellmajor_perm(run.var_map)
    f_trans_cm! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
    f_chem_cm!  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
    jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)
    u0 = run.u0[P.sm_of_cm]
    mb = P.base_pos["Transport3D.m"]
    foreach(d -> d.materialize!(), run.dms)
    let dA = Float64.(FO.const_arrays["Transport3D.dA"]), dB = Float64.(FO.const_arrays["Transport3D.dB"])
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
    mf = u[mb:P.NS:P.N]
    o3b = P.base_pos["SuperFast.O3"]; o3 = u[o3b:P.NS:P.N]
    ok = rc_ok(sT.retcode) && rc_ok(sC.retcode) && all(isfinite, u) && all(>(0), mf)
    return (; P, solve_s, rcT=sT.retcode, rcC=sC.retcode, nT=sT.stats.naccept, nC=sC.stats.naccept,
            mmin=minimum(mf), mmax=maximum(mf), o3min=minimum(o3), o3max=maximum(o3), ok)
end

# warmup: build+solve the smallest grid once, discard timing (JIT the build machinery)
say("=== warmup (excluded from results): $(grids[1][1]) ===")
Logging.with_logger(Logging.NullLogger()) do
    r, _ = build_one(grids[1][2]); solve_one(r)
end
say("warmup done\n")

for (label, model) in grids
    say("=== $label : $model ===")
    local run, build_s, res
    Logging.with_logger(Logging.NullLogger()) do
        run, build_s = build_one(model)
        res = solve_one(run)
    end
    say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d build_s=%.2f solve_s=%.2f solve_secs=%.0f nT=%d nC=%d rcT=%s rcC=%s m=[%.3e,%.3e] O3=[%.4e,%.4e] ok=%s",
        label, res.P.NC, res.P.N, res.P.NS, build_s, res.solve_s, SOLVE_SECS,
        res.nT, res.nC, res.rcT, res.rcC, res.mmin, res.mmax, res.o3min, res.o3max, res.ok))
    flush(stdout); GC.gc()
end
say("SWEEP DONE")
