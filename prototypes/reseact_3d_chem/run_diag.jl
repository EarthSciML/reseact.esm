#!/usr/bin/env julia
# One build (~25 min), then DIAGNOSE the instability: is the TRANSPORT stiff
# (explicit fails) or is the coupled failure a settings/coupling artifact?
#   1. per-state transport tendency + implied timescale (find the stiff states/cells)
#   2. air-mass m per cell (thin near-surface layers => large 1/m => stiff advection)
#   3. transport-ONLY explicit solve (SSPRK43) at 60/300 s, retcode checked
#   4. chem-ONLY implicit solve (confirm still stable)
#   5. transport-ONLY IMPLICIT via a sparse-stencil Jacobian? (report only)
import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io=devnull)
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqSSPRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSDIRK
import LinearSolve
using LinearAlgebra, Printf
const DIR = @__DIR__
include(joinpath(DIR, "split_common.jl"))
include(joinpath(DIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(DIR, "block_jac.jl"))
say(s)=(println(s); flush(stdout))
const MODEL = joinpath(DIR, "reseact_3d_chem.esm")
const T0 = 64800.0
const LU = LinearSolve.LUFactorization()

docs = prepare_split_docs(MODEL)
fo = reseact_forcing(DIR)
say("building both parts …")
run = build_split_run(docs, (T0, T0+3600.0);
    providers=fo.providers, parameters=fo.parameters, const_arrays=fo.const_arrays)
f_trans_sm!, f_chem_sm! = run.funcs[1], run.funcs[2]
P = cellmajor_perm(run.var_map)
f_trans_cm! = cellmajor_rhs(f_trans_sm!, P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(f_chem_sm!,  P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)
u0 = run.u0[P.sm_of_cm]
foreach(d->d.materialize!(), run.dms)   # seed forcing at T0

# ---- 1+2. transport tendency / timescale / m per cell ----
dT = similar(u0); f_trans_cm!(dT, u0, run.p, T0)
dC = similar(u0); f_chem_cm!(dC, u0, run.p, T0)
say("\n=== per-BASE transport tendency + timescale (|u|/|du|) at t0 ===")
for (b,nm) in enumerate(P.bases)
    rng = b:P.NS:(P.N)              # cell-major: base b at positions b, b+NS, ...
    du = dT[rng]; u = u0[rng]
    # stiffest timescale = min over cells of |u|/|du| (guard tiny du)
    ts = [abs(du[c])>0 ? abs(u[c])/abs(du[c]) : Inf for c in 1:P.NC]
    @printf("  %-16s |du|_max=%.3e  min timescale=%.3e s  (u∈[%.2e,%.2e])\n",
            split(nm,'.')[2], maximum(abs,du), minimum(ts), minimum(u), maximum(u))
end
mb = P.base_pos["Transport3D.m"]
mvals = u0[mb:P.NS:P.N]
say(@sprintf("air-mass m: min=%.4e  max=%.4e  (min/max ratio=%.3e)", minimum(mvals), maximum(mvals), minimum(mvals)/maximum(mvals)))
say(@sprintf("transport |f|_inf=%.3e   chem |f|_inf=%.3e", maximum(abs,dT), maximum(abs,dC)))

# per-state global stiffest transport timescale
tsall = [abs(dT[i])>1e-30 ? abs(u0[i])/abs(dT[i]) : Inf for i in 1:P.N]
imin = argmin(tsall)
bmin = ((imin-1) % P.NS) + 1; cmin = ((imin-1) ÷ P.NS) + 1
say(@sprintf("STIFFEST transport state: %s cell#%d  timescale=%.3e s (u=%.3e, du=%.3e)",
             P.bases[bmin], cmin, tsall[imin], u0[imin], dT[imin]))

# ---- 3. transport-ONLY explicit solve (SSPRK43) ----
say("\n=== transport-ONLY explicit (SSPRK43) stability ===")
for tend in (60.0, 300.0)
    p = SciMLBase.ODEProblem(f_trans_cm!, copy(u0), (T0, T0+tend), run.p)
    t=time()
    s = SciMLBase.solve(p, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=1e-4, abstol=1e-6,
                        callback=run.cb, tstops=run.tstops, save_everystep=false, maxiters=200000)
    uf = s.u[end]
    mfin = uf[mb:P.NS:P.N]
    @printf("  tend=%.0fs: retcode=%s wall=%.1fs  nsteps=%d  finite=%s  m>0 all=%s  m∈[%.3e,%.3e]\n",
            tend, s.retcode, time()-t, s.stats.naccept + s.stats.nreject, all(isfinite,uf),
            all(>(0), mfin), minimum(mfin), maximum(mfin))
end

# ---- 4. chem-ONLY implicit solve (SplitODEProblem+Rosenbrock23, BlockDiagonal) ----
say("\n=== chem-ONLY implicit (Rosenbrock23 + BlockDiagonal) stability ===")
f1 = SciMLBase.ODEFunction(f_chem_cm!; jac=jac_cm!, jac_prototype=mkjp(), tgrad=(g,u,p,t)->fill!(g,0))
pc = SciMLBase.SplitODEProblem(f1, (du,u,p,t)->fill!(du,0), copy(u0), (T0,T0+300.0), run.p)
t=time()
sc = SciMLBase.solve(pc, OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
                     reltol=1e-4, abstol=1e-12, callback=run.cb, tstops=run.tstops, save_everystep=false)
say(@sprintf("  chem 300s: retcode=%s wall=%.1fs finite=%s", sc.retcode, time()-t, all(isfinite,sc.u[end])))
say("DONE")
