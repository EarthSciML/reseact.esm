#!/usr/bin/env julia
# Focused validation of the 72-level SPLIT solver, with the build memory-wall now
# cleared (array-IR EarthSciAST fits 7x7x72 = 3528 cells / 45864 states on 8 GB).
#
# CORRECTED MODEL of the cost/behavior (learned from the first full run):
#   * The apparent "slowness" was NOT CFL — the transport-only 60 s solve took only
#     nsteps=3 (dt~20 s). The wall time is ONE-TIME JIT compilation of each solver
#     stack over the 45864-state system. => run under `julia -O1` (runtime is a few
#     hundred cheap steps, so low opt level trades negligible runtime for far less
#     compile).
#   * 72 levels did NOT fully fix the air-mass blowup: with 3 BIG explicit steps and
#     NO positivity enforcement, m dipped to -2.89 in the thin TOA cells (vs the 7 s
#     CATASTROPHIC blowup of the 7-level false top — so 72 levels turned it from
#     catastrophic into a mild large-dt OVERSHOOT). The root cause is an explicit
#     step over-depleting a thin cell's tiny mass, exactly the "underlying problem".
#
# This script answers three things in ONE build:
#   (1) DIAGNOSE: stiffest transport timescale (= max stable explicit dt) + how m
#       thins with height (which cells are overshoot-prone).
#   (2) SWEEP dtmax on transport-only (NO clipping) to find the step size that keeps
#       m >= 0 by RESPECTING the thin-cell CFL — the honest fix, not a band-aid.
#   (3) VALIDATE both split schemes (Option B IMEX KenCarp47; Option A op-split
#       SSPRK43 + Rosenbrock23/BlockDiagonal) with that dtmax + PositiveDomain.
# ADVECTION IS ALWAYS EXPLICIT.
import Pkg; Pkg.activate("/Users/ctessum/code/earthsciml/reseact.esm/run-model-jl"; io=devnull)
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf
const DIR = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/reseact_3d_chem"
include(joinpath(DIR, "split_common.jl"))
include(joinpath(DIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(DIR, "block_jac.jl"))
say(s) = (println(s); flush(stdout))

const MODEL = joinpath(DIR, "reseact_3d_chem.esm")
const T0    = 64800.0
const T_END = T0 + (length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 300.0)
const RTOL  = 1e-4
const ATOL  = 1e-9
const LU    = LinearSolve.LUFactorization()
rc_ok(rc) = (rc == SciMLBase.ReturnCode.Success || rc == SciMLBase.ReturnCode.Default)

let ext = Base.get_extension(EA, :EarthSciASTDataRefreshExt)
    ext === nothing && error("EarthSciASTDataRefreshExt not loaded")
    say("DataRefreshExt loaded ✓")
end

say("validate …"); r = EA.validate(MODEL)
say("  is_valid=$(r.is_valid)  struct_errs=$(length(r.structural_errors))")
r.is_valid || error("invalid model")

# --- build both parts (transport tier build) --------------------------------------
say("split + build …")
docs = prepare_split_docs(MODEL)
fo   = reseact_forcing(DIR)
tb = time()
run = build_split_run(docs, (T0, T_END);
    providers=fo.providers, parameters=fo.parameters, const_arrays=fo.const_arrays)
say(@sprintf("build: %.1f s   nstates=%d   ntstops=%d", time()-tb, length(run.u0), length(run.tstops)))

P = cellmajor_perm(run.var_map)
say(@sprintf("layout: NS=%d bases  NC=%d cells  N=%d", P.NS, P.NC, P.N))
f_trans_cm! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)
u0_cm = run.u0[P.sm_of_cm]
zerof!(du,u,p,t) = (fill!(du,0); nothing)
tgz(g,u,p,t) = (fill!(g,0); nothing)
mb = P.base_pos["Transport3D.m"]; mrng() = mb:P.NS:P.N
foreach(d->d.materialize!(), run.dms)

# --- (1) DIAGNOSE: stiffest transport timescale + m thinning with height ----------
say("\n=== (1) diagnostic ===")
dT = similar(u0_cm); f_trans_cm!(dT, u0_cm, run.p, T0)
dC = similar(u0_cm); f_chem_cm!(dC, u0_cm, run.p, T0)
say(@sprintf("  |f_transport|_inf=%.3e   |f_chem|_inf=%.3e", maximum(abs,dT), maximum(abs,dC)))
# stiffest transport state (min |u|/|du|) => the explicit dt that state can tolerate
tsall = [abs(dT[i])>1e-30 ? abs(u0_cm[i])/abs(dT[i]) : Inf for i in 1:P.N]
imin = argmin(tsall)
bmin = ((imin-1) % P.NS) + 1; cmin_pos = ((imin-1) ÷ P.NS) + 1
say(@sprintf("  STIFFEST transport state: base=%s  timescale=%.3e s  (u=%.3e du=%.3e)",
             P.bases[bmin], tsall[imin], u0_cm[imin], dT[imin]))
# m thinning with height: min m per vertical level k
mvals = u0_cm[mrng()]
say(@sprintf("  m over all cells: min=%.4e max=%.4e  (#cells m<1: %d, m<0.1: %d)",
             minimum(mvals), maximum(mvals), count(<(1.0),mvals), count(<(0.1),mvals)))
levs = sort(unique(c[3] for c in P.cells))
for k in (levs[1], levs[2], levs[end-1], levs[end])
    ck = [c for c in P.cells if c[3]==k]
    mk = [u0_cm[(P.cell_pos[c]-1)*P.NS + mb] for c in ck]
    @printf("    level k=%d: m∈[%.4e, %.4e]\n", k, minimum(mk), maximum(mk))
end

# --- (2) SWEEP dtmax on transport-only (NO clipping): find CFL that keeps m>=0 ----
say("\n=== (2) transport-only dtmax sweep (SSPRK43, NO PositiveDomain), window=120 s ===")
best_dtmax = 0.2
for dtm in (Inf, 5.0, 1.0, 0.2)
    pr = SciMLBase.ODEProblem(f_trans_cm!, copy(u0_cm), (T0, T0+120.0), run.p)
    local s
    try
        s = SciMLBase.solve(pr, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
            dtmax=dtm, callback=run.cb, tstops=run.tstops, save_everystep=false, maxiters=2_000_000)
    catch e; @printf("  dtmax=%-5s ERROR: %s\n", dtm, sprint(showerror,e)[1:min(120,end)]); continue; end
    mf = s.u[end][mrng()]
    ap = all(>(0), mf)
    @printf("  dtmax=%-5s retcode=%s nsteps=%d  m_min=%.4e all_pos=%s finite=%s\n",
        dtm, s.retcode, s.stats.naccept, minimum(mf), ap, all(isfinite,s.u[end]))
    ap && isfinite(minimum(mf)) && (global best_dtmax = min(best_dtmax, dtm))
end
say(@sprintf("  -> using dtmax=%.3g s for the split schemes (smallest swept dtmax that held m>=0, capped)", best_dtmax))

report(tag, uf_cm, wall, rc) = begin
    say(@sprintf("\n[%s] retcode=%s wall=%.1fs", tag, rc, wall))
    uf = uf_cm[P.cm_of_sm]
    for sp in ["SuperFast.O3","SuperFast.NO","SuperFast.NO2","SuperFast.OH","SuperFast.HO2","SuperFast.CO","SuperFast.CH2O","Transport3D.m"]
        b = P.base_pos[sp]; rng = b:P.NS:P.N
        @printf("  %-10s t0∈[%.3e,%.3e]  t1∈[%.3e,%.3e]\n", split(sp,'.')[2],
                minimum(run.u0[(b-1)*P.NC+1:b*P.NC]),maximum(run.u0[(b-1)*P.NC+1:b*P.NC]),
                minimum(uf_cm[rng]),maximum(uf_cm[rng]))
    end
    say(@sprintf("  m all positive: %s", all(>(0), uf_cm[mrng()])))
    uf_cm
end

# --- (3a) Option B: IMEX SplitODEProblem(chem implicit, transport explicit) + KenCarp47 ---
say("\n=== (3a) Option B: IMEX KenCarp47 (dtmax=$(best_dtmax), PositiveDomain) ===")
uB = nothing; t = time()
try
    f1 = SciMLBase.ODEFunction(f_chem_cm!; jac=jac_cm!, jac_prototype=mkjp())
    prob = SciMLBase.SplitODEProblem(f1, f_trans_cm!, copy(u0_cm), (T0,T_END), run.p)
    cbset = SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u0_cm)))
    sol = SciMLBase.solve(prob, OrdinaryDiffEqSDIRK.KenCarp47(autodiff=false, linsolve=LU);
        reltol=RTOL, abstol=ATOL, dtmax=best_dtmax, callback=cbset, tstops=run.tstops,
        saveat=[T0,T_END], maxiters=2_000_000)
    global uB = rc_ok(sol.retcode) ? report("IMEX KenCarp47", sol.u[end], time()-t, sol.retcode) :
                (say("  Option B did NOT converge: $(sol.retcode) (wall $(round(time()-t,digits=1))s)"); nothing)
catch e; say("  Option B ERROR: $(sprint(showerror,e)[1:min(400,end)])"); end

# --- (3b) Option A: operator splitting (Lie-Trotter) SSPRK43 + Rosenbrock23/BlockDiagonal ---
say("\n=== (3b) Option A: op-split SSPRK43 + Rosenbrock23/BD (dtmax=$(best_dtmax), PositiveDomain) ===")
uA = nothing; t = time()
try
    macro_dt = min(60.0, T_END-T0); u = copy(u0_cm); tc = T0; nstep = 0; ok = true
    while tc < T_END - 1e-9 && ok
        h = min(macro_dt, T_END-tc)
        pT = SciMLBase.ODEProblem(f_trans_cm!, u, (tc,tc+h), run.p)
        sT = SciMLBase.solve(pT, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
            dtmax=best_dtmax, callback=SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u))),
            tstops=run.tstops, save_everystep=false, maxiters=2_000_000)
        rc_ok(sT.retcode) || (say("  transport sub-step $(nstep+1) failed: $(sT.retcode)"); ok=false; break)
        u = sT.u[end]
        f1c = SciMLBase.ODEFunction(f_chem_cm!; jac=jac_cm!, jac_prototype=mkjp(), tgrad=tgz)
        pC = SciMLBase.SplitODEProblem(f1c, zerof!, u, (tc,tc+h), run.p)
        sC = SciMLBase.solve(pC, OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
            reltol=RTOL, abstol=ATOL, callback=DiffEqCallbacks.PositiveDomain(copy(u)),
            save_everystep=false, maxiters=2_000_000)
        rc_ok(sC.retcode) || (say("  chem sub-step $(nstep+1) failed: $(sC.retcode)"); ok=false; break)
        u = sC.u[end]; tc += h; nstep += 1
    end
    global uA = ok ? (say("  Lie-Trotter steps: $nstep (macro_dt=$macro_dt s)"); report("split SSPRK43+Rosenbrock23", u, time()-t, "ok")) : nothing
catch e; say("  Option A ERROR: $(sprint(showerror,e)[1:min(400,end)])"); end

if uA !== nothing && uB !== nothing
    d = maximum(abs, uA .- uB); rel = d/max(maximum(abs,uB),eps())
    say(@sprintf("\nOption A vs B  max|Δ|=%.3e  (rel %.3e)", d, rel))
end
say("DONE")
