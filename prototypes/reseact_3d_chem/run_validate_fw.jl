#!/usr/bin/env julia
# Same as run_validate.jl BUT the RHS closures are wrapped in FunctionWrappers so
# each solver specializes on a TRIVIAL concrete type (FunctionWrapper{Nothing,...})
# instead of inlining the giant compile-once tier closure. The full-specialization
# path was thrashing (>30 min, 22% CPU, 4-5 GB swap) PER solver compile; the wrapper
# should cut both compile time AND memory to ~minutes. Runtime cost is one indirect
# call per RHS eval — negligible for a few hundred steps. ADVECTION STAYS EXPLICIT.
import Pkg; Pkg.activate("/Users/ctessum/code/earthsciml/reseact.esm/run-model-jl"; io=devnull)
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
import FunctionWrappers: FunctionWrapper
using LinearAlgebra, Printf
const DIR = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/reseact_3d_chem"
include(joinpath(DIR, "split_common.jl"))
include(joinpath(DIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(DIR, "block_jac.jl"))
say(s) = (println(s); flush(stdout))

const MODEL = joinpath(DIR, "reseact_3d_chem.esm")
const T0    = 64800.0
const T_END = T0 + (length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 120.0)
const RTOL  = 1e-4
const ATOL  = 1e-9
const LU    = LinearSolve.LUFactorization()
rc_ok(rc) = (rc == SciMLBase.ReturnCode.Success || rc == SciMLBase.ReturnCode.Default)

let ext = Base.get_extension(EA, :EarthSciASTDataRefreshExt)
    ext === nothing && error("EarthSciASTDataRefreshExt not loaded")
    say("DataRefreshExt loaded ✓")
end

say("validate …"); r = EA.validate(MODEL)
r.is_valid || error("invalid model")
say("  is_valid=$(r.is_valid)")

say("split + build …")
docs = prepare_split_docs(MODEL)
fo   = reseact_forcing(DIR)
tb = time()
run = build_split_run(docs, (T0, T_END);
    providers=fo.providers, parameters=fo.parameters, const_arrays=fo.const_arrays)
say(@sprintf("build: %.1f s   nstates=%d   ntstops=%d", time()-tb, length(run.u0), length(run.tstops)))

P = cellmajor_perm(run.var_map)
say(@sprintf("layout: NS=%d bases  NC=%d cells  N=%d", P.NS, P.NC, P.N))
f_trans_raw = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
f_chem_raw  = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
u0_cm = run.u0[P.sm_of_cm]
zerof!(du,u,p,t) = (fill!(du,0); nothing)
tgz(g,u,p,t) = (fill!(g,0); nothing)
mb = P.base_pos["Transport3D.m"]; mrng() = mb:P.NS:P.N
foreach(d->d.materialize!(), run.dms)

# --- STAGE-A INITIAL CONDITION: seed m(0) = dp = dA[k] + dB[k]*PS ------------------
# gen_t3d.py:63-69 KEEPS the Stage-B analytic m IC (~2.3 uniform, scale-invariant for
# the CWC gate) and states "Stage A supplies the physical air mass m(0) = dp via
# const_arrays/providers." That override was never wired: m started ~2.3 while the
# fluxes use the real dp (1520 Pa surface -> 1 Pa top) => continuity drives m negative
# and the explicit solve blows up. Seed it here. PS_REF is used for the profile; the
# ~10-30% horizontal PS variation is a gentle continuity transient (the 1000x SCALE
# error is the blowup). k is the vertical index (c[3]); dA/dB are 1..NLEV.
const PS_REF = 101325.0
let dA = Float64.(fo.const_arrays["Transport3D.dA"]), dB = Float64.(fo.const_arrays["Transport3D.dB"])
    for c in P.cells
        u0_cm[(P.cell_pos[c]-1)*P.NS + mb] = dA[c[3]] + dB[c[3]]*PS_REF
    end
    say(@sprintf("seeded m(0)=dp: m∈[%.3e, %.3e] over %d cells (was ~2.3 uniform)",
                 minimum(u0_cm[mrng()]), maximum(u0_cm[mrng()]), P.NC))
end

# Wrap for the SOLVER: trivial concrete type => tiny solver compile.
const WT = FunctionWrapper{Nothing, Tuple{Vector{Float64},Vector{Float64},typeof(run.p),Float64}}
f_trans_w = WT(f_trans_raw)
f_chem_w  = WT(f_chem_raw)
jac_cm!, mkjp = block_fd_jac(f_chem_w, P.NS, P.NC)   # jac calls the WRAPPER too

# --- (1) diagnostic (raw closures; called once) ------------------------------------
say("\n=== (1) diagnostic ===")
dT = similar(u0_cm); f_trans_raw(dT, u0_cm, run.p, T0)
dC = similar(u0_cm); f_chem_raw(dC, u0_cm, run.p, T0)
say(@sprintf("  |f_transport|_inf=%.3e   |f_chem|_inf=%.3e", maximum(abs,dT), maximum(abs,dC)))
tsall = [abs(dT[i])>1e-30 ? abs(u0_cm[i])/abs(dT[i]) : Inf for i in 1:P.N]
imin = argmin(tsall); bmin = ((imin-1) % P.NS) + 1
say(@sprintf("  STIFFEST transport state: base=%s  timescale=%.3e s", P.bases[bmin], tsall[imin]))
mvals = u0_cm[mrng()]
say(@sprintf("  m over all cells: min=%.4e max=%.4e  (#cells m<1: %d)", minimum(mvals), maximum(mvals), count(<(1.0),mvals)))

# --- (2) transport-only dtmax sweep (WRAPPED, NO PositiveDomain), window=120 s ------
say("\n=== (2) transport-only dtmax sweep (SSPRK43 wrapped), window=120 s ===")
best_dtmax = 0.2
for dtm in (Inf, 5.0, 1.0, 0.2)
    pr = SciMLBase.ODEProblem(f_trans_w, copy(u0_cm), (T0, T0+120.0), run.p)
    local s
    try
        s = SciMLBase.solve(pr, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
            dtmax=dtm, callback=run.cb, tstops=run.tstops, save_everystep=false, maxiters=2_000_000)
    catch e; @printf("  dtmax=%-5s ERROR: %s\n", dtm, sprint(showerror,e)[1:min(120,end)]); continue; end
    mf = s.u[end][mrng()]; ap = all(>(0), mf)
    @printf("  dtmax=%-5s retcode=%s nsteps=%d  m_min=%.4e all_pos=%s finite=%s\n",
        dtm, s.retcode, s.stats.naccept, minimum(mf), ap, all(isfinite,s.u[end]))
    ap && isfinite(minimum(mf)) && (global best_dtmax = min(best_dtmax, dtm))
end
say(@sprintf("  -> split schemes use dtmax=%.3g s", best_dtmax))

report(tag, uf_cm, wall, rc) = begin
    say(@sprintf("\n[%s] retcode=%s wall=%.1fs", tag, rc, wall))
    for sp in ["SuperFast.O3","SuperFast.NO","SuperFast.NO2","SuperFast.OH","SuperFast.HO2","SuperFast.CO","SuperFast.CH2O","Transport3D.m"]
        b = P.base_pos[sp]; rng = b:P.NS:P.N; t0r = (b-1)*P.NC+1:b*P.NC
        @printf("  %-10s t0∈[%.3e,%.3e]  t1∈[%.3e,%.3e]\n", split(sp,'.')[2],
                minimum(run.u0[t0r]),maximum(run.u0[t0r]), minimum(uf_cm[rng]),maximum(uf_cm[rng]))
    end
    say(@sprintf("  m all positive: %s", all(>(0), uf_cm[mrng()])))
    uf_cm
end

# --- (3a) Option B: IMEX KenCarp47 --------------------------------------------------
say("\n=== (3a) Option B: IMEX KenCarp47 (dtmax=$(best_dtmax), PositiveDomain) ===")
uB = nothing; t = time()
try
    f1 = SciMLBase.ODEFunction(f_chem_w; jac=jac_cm!, jac_prototype=mkjp())
    prob = SciMLBase.SplitODEProblem(f1, f_trans_w, copy(u0_cm), (T0,T_END), run.p)
    cbset = SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u0_cm)))
    sol = SciMLBase.solve(prob, OrdinaryDiffEqSDIRK.KenCarp47(autodiff=false, linsolve=LU);
        reltol=RTOL, abstol=ATOL, dtmax=best_dtmax, callback=cbset, tstops=run.tstops,
        saveat=[T0,T_END], maxiters=2_000_000)
    global uB = rc_ok(sol.retcode) ? report("IMEX KenCarp47", sol.u[end], time()-t, sol.retcode) :
                (say("  Option B did NOT converge: $(sol.retcode) ($(round(time()-t,digits=1))s)"); nothing)
catch e; say("  Option B ERROR: $(sprint(showerror,e)[1:min(400,end)])"); end

# --- (3b) Option A: op-split SSPRK43 + Rosenbrock23/BD -------------------------------
say("\n=== (3b) Option A: op-split SSPRK43 + Rosenbrock23/BD (dtmax=$(best_dtmax), PositiveDomain) ===")
uA = nothing; t = time()
try
    macro_dt = min(60.0, T_END-T0); u = copy(u0_cm); tc = T0; nstep = 0; ok = true
    while tc < T_END - 1e-9 && ok
        h = min(macro_dt, T_END-tc)
        pT = SciMLBase.ODEProblem(f_trans_w, u, (tc,tc+h), run.p)
        sT = SciMLBase.solve(pT, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
            dtmax=best_dtmax, callback=SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u))),
            tstops=run.tstops, save_everystep=false, maxiters=2_000_000)
        rc_ok(sT.retcode) || (say("  transport sub-step $(nstep+1): $(sT.retcode)"); ok=false; break)
        u = sT.u[end]
        f1c = SciMLBase.ODEFunction(f_chem_w; jac=jac_cm!, jac_prototype=mkjp(), tgrad=tgz)
        pC = SciMLBase.SplitODEProblem(f1c, zerof!, u, (tc,tc+h), run.p)
        sC = SciMLBase.solve(pC, OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
            reltol=RTOL, abstol=ATOL, callback=DiffEqCallbacks.PositiveDomain(copy(u)),
            save_everystep=false, maxiters=2_000_000)
        rc_ok(sC.retcode) || (say("  chem sub-step $(nstep+1): $(sC.retcode)"); ok=false; break)
        u = sC.u[end]; tc += h; nstep += 1
    end
    global uA = ok ? (say("  Lie-Trotter steps: $nstep (macro_dt=$macro_dt s)"); report("split SSPRK43+Rosenbrock23", u, time()-t, "ok")) : nothing
catch e; say("  Option A ERROR: $(sprint(showerror,e)[1:min(400,end)])"); end

if uA !== nothing && uB !== nothing
    d = maximum(abs, uA .- uB); rel = d/max(maximum(abs,uB),eps())
    say(@sprintf("\nOption A vs B  max|Δ|=%.3e  (rel %.3e)", d, rel))
end
say("DONE")
