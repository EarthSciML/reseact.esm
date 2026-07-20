#!/usr/bin/env julia
# Stage C, SPLIT solve on the FULL 72-level GEOS-FP column (7x7x72 = 3528 cells).
#
# Operator splitting: f_full = f_transport + f_chemistry (exactly). ADVECTION IS
# EXPLICIT (SSPRK43) — always. Chemistry is stiff but BLOCK-DIAGONAL (each cell's
# species couple only among themselves) => implicit Rosenbrock23/KenCarp47 with
# EarthSciMLBase's BlockDiagonal Jacobian (3528 blocks of 13x13), filled by a
# 13-color finite difference.
#
# WHY 72 LEVELS: the 7-level slab was CLOSED at a false ~906 hPa top, where real
# OMEGA pumps mass through with nowhere to go => air-mass m piled up and went
# negative within ~7 s, blowing up the 1/m CWC advection. The full column's
# no-flux top is the TRUE TOA (OMEGA->0 there), so m no longer accumulates and
# explicit advection is stable. PositiveDomain is added as a safety net for
# stiff-chemistry tiny-negatives (NOT as a fix for the m blowup — that's the grid).
import Pkg; Pkg.activate(get(ENV, "RESEACT_RUN_ENV", normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io=devnull)
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqSDIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf
const DIR = @__DIR__
include(joinpath(DIR, "split_common.jl"))
include(joinpath(DIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(DIR, "block_jac.jl"))
say(s) = (println(s); flush(stdout))

const MODEL = joinpath(DIR, "reseact_3d_chem.esm")
const T0    = 64800.0
const T_END = T0 + (length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 600.0)
const RTOL  = 1e-4
const ATOL  = 1e-9
const LU    = LinearSolve.LUFactorization()
rc_ok(rc) = (rc == SciMLBase.ReturnCode.Success || rc == SciMLBase.ReturnCode.Default)

say("validate …"); r = EA.validate(MODEL)
say("  is_valid=$(r.is_valid)  struct_errs=$(length(r.structural_errors))")
r.is_valid || (for e in r.structural_errors[1:min(6,end)]; say("   $(e.error_type)@$(e.path): $(e.message)"); end; error("invalid"))

# Fail FAST (seconds, not after a ~35-min build): the forcing refresh callback is a
# package extension (EarthSciASTDataRefreshExt, triggered by DiffEqCallbacks+SciMLBase).
# If it isn't loaded, build_refresh_callback hits the throwing varargs fallback — but
# only AFTER the expensive build. Assert it's live up front.
let ext = Base.get_extension(EA, :EarthSciASTDataRefreshExt)
    ext === nothing && error("EarthSciASTDataRefreshExt not loaded; build_refresh_callback " *
        "would throw after the build. Ensure `using SciMLBase, DiffEqCallbacks` ran.")
    say("DataRefreshExt loaded ✓ (forcing refresh callback available)")
end

# --- build both parts over shared GEOS-FP forcing (transport tier build) ----------
say("split + build …")
docs = prepare_split_docs(MODEL)
fo = reseact_forcing(DIR)
t = time()
run = build_split_run(docs, (T0, T_END);
    providers=fo.providers, parameters=fo.parameters, const_arrays=fo.const_arrays)
say(@sprintf("build: %.1f s   nstates=%d   ntstops=%d", time()-t, length(run.u0), length(run.tstops)))
f_trans_sm!, f_chem_sm! = run.funcs[1], run.funcs[2]

P = cellmajor_perm(run.var_map)
say(@sprintf("layout: NS=%d bases  NC=%d cells  N=%d", P.NS, P.NC, P.N))
f_trans_cm! = cellmajor_rhs(f_trans_sm!, P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(f_chem_sm!,  P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)
u0_cm = run.u0[P.sm_of_cm]
zerof!(du,u,p,t) = (fill!(du,0); nothing)
tgz(g,u,p,t) = (fill!(g,0); nothing)
mb = P.base_pos["Transport3D.m"]; mrng() = mb:P.NS:P.N   # cell-major: base mb at stride NS
foreach(d->d.materialize!(), run.dms)

# --- STAGE-A INITIAL CONDITION: seed m(0) = dp = dA[k] + dB[k]*PS ------------------
# The .esm keeps the Stage-B analytic test mass m (~0..100, scale-invariant for the
# CWC gate). The fluxes, however, use the REAL dp (~1..5000 Pa over 72 levels), so a
# ~2 Pa mass is driven negative by continuity within seconds and the explicit advection
# blows up. gen_t3d.py says Stage A must supply m(0)=dp; that override was never wired
# into run_split.jl (run_validate_fw.jl already does it). Seed it here — same mechanism.
# PS_REF gives the profile; the ~10-30% horizontal PS variation is a gentle continuity
# transient (the 1000x SCALE error is the blowup). k is the vertical index (c[3]).
const PS_REF = 101325.0
let dA = Float64.(fo.const_arrays["Transport3D.dA"]), dB = Float64.(fo.const_arrays["Transport3D.dB"])
    for c in P.cells
        u0_cm[(P.cell_pos[c]-1)*P.NS + mb] = dA[c[3]] + dB[c[3]]*PS_REF
    end
    say(@sprintf("seeded m(0)=dp: m∈[%.3e, %.3e] over %d cells (was ~analytic test mass)",
                 minimum(u0_cm[mrng()]), maximum(u0_cm[mrng()]), P.NC))
end

# --- transport-only explicit PRE-CHECK: did 72 levels fix the m blowup? -----------
say("\n=== transport-only explicit (SSPRK43) — is m bounded now? ===")
dT = similar(u0_cm); f_trans_cm!(dT, u0_cm, run.p, T0)
say(@sprintf("  transport |f|_inf@t0 = %.3e   m@t0 ∈ [%.4f, %.4f]", maximum(abs,dT), minimum(u0_cm[mrng()]), maximum(u0_cm[mrng()])))
for tend in (60.0, 300.0)
    pr = SciMLBase.ODEProblem(f_trans_cm!, copy(u0_cm), (T0, T0+tend), run.p)
    s = SciMLBase.solve(pr, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
        callback=run.cb, tstops=run.tstops, save_everystep=false, maxiters=500000)
    mf = s.u[end][mrng()]
    @printf("  tend=%.0fs: retcode=%s nsteps=%d  m∈[%.4f,%.4f] all_pos=%s finite=%s\n",
        tend, s.retcode, s.stats.naccept, minimum(mf), maximum(mf), all(>(0),mf), all(isfinite,s.u[end]))
end

report(tag, uf_cm, wall, rc) = begin
    say(@sprintf("\n[%s] retcode=%s wall=%.1fs", tag, rc, wall))
    uf = uf_cm[P.cm_of_sm]
    for sp in ["SuperFast.O3","SuperFast.NO","SuperFast.NO2","SuperFast.OH","SuperFast.HO2","SuperFast.CO","SuperFast.CH2O","Transport3D.m"]
        b = P.base_pos[sp]; rng = (b-1)*P.NC+1 : b*P.NC
        @printf("  %-16s t0∈[%.4e,%.4e]  t1∈[%.4e,%.4e]\n", split(sp,'.')[2],
                minimum(run.u0[rng]),maximum(run.u0[rng]),minimum(uf[rng]),maximum(uf[rng]))
    end
    say(@sprintf("  m all positive: %s", all(>(0), uf[(mb-1)*P.NC+1 : mb*P.NC])))  # uf is species-major
    uf
end

# --- Option B: IMEX  SplitODEProblem(chem implicit, transport explicit) + KenCarp47 ---
say("\n=== Option B: IMEX (KenCarp47) ===")
uB = nothing; t = time()
try
    f1 = SciMLBase.ODEFunction(f_chem_cm!; jac=jac_cm!, jac_prototype=mkjp())
    prob = SciMLBase.SplitODEProblem(f1, f_trans_cm!, copy(u0_cm), (T0,T_END), run.p)
    cbset = SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u0_cm)))
    sol = SciMLBase.solve(prob, OrdinaryDiffEqSDIRK.KenCarp47(autodiff=false, linsolve=LU);
        reltol=RTOL, abstol=ATOL, callback=cbset, tstops=run.tstops, saveat=[T0,T_END], maxiters=500000)
    global uB = rc_ok(sol.retcode) ? report("IMEX KenCarp47", sol.u[end], time()-t, sol.retcode) :
                (say("  Option B did NOT converge: $(sol.retcode) (wall $(round(time()-t,digits=1))s)"); nothing)
catch e; say("  Option B ERROR: $(sprint(showerror,e)[1:min(400,end)])"); end

# --- Option A: operator splitting (Lie-Trotter) SSPRK43 + Rosenbrock23/BlockDiagonal ---
say("\n=== Option A: operator splitting (SSPRK43 transport + Rosenbrock23/BD chem) ===")
uA = nothing; t = time()
try
    macro_dt = min(300.0, T_END-T0); u = copy(u0_cm); tc = T0; nstep = 0; ok = true
    while tc < T_END - 1e-9 && ok
        h = min(macro_dt, T_END-tc)
        pT = SciMLBase.ODEProblem(f_trans_cm!, u, (tc,tc+h), run.p)
        sT = SciMLBase.solve(pT, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
            callback=SciMLBase.CallbackSet(run.cb, DiffEqCallbacks.PositiveDomain(copy(u))),
            tstops=run.tstops, save_everystep=false, maxiters=500000)
        rc_ok(sT.retcode) || (say("  transport sub-step $(nstep+1) failed: $(sT.retcode)"); ok=false; break)
        u = sT.u[end]
        f1c = SciMLBase.ODEFunction(f_chem_cm!; jac=jac_cm!, jac_prototype=mkjp(), tgrad=tgz)
        pC = SciMLBase.SplitODEProblem(f1c, zerof!, u, (tc,tc+h), run.p)
        sC = SciMLBase.solve(pC, OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
            reltol=RTOL, abstol=ATOL, callback=DiffEqCallbacks.PositiveDomain(copy(u)),
            save_everystep=false, maxiters=500000)
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
