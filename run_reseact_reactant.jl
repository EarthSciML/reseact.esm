#!/usr/bin/env julia
# ===========================================================================
# run_reseact_reactant.jl -- run the full ReSEACT model under Reactant/XLA, with
# the ENTIRE operator-split window compiled as ONE XLA program.
# ===========================================================================
# Same Lie-Trotter scheme as run_reseact.jl (SSPRK43 transport then
# Rosenbrock23 chemistry with a block-diagonal FD Jacobian), but both adaptive
# time loops run inside `Reactant.@trace while` -> stablehlo.while regions, so
# each half's RHS graph appears exactly ONCE in the compiled module no matter
# how many steps the solve takes. There is no host<->device round-trip per RHS
# eval -- one host call runs the whole window on device.
#
# Result at 7x7x72 (45,864 states): traced window 4.52 s vs 36.57 s for the
# native runner (8.1x), m/O3 identical to the native baseline. The @compile is a
# one-time ~54 min and is grid-independent (identical trace at every grid).
#
# This runner reuses the native in-place path (run_reseact.jl's scheme) as a HOST
# reference arm and reports the traced-vs-host difference rather than hiding it.
#
# Env:
#   RESEACT_MODEL      .esm to run             (default: repo-root reseact.esm)
#   RESEACT_LABEL      tag for the RESULT line (default: basename of the model)
#   RESEACT_SOLVE_SECS solve window, seconds   (default 60, one Lie-Trotter step)
#   RESEACT_DT0T       initial transport dt    (default 15.0)
#   RESEACT_DT0C       initial chemistry dt    (default 0.5)
#   RESEACT_HOST_ARM   also run the host reference arm (default 1)
#   RESEACT_RXENV      Julia env WITH Reactant (default: <repo>/run-model-jl)
#
# Helper code it pulls in (see HELPERS.md for the migration plan):
#   prototypes/reseact_3d_chem/{split_common,blockdiag_local,block_jac}.jl
#       split build, forcing, cell-major layout, block-diagonal FD Jacobian (host arm)
#   tools/reactant_handoff/rx_native_patch.jl
#       native stablehlo broadcast lowering + make_tracer opaque leaves (trace enablers)
#   tools/reactant_handoff/rx_merge_lib.jl
#       driver-side kernel-class merge -- now a bit-identity GATE over the
#       in-package merge that landed in EarthSciAST (build.jl); kept as a fallback
#   tools/reactant_handoff/rx_traced_integrator.jl
#       the purpose-built traced integrator (ROS23 + SSPRK43 under @trace while)
# ===========================================================================
import Pkg
const REPO = @__DIR__
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system, stencil_vs_pointwise
using Reactant
using Logging
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end   # no GPU on this box

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", basename(MODEL))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "60"))
const DT0T       = parse(Float64, get(ENV, "RESEACT_DT0T", "15.0"))
const DT0C       = parse(Float64, get(ENV, "RESEACT_DT0C", "0.5"))
const HOST_ARM   = get(ENV, "RESEACT_HOST_ARM", "1") == "1"
const T0    = 64800.0
const T_END = T0 + SOLVE_SECS
const RTOL  = 1e-4
const ATOL_T = 1e-6         # transport abstol
const ATOL_C = 1e-9         # chemistry abstol
const LU    = LinearSolve.LUFactorization()
const PS_REF = 101325.0
rc_ok(rc) = (rc == SciMLBase.ReturnCode.Success || rc == SciMLBase.ReturnCode.Default)

# --------------------------------------------------------------------------- #
# 1. Split + build both halves as out-of-place (form=:oop) RHS over shared live
#    GEOS-FP forcing buffers. :oop is required for tracing: an in-place f!
#    captures host scratch per node and cannot trace.
# --------------------------------------------------------------------------- #
say("=== $LABEL : split + build oop halves ($MODEL) ===")
fo = Vector{Any}(undef, 2); u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); discrete_providers = Dict{String,Any}()
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, u0, p, var_map, merged_param, discrete_providers
    file = EA.load(MODEL); flat = EA.flatten(file)
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
    parts = split_system(flat, stencil_vs_pointwise; nparts = 2)
    docs  = [EA.flattened_to_esm(pt) for pt in parts]
    ff = reseact_forcing(CHEMDIR)
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld; discrete_providers[k] = prov
        end
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    for i in 1:2
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const, param_arrays = merged_param)
        fo[i] = fi
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part 2 var_map != part 1")
        end
    end
end
say(@sprintf("BUILD %s: %.2f s   nstates=%d", LABEL, time() - tb, length(u0)))

# Seed air mass m(0) = dA + dB*ps_ref (species-major, via var_map names).
let ff = reseact_forcing(CHEMDIR)
    dA = Float64.(ff.const_arrays["Transport3D.dA"]); dB = Float64.(ff.const_arrays["Transport3D.dB"])
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dA[parse(Int, mm.captures[3])] + dB[parse(Int, mm.captures[3])] * PS_REF
    end
end

# --------------------------------------------------------------------------- #
# 2. Native broadcast lowering + kernel-class merge (bit-identity gated).
#    The in-package merge (EarthSciAST build.jl) already runs at build; the
#    driver-side merge is now a no-op gate that verifies bit-identity and falls
#    back to the stock RHS for any half that is not bit-identical.
# --------------------------------------------------------------------------- #
include(joinpath(RXDIR, "rx_native_patch.jl"))      # AFTER `using Reactant`, `using EarthSciAST`
include(joinpath(RXDIR, "rx_merge_lib.jl"))
include(joinpath(RXDIR, "rx_traced_integrator.jl"))
const PARTNAME = ("transport", "pointwise")
host_bufs = [EA.forcing_buffers(fo[i]) for i in 1:2]
g4 = Vector{Any}(undef, 2)
for i in 1:2
    g4[i] = EA.rhs_with_buffers(fo[i]); kdesc = "stock"
    try
        mg = RxOopMerge.merge_oop_rhs(getfield(fo[i], :rhs))
        dmax = maximum(abs, mg.rhs(u0, p, T0, host_bufs[i]) .- fo[i](u0, p, T0))
        if dmax == 0.0
            g4[i] = mg.rhs; kdesc = "merged $(mg.stats.n_kernels)->$(mg.stats.n_merged)"
        else
            say("  half[$i]=$(PARTNAME[i]) merge NOT bit-identical (maxabs=$dmax) -> stock")
        end
    catch e
        say("  half[$i]=$(PARTNAME[i]) merge FAILED ($(sprint(showerror, e))) -> stock")
    end
    say("  half[$i]=$(PARTNAME[i]) rhs: $kdesc")
end

P = cellmajor_perm(var_map)
const NS = P.NS; const NC = P.NC; const N = P.N
masks = RxTracedIntegrator.species_masks(var_map, NS, NC)
say("  species-major layout verified: NS=$NS NC=$NC")
if !isempty(discrete_providers)
    say("  note: $(length(discrete_providers)) discrete providers; single-window solve (no interior refresh for $(SOLVE_SECS)s)")
end

# --------------------------------------------------------------------------- #
# 3. HOST reference arm: the native in-place Lie-Trotter step (as run_reseact.jl)
#    on the merged interpreted RHS reading the live buffers.
# --------------------------------------------------------------------------- #
host_iip(g, bufs) = (du, u, p, t) -> (copyto!(du, g(u, p, t, bufs)); nothing)
tgz(g, u, p, t)   = (fill!(g, 0); nothing)
zerof!(du, u, p, t) = (fill!(du, 0); nothing)
to_sm(ucm) = (usm = similar(ucm); for i in 1:N; usm[P.sm_of_cm[i]] = ucm[i]; end; usm)

uH = nothing; hostT = (0, 0); hostC = (0, 0); host_s = NaN; host_ok = false
if HOST_ARM
    f_trans_cm! = cellmajor_rhs(host_iip(g4[1], host_bufs[1]), P.sm_of_cm)
    f_chem_cm!  = cellmajor_rhs(host_iip(g4[2], host_bufs[2]), P.sm_of_cm)
    jac_cm!, mkjp = block_fd_jac(f_chem_cm!, NS, NC)
    u0cm = u0[P.sm_of_cm]
    ts = time()
    sT = SciMLBase.solve(SciMLBase.ODEProblem(f_trans_cm!, copy(u0cm), (T0, T_END), p),
        OrdinaryDiffEqSSPRK.SSPRK43(); reltol = RTOL, abstol = ATOL_T, dt = DT0T,
        callback = DiffEqCallbacks.PositiveDomain(copy(u0cm)), save_everystep = false, maxiters = 500000)
    um = sT.u[end]
    fc = SciMLBase.ODEFunction(f_chem_cm!; jac = jac_cm!, jac_prototype = mkjp(), tgrad = tgz)
    sC = SciMLBase.solve(SciMLBase.SplitODEProblem(fc, zerof!, um, (T0, T_END), p),
        OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false, linsolve = LU);
        reltol = RTOL, abstol = ATOL_C, dt = DT0C,
        callback = DiffEqCallbacks.PositiveDomain(copy(um)), save_everystep = false, maxiters = 500000)
    global host_s = time() - ts
    global uH = to_sm(sC.u[end])
    global hostT = (Int(sT.stats.naccept), Int(sT.stats.nreject))
    global hostC = (Int(sC.stats.naccept), Int(sC.stats.nreject))
    global host_ok = rc_ok(sT.retcode) && rc_ok(sC.retcode)
    say(@sprintf("HOST   window: %.2f s  transport nacc=%d nrej=%d  chem nacc=%d nrej=%d  ok=%s",
        host_s, hostT[1], hostT[2], hostC[1], hostC[2], host_ok))
end

# --------------------------------------------------------------------------- #
# 4. TRACED arm: one @compile for the whole window (both adaptive loops).
#    Traced deps (params, forcing buffers) flow through the loop-carried `aux`,
#    NOT closure capture -- @trace while builds its operands only from the
#    syntactically-collected loop variables.
# --------------------------------------------------------------------------- #
const CTRL_T = RxTracedIntegrator.pictrl_ssprk43()
const CTRL_C = RxTracedIntegrator.pictrl_ros23()
adv = let gT = g4[1], gC = g4[2], NS = NS, NC = NC, masks = masks
    (uR, pR, t0R, t1R, dtTR, dtCR, bufsT, bufsC) -> begin
        fT = (u, t, dtc, aux) -> RxTracedIntegrator.ssprk43_step(
            (uu, tt) -> gT(uu, aux.p, tt, aux.bufs), u, t, dtc, ATOL_T, RTOL)
        uT, _, dtTend, naT, nrT = RxTracedIntegrator.adaptive_solve(
            fT, uR, t0R, t1R, dtTR, CTRL_T, (p = pR, bufs = bufsT); clamp_nonneg = true)
        fC = (u, t, dtc, aux) -> RxTracedIntegrator.ros23_step(
            (uu, tt) -> gC(uu, aux.p, tt, aux.bufs), u, t, dtc, NS, NC, masks, ATOL_C, RTOL)
        uC, _, dtCend, naC, nrC = RxTracedIntegrator.adaptive_solve(
            fC, uT, t0R, t1R, dtCR, CTRL_C, (p = pR, bufs = bufsC); clamp_nonneg = true)
        return uC, naT, nrT, naC, nrC, dtTend, dtCend
    end
end
_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
dev_bufs = [map(RX.ConcreteRArray, host_bufs[i]) for i in 1:2]
for i in 1:2
    EA.sync_forcing!(dev_bufs[i], EA.forcing_buffers(fo[i]))
end
UR = RX.ConcreteRArray(u0); PRd = _dev(p)
targs() = (UR, PRd, RX.ConcreteRNumber(T0), RX.ConcreteRNumber(T_END),
           RX.ConcreteRNumber(DT0T), RX.ConcreteRNumber(DT0C), dev_bufs[1], dev_bufs[2])
say("  @compile of the full window starting (expect ~tens of minutes: 4x transport RHS + 16x chem RHS in the two while bodies)...")
tc = time()
xadv = RX.@compile sync=true adv(targs()...)
comp_s = time() - tc
say(@sprintf("TRACED @compile: %.1f s", comp_s))
report_patch_stats()
tr = time()
res = xadv(targs()...)
run_s = time() - tr
uT = Array(res[1])
naT = Int(round(Float64(res[2]))); nrT = Int(round(Float64(res[3])))
naC = Int(round(Float64(res[4]))); nrC = Int(round(Float64(res[5])))
say(@sprintf("TRACED window: %.2f s  transport nacc=%d nrej=%d  chem nacc=%d nrej=%d",
    run_s, naT, nrT, naC, nrC))

# --------------------------------------------------------------------------- #
# 5. Report (m/O3 ranges + host comparison).
# --------------------------------------------------------------------------- #
midx = sort!([idx for (nm, idx) in var_map if startswith(nm, "Transport3D.m[")])
o3idx = sort!([idx for (nm, idx) in var_map if startswith(nm, "SuperFast.O3[")])
mf = uT[midx]; o3 = uT[o3idx]
finiteT = all(isfinite, uT)
mrel = uH === nothing ? NaN : maximum(abs.(uT .- uH) ./ max.(abs.(uH), 1e-9))
ok = finiteT && all(>(0), mf) && (uH === nothing || mrel < 5e-2) && (!HOST_ARM || host_ok)
say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d window=ONE_XLA_PROGRAM solve_secs=%.0f | traced: nT=%d/%d nC=%d/%d run=%.2fs compile=%.1fs | host: nT=%d/%d nC=%d/%d run=%.2fs | m=[%.3e,%.3e] O3=[%.4e,%.4e] maxrel_vs_host=%.3e ok=%s",
    LABEL, NC, N, NS, SOLVE_SECS, naT, nrT, naC, nrC, run_s, comp_s,
    hostT[1], hostT[2], hostC[1], hostC[2], host_s,
    minimum(mf), maximum(mf), minimum(o3), maximum(o3), mrel, ok))
say("DONE $LABEL")
