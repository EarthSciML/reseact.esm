#!/usr/bin/env julia
# End-to-end operator-split ReSEACT solve with BOTH halves traced by Reactant/XLA.
#
# This is the payoff of the split-and-trace approach (see reactant_handoff/README.md):
#   * split the model into [transport, pointwise] with EarthSciASTSplitter,
#   * build EACH half as an out-of-place RHS (form=:oop) over ONE shared set of live
#     GEOS-FP forcing buffers,
#   * @compile EACH half through rhs_with_buffers (buffers as traced args),
#   * drive the SAME Lie-Trotter scheme as tools/run_scale.jl -- SSPRK43 for the
#     (non-stiff) transport half, then Rosenbrock23 with a block-diagonal FD Jacobian
#     for the (stiff, cell-local) chemistry half -- but with the compiled XLA RHS,
#   * refresh forcing at each GEOS-FP cadence boundary on the host and mirror it to the
#     device buffers with sync_forcing! (no retrace).
#
# Env:
#   RESEACT_MODEL      full .esm to run          (default: repo-root reseact.esm)
#   RESEACT_LABEL      tag for the RESULT line   (default: basename of the model)
#   RESEACT_SOLVE_SECS solve window in seconds   (default 60)
#   RESEACT_RXENV      Julia env with Reactant   (default: <repo>/run-model-jl)
import Pkg
const HERE  = @__DIR__
const REPO  = normpath(joinpath(HERE, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io=devnull)
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
include(joinpath(CHEMDIR, "split_common.jl"))                   # reseact_forcing
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))                      # cellmajor_perm/rhs, block_fd_jac
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

# --------------------------------------------------------------------------- #
# 1. Split + build both halves as out-of-place RHS over shared live forcing.
#    Mirrors split_common.build_split_run's forcing wiring, but form=:oop (no
#    DiscreteMaterializer: derived forcing quantities are computed INSIDE the traced
#    program from the raw buffers, so refreshing the raw buffers is enough).
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
    merged_const = Dict{String,Any}(String(k)=>v for (k,v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld; discrete_providers[k] = prov
        end
    end
    ov = Dict{String,Float64}(String(k)=>Float64(v) for (k,v) in ff.parameters)

    for i in 1:2
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form=:oop,
            parameter_overrides=ov, const_arrays=merged_const, param_arrays=merged_param)
        fo[i] = fi
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part 2 var_map != part 1")
        end
    end
end
say(@sprintf("BUILD %s: %.2f s   nstates=%d", LABEL, time()-tb, length(u0)))

# --------------------------------------------------------------------------- #
# 2. Trace each half through rhs_with_buffers -- BOTH on XLA. Two fixes make the
#    transport half compile (see RESULTS.md §2026-07-21):
#      * rx_native_patch.jl lowers primitive broadcasts to native stablehlo ops
#        (no per-site *_broadcast_scalar helper -> no 10000-name cap),
#      * rx_merge_lib.jl merges the per-cell-fragmented :oop kernels into
#        lane-batched class kernels (7.93M -> 1.06M IR nodes at 7x7x8), which is
#        REQUIRED: the unmerged closure hangs Reactant's capture walk for hours.
#    The merged host RHS is verified bit-identical against the stock evaluator
#    before it is used; on any mismatch we fall back to the stock RHS for that
#    half. A half that still fails to compile falls back to the (merged, if
#    verified) interpreted HOST eval reading the same live forcing buffers.
# --------------------------------------------------------------------------- #
include(joinpath(HERE, "rx_native_patch.jl"))   # AFTER `using Reactant`
include(joinpath(HERE, "rx_merge_lib.jl"))
const PARTNAME = ("transport", "pointwise")
_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
const UR = RX.ConcreteRArray(u0); const PR = _dev(p)
host_bufs = [EA.forcing_buffers(fo[i]) for i in 1:2]
dev_bufs  = [map(RX.ConcreteRArray, host_bufs[i]) for i in 1:2]
on_xla    = falses(2)

# Species-major in-place RHS for a compiled XLA half: host u -> device, run,
# device du -> host (tested idiom, test/reactant_oop_test.jl:201: fresh
# ConcreteRArray per call). Parameters constant (PR); forcing via aliased dev buffers.
function reactant_iip(xla, dev)
    return function (du, u, p, t)
        res = xla(RX.ConcreteRArray(u), PR, RX.ConcreteRNumber(t), dev)
        copyto!(du, Array(res))
        return nothing
    end
end
# Fallback: interpreted out-of-place HOST eval of a 4-arg (u,p,t,buffers) rhs
# reading the LIVE host buffers (refreshed in place by the callback -- no sync).
function host_iip(g4, bufs)
    return function (du, u, p, t)
        copyto!(du, g4(u, p, t, bufs))
        return nothing
    end
end

# Which halves to ATTEMPT compiling (comma-sep part indices). Default "1,2":
# with the merge + native lowering both halves compile.
const XLA_PARTS = Set(parse(Int, s) for s in split(get(ENV, "RESEACT_XLA_PARTS", "1,2"), ',') if !isempty(s))
f_sm = Vector{Function}(undef, 2)
for i in 1:2
    # Merge the half's kernels; keep only if bit-identical to the stock evaluator.
    g = EA.rhs_with_buffers(fo[i]); tm = time()
    local kdesc = "stock"
    try
        mg = RxOopMerge.merge_oop_rhs(getfield(fo[i], :rhs))
        du_ref = fo[i](u0, p, T0)
        du_mrg = mg.rhs(u0, p, T0, host_bufs[i])
        dmax = maximum(abs, du_mrg .- du_ref)
        if dmax == 0.0
            g = mg.rhs
            kdesc = "merged $(mg.stats.n_kernels)->$(mg.stats.n_merged) kernels"
        else
            say("  half[$i]=$(PARTNAME[i]) merge NOT bit-identical (maxabs=$dmax) -> stock rhs")
        end
        say(@sprintf("  half[%d]=%s merge: %s (%.1f s, blocked=%d)", i, PARTNAME[i],
            kdesc, time()-tm, mg.stats.n_blocked))
    catch e
        say("  half[$i]=$(PARTNAME[i]) merge FAILED ($(sprint(showerror, e))) -> stock rhs")
    end
    if !(i in XLA_PARTS)
        say("  half[$i]=$(PARTNAME[i]): CPU (interpreted oop [$kdesc]; XLA not attempted)")
        f_sm[i] = host_iip(g, host_bufs[i]); on_xla[i] = false; continue
    end
    tc = time()
    try
        xla = RX.@compile sync=true g(UR, PR, RX.ConcreteRNumber(T0), dev_bufs[i])
        say(@sprintf("  half[%d]=%s [%s] @compile OK in %.1f s -> XLA", i, PARTNAME[i], kdesc, time()-tc))
        f_sm[i] = reactant_iip(xla, dev_bufs[i]); on_xla[i] = true
    catch e
        msg = sprint(showerror, e); reason = String(last(filter(!isempty, split(msg, '\n'))))
        say("  half[$i]=$(PARTNAME[i]) @compile FAILED -> interpreted CPU fallback ($reason)")
        f_sm[i] = host_iip(g, host_bufs[i]); on_xla[i] = false
    end
end
f_sm_trans!, f_sm_chem! = f_sm[1], f_sm[2]

# --------------------------------------------------------------------------- #
# 3. Cell-major reorder + block-diagonal chem Jacobian (as in run_scale.jl).
# --------------------------------------------------------------------------- #
P = cellmajor_perm(var_map)
f_trans_cm! = cellmajor_rhs(f_sm_trans!, P.sm_of_cm)
f_chem_cm!  = cellmajor_rhs(f_sm_chem!,  P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_chem_cm!, P.NS, P.NC)

u0cm = u0[P.sm_of_cm]
mb = P.base_pos["Transport3D.m"]
mrng() = mb:P.NS:P.N
let dA = Float64.(reseact_forcing(CHEMDIR).const_arrays["Transport3D.dA"]),
    dB = Float64.(reseact_forcing(CHEMDIR).const_arrays["Transport3D.dB"])
    for c in P.cells
        u0cm[(P.cell_pos[c]-1)*P.NS + mb] = dA[c[3]] + dB[c[3]]*PS_REF
    end
end

# --------------------------------------------------------------------------- #
# 4. Forcing refresh: rewrite host buffers at each cadence boundary, then mirror
#    host -> device with sync_forcing! (no retrace). Initial sync before solving.
# --------------------------------------------------------------------------- #
# Only XLA halves need a host->device mirror; a host-fallback half reads the live
# buffers directly, so it sees the refresh with no sync.
sync_all() = for i in 1:2
    on_xla[i] && EA.sync_forcing!(dev_bufs[i], EA.forcing_buffers(fo[i]))
end
sync_all()
cb = nothing; tstops = Float64[]
if !isempty(discrete_providers)
    cb, tstops = EA.build_refresh_callback(; providers = discrete_providers,
        buffers = EA.RefreshBuffers(merged_param), post_refresh = sync_all)
end

# --------------------------------------------------------------------------- #
# 5. Lie-Trotter step: transport (SSPRK43) then chemistry (Rosenbrock23/BlockDiag).
# --------------------------------------------------------------------------- #
zerof!(du,u,p,t) = (fill!(du,0); nothing)
tgz(g,u,p,t)     = (fill!(g,0); nothing)
ts = time()
u = copy(u0cm)
pT = SciMLBase.ODEProblem(f_trans_cm!, u, (T0, T_END), p)
sT = SciMLBase.solve(pT, OrdinaryDiffEqSSPRK.SSPRK43(); reltol=RTOL, abstol=1e-6,
    callback = cb === nothing ? DiffEqCallbacks.PositiveDomain(copy(u)) :
        SciMLBase.CallbackSet(cb, DiffEqCallbacks.PositiveDomain(copy(u))),
    tstops=tstops, save_everystep=false, maxiters=500000)
u = sT.u[end]
f1c = SciMLBase.ODEFunction(f_chem_cm!; jac=jac_cm!, jac_prototype=mkjp(), tgrad=tgz)
pC = SciMLBase.SplitODEProblem(f1c, zerof!, u, (T0, T_END), p)
sC = SciMLBase.solve(pC, OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
    reltol=RTOL, abstol=ATOL, callback=DiffEqCallbacks.PositiveDomain(copy(u)),
    save_everystep=false, maxiters=500000)
u = sC.u[end]
solve_s = time() - ts

mf = u[mrng()]
o3b = P.base_pos["SuperFast.O3"]; o3 = u[o3b:P.NS:P.N]
ok = rc_ok(sT.retcode) && rc_ok(sC.retcode) && all(isfinite, u) && all(>(0), mf)
backend(i) = on_xla[i] ? "XLA" : "CPU"
say(@sprintf("RESULT label=%s cells=%d nstates=%d NS=%d transport=%s chem=%s solve_s=%.2f solve_secs=%.0f nT=%d nC=%d rcT=%s rcC=%s m=[%.3e,%.3e] O3=[%.4e,%.4e] ok=%s",
    LABEL, P.NC, P.N, P.NS, backend(1), backend(2), solve_s, SOLVE_SECS, sT.stats.naccept, sC.stats.naccept,
    sT.retcode, sC.retcode, minimum(mf), maximum(mf), minimum(o3), maximum(o3), ok))
say("DONE $LABEL")
