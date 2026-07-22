#!/usr/bin/env julia
# Validate the purpose-built TRACED adaptive integrator on the CHEMISTRY half:
# the whole adaptive Rosenbrock23 solve (FD block Jacobian, batched 13x13 block
# solves, hairer error norm, PI controller, accept/reject, stablehlo.while time
# loop) is ONE Reactant @compile'd XLA program -- not just the RHS forward pass.
#
# Three-way comparison over the same window and same merged RHS:
#   HOST : OrdinaryDiffEq Rosenbrock23(autodiff=false, LU) + block FD Jacobian
#          (the run_split_reactant.jl chemistry path, minus PositiveDomain)
#   TRACED: rx_traced_integrator.jl ros23_step + adaptive_solve under @compile
#   REF  : host solve at reltol=1e-7 (accuracy yardstick for both)
#
# Env: RESEACT_MODEL (default /tmp/reseact_7x7x8.esm), RESEACT_LABEL,
#      RESEACT_SOLVE_SECS (60), RESEACT_DT0 (0.5), RESEACT_RXENV, RESEACT_REF (1)
import Pkg
const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io=devnull)
using SciMLBase
import OrdinaryDiffEqRosenbrock
import LinearSolve
using LinearAlgebra, Printf
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system, stencil_vs_pointwise
using Reactant
using Logging
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))                   # reseact_forcing
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))                      # cellmajor_*, block_fd_jac
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", "/tmp/reseact_7x7x8.esm")
const LABEL = get(ENV, "RESEACT_LABEL", basename(MODEL))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "60"))
const DT0 = parse(Float64, get(ENV, "RESEACT_DT0", "0.5"))
const DO_REF = get(ENV, "RESEACT_REF", "1") == "1"
const T0 = 64800.0
const T_END = T0 + SOLVE_SECS
const RTOL = 1e-4; const ATOL = 1e-9
const LU = LinearSolve.LUFactorization()
const PS_REF = 101325.0

# --------------------------------------------------------------------------- #
# 1. Build ONLY the pointwise (chemistry) half, form=:oop over live forcing.
# --------------------------------------------------------------------------- #
say("=== $LABEL : build pointwise half ($MODEL) ===")
fo2 = u0 = p = var_map = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo2, u0, p, var_map
    file = EA.load(MODEL); flat = EA.flatten(file)
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
    parts = split_system(flat, stencil_vs_pointwise; nparts = 2)
    doc = EA.flattened_to_esm(parts[2])
    ff = reseact_forcing(CHEMDIR)
    merged_const = Dict{String,Any}(String(k)=>v for (k,v) in ff.const_arrays)
    merged_param = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld
        end
    end
    ov = Dict{String,Float64}(String(k)=>Float64(v) for (k,v) in ff.parameters)
    fo2, u0, p, _, var_map = EA.build_evaluator(doc; form=:oop,
        parameter_overrides=ov, const_arrays=merged_const, param_arrays=merged_param)
end
say(@sprintf("BUILD %s: %.2f s   nstates=%d", LABEL, time()-tb, length(u0)))

# Seed m(0) = dA + dB*ps_ref exactly as run_split_reactant.jl does (species-major
# via var_map names, so no layout assumptions).
let ff = reseact_forcing(CHEMDIR)
    dA = Float64.(ff.const_arrays["Transport3D.dA"])
    dB = Float64.(ff.const_arrays["Transport3D.dB"])
    n = 0
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dA[parse(Int, mm.captures[3])] + dB[parse(Int, mm.captures[3])] * PS_REF
        n += 1
    end
    say("  seeded m at $n states")
end

# --------------------------------------------------------------------------- #
# 2. Merge the chem kernels (bit-identity gated) -- both arms use the SAME rhs.
# --------------------------------------------------------------------------- #
include(joinpath(HERE, "rx_native_patch.jl"))   # AFTER `using Reactant`
include(joinpath(HERE, "rx_merge_lib.jl"))
include(joinpath(HERE, "rx_traced_integrator.jl"))
host_bufs = EA.forcing_buffers(fo2)
g4 = EA.rhs_with_buffers(fo2); kdesc = "stock"
try
    mg = RxOopMerge.merge_oop_rhs(getfield(fo2, :rhs))
    dmax = maximum(abs, mg.rhs(u0, p, T0, host_bufs) .- fo2(u0, p, T0))
    if dmax == 0.0
        global g4 = mg.rhs
        global kdesc = "merged $(mg.stats.n_kernels)->$(mg.stats.n_merged)"
    else
        say("  merge NOT bit-identical (maxabs=$dmax) -> stock rhs")
    end
catch e
    say("  merge FAILED ($(sprint(showerror, e))) -> stock rhs")
end
say("  chem rhs: $kdesc")

# --------------------------------------------------------------------------- #
# 3. Layout: cell-major perm for the host arm; species-major block masks for the
#    traced arm (asserts every species block is contiguous with one shared cell
#    ordering -- the traced FD Jacobian depends on that alignment).
# --------------------------------------------------------------------------- #
P = cellmajor_perm(var_map)
const NS = P.NS; const NC = P.NC; const N = P.N

function species_masks(var_map, NS, NC)
    N = NS * NC
    bygroup = Dict{String,Vector{Tuple{NTuple{3,Int},Int}}}()
    for (nm, idx) in var_map
        mm = match(r"^(.*)\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && error("unparseable state name '$nm'")
        c = (parse(Int, mm.captures[2]), parse(Int, mm.captures[3]), parse(Int, mm.captures[4]))
        push!(get!(bygroup, String(mm.captures[1]), Tuple{NTuple{3,Int},Int}[]), (c, idx))
    end
    length(bygroup) == NS || error("expected $NS species, got $(length(bygroup))")
    masks = Vector{Vector{Float64}}(undef, NS)
    seen = falses(NS); cellorder_ref = nothing
    for (b, v) in bygroup
        idxs = sort!([x[2] for x in v])
        idxs == collect(idxs[1]:(idxs[1] + NC - 1)) || error("species $b not contiguous")
        (idxs[1] - 1) % NC == 0 || error("species $b block misaligned")
        sblk = (idxs[1] - 1) ÷ NC + 1
        seen[sblk] && error("duplicate block index for $b")
        seen[sblk] = true
        co = [x[1] for x in sort(v; by = x -> x[2])]
        if cellorder_ref === nothing
            cellorder_ref = co
        else
            co == cellorder_ref || error("cell order differs for species $b")
        end
        mask = zeros(N); mask[idxs[1]:idxs[end]] .= 1.0
        masks[sblk] = mask
    end
    all(seen) || error("missing species blocks")
    return masks
end
masks = species_masks(var_map, NS, NC)
say("  species-major layout verified: NS=$NS NC=$NC")

# --------------------------------------------------------------------------- #
# 4. HOST reference arm (and tight-tolerance accuracy yardstick).
# --------------------------------------------------------------------------- #
function host_iip(g4, bufs)
    return function (du, u, p, t)
        copyto!(du, g4(u, p, t, bufs))
        return nothing
    end
end
tgz(g, u, p, t) = (fill!(g, 0); nothing)
zerof!(du, u, p, t) = (fill!(du, 0); nothing)
f_sm! = host_iip(g4, host_bufs)
f_cm! = cellmajor_rhs(f_sm!, P.sm_of_cm)
jac_cm!, mkjp = block_fd_jac(f_cm!, NS, NC)
u0cm = u0[P.sm_of_cm]
fH = SciMLBase.ODEFunction(f_cm!; jac=jac_cm!, jac_prototype=mkjp(), tgrad=tgz)

to_sm(ucm) = (usm = similar(ucm); for i in 1:N; usm[P.sm_of_cm[i]] = ucm[i]; end; usm)

th = time()
solH = SciMLBase.solve(SciMLBase.SplitODEProblem(fH, zerof!, u0cm, (T0, T_END), p),
    OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
    reltol=RTOL, abstol=ATOL, dt=DT0, save_everystep=false, maxiters=500000)
host_s = time() - th
uH = to_sm(solH.u[end])
say(@sprintf("HOST   solve: %6.2f s  nacc=%d nrej=%d rc=%s",
    host_s, solH.stats.naccept, solH.stats.nreject, solH.retcode))

uREF = nothing
if DO_REF
    tref = time()
    solR = SciMLBase.solve(SciMLBase.SplitODEProblem(fH, zerof!, u0cm, (T0, T_END), p),
        OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false, linsolve=LU);
        reltol=1e-7, abstol=1e-12, save_everystep=false, maxiters=2_000_000)
    global uREF = to_sm(solR.u[end])
    say(@sprintf("REF    solve: %6.2f s  nacc=%d rc=%s", time()-tref, solR.stats.naccept, solR.retcode))
end

# --------------------------------------------------------------------------- #
# 5. TRACED arm: @compile the entire adaptive solve.
# --------------------------------------------------------------------------- #
const CTRL = RxTracedIntegrator.pictrl_ros23()
adv = let g4 = g4, NS = NS, NC = NC, masks = masks, atol = ATOL, rtol = RTOL, ctrl = CTRL
    (u0R, pR, t0R, teR, dt0R, bufsR) -> begin
        # traced deps (params, forcing buffers) flow via aux, NOT closure capture
        fstep = (u, t, dtc, aux) -> RxTracedIntegrator.ros23_step(
            (uu, tt) -> g4(uu, aux.p, tt, aux.bufs), u, t, dtc, NS, NC, masks, atol, rtol)
        RxTracedIntegrator.adaptive_solve(fstep, u0R, t0R, teR, dt0R, ctrl,
            (p=pR, bufs=bufsR))
    end
end
_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
dev_bufs = map(RX.ConcreteRArray, host_bufs)
EA.sync_forcing!(dev_bufs, EA.forcing_buffers(fo2))
UR = RX.ConcreteRArray(u0); PRd = _dev(p)
targs() = (UR, PRd, RX.ConcreteRNumber(T0), RX.ConcreteRNumber(T_END),
           RX.ConcreteRNumber(DT0), dev_bufs)
tc = time()
xadv = RX.@compile sync=true adv(targs()...)
comp_s = time() - tc
say(@sprintf("TRACED @compile: %.1f s", comp_s))
report_patch_stats()
tr = time()
res = xadv(targs()...)
run_s = time() - tr
uT = Array(res[1])
tT = Float64(res[2]); naccT = Int(round(Float64(res[4]))); nrejT = Int(round(Float64(res[5])))
say(@sprintf("TRACED solve: %6.2f s  nacc=%d nrej=%d t_end=%.6f", run_s, naccT, nrejT, tT))

# --------------------------------------------------------------------------- #
# 6. Compare.
# --------------------------------------------------------------------------- #
relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-9))
mrTH = relerr(uT, uH)
errT = uREF === nothing ? NaN : relerr(uT, uREF)
errH = uREF === nothing ? NaN : relerr(uH, uREF)
finiteT = all(isfinite, uT)
reached = abs(tT - T_END) <= 1e-6
ok = finiteT && reached &&
     (uREF === nothing ? mrTH < 5e-2 : errT <= max(2 * errH, 1e-6))
say(@sprintf("RESULT label=%s N=%d NS=%d NC=%d dt0=%.3g rhs=[%s] | traced nacc=%d nrej=%d | host nacc=%d nrej=%d | maxrel(T,H)=%.3e errT(ref)=%.3e errH(ref)=%.3e | compile=%.1fs run=%.2fs host=%.2fs | ok=%s",
    LABEL, N, NS, NC, DT0, kdesc, naccT, nrejT, solH.stats.naccept, solH.stats.nreject,
    mrTH, errT, errH, comp_s, run_s, host_s, ok))
say("DONE $LABEL")
