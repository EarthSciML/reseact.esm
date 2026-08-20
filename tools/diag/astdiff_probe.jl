#!/usr/bin/env julia
# ===========================================================================
# astdiff_probe.jl -- can EarthSciASTDiff's symbolic Jacobian replace ReSEACT's
# FD block Jacobian on the TRACED arm?
# ===========================================================================
# WHY THIS IS THE QUESTION THAT MATTERS. Profiling put the FD block Jacobian at
# the top of the cost table: one chemistry step costs 6.8 RHS-equivalents at
# CONUS, and the 13 colour evaluations dominate that. An analytical Jacobian is
# ~1 RHS-equivalent, i.e. the ~2.5x lever -- the largest one left now that the
# race question is settled and the replay is understood. The existing exact
# alternative (RESEACT_RXJAC=ad) is unusable: it segfaults under reverse-over-
# forward, which is why jac=:fd is the default in rx_traced_integrator.jl.
#
# THE ONE PROPERTY THAT DECIDES FEASIBILITY. run_reseact_reactant.jl builds both
# halves with `form = :oop`, and says why (line ~150): "an in-place f! captures
# host scratch per node and cannot trace". So a symbolic Jacobian is only usable
# on the traced arm if ITS evaluator can also be built :oop. EarthSciASTDiff's
# v1 evaluates band coefficients as a DERIVED ESM MODEL through the unmodified
# EarthSciAST.build_evaluator, and `prepare_jacobian(...; build_kwargs = ...)`
# forwards kwargs to it (assemble.jl:138,152). So `form = :oop` should pass
# straight through. This probe tests that end to end rather than assuming it.
#
# WHAT IT REPORTS
#   1. does prepare_jacobian survive the real chemistry half at all
#   2. the detected structure -- MUST be :block_diagonal for the per-cell 13x13
#      shape rx_traced_integrator.jl's blocksolve wants
#   3. pattern size / nnz / per-cell block density
#   4. cost: one-off prepare, and per-call fill vs one RHS evaluation -- the
#      ratio that decides whether this actually buys the 2.5x
#   5. CORRECTNESS: J*v against a ForwardDiff JVP through the same RHS
#
# ENV. Needs the PINNED EarthSciAST (reseact.esm does not load against
# EarthSciAST main since PR #167 renamed `examples` -> `analyses`; see
# tools/diag/native_24h_reference.sbatch). Point RESEACT_RXENV at an env that
# has both the pinned EarthSciAST and EarthSciASTDiff dev'd in.
#
#   RESEACT_NLON/NLAT/NLEV default to a SMALL 6x6x8 grid here -- this is a
#   feasibility probe, not a benchmark. Cost numbers at this size are indicative
#   only; the step/RHS ratio is what transfers, not the absolute times.
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Printf, Statistics, Logging, SparseArrays
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using EarthSciASTDiff
using ForwardDiff
const EA = EarthSciAST
const ED = EarthSciASTDiff

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
_env(k, d)  = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 6), nlat = _env("NLAT", 6),
                           nlev = _env("NLEV", 8))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]

say("=== astdiff probe: grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF ===")

# --- build both halves exactly as run_reseact_reactant.jl does ------------- #
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); merged_const = Dict{String,Any}()
ov = Dict{String,Float64}(); docs = nothing; ff = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, merged_const, ov, docs, ff
    file = EA.load(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
    f0 = reseact_forcing(CHEMDIR; ndays = 1)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi
        i == 1 && (u0 = u0i; p = pi; var_map = vmi)
    end
end
foreach(d -> d.materialize!(), dms)
say(@sprintf("BUILD %.2f s   nstates=%d", time() - tb, length(u0)))

# Chemistry is part 2 (the pointwise half); it is the one with the block
# Jacobian the Rosenbrock step needs.
const CHEM_DOC = docs[2]
gC = EA.rhs_with_buffers(fo[2])
bufsC = EA.forcing_buffers(fo[2])
u = copy(u0)

# --- 1/2/3. prepare the symbolic Jacobian --------------------------------- #
say("\n---- prepare_jacobian on the CHEMISTRY half, build_kwargs form=:oop ----")
bk = (; form = :oop, parameter_overrides = ov, const_arrays = merged_const,
        param_arrays = merged_param)
jac = nothing
tp = time()
try
    global jac = ED.prepare_jacobian(CHEM_DOC; wrt = :states, build_kwargs = bk)
    say(@sprintf("  prepare_jacobian OK in %.2f s", time() - tp))
catch e
    say("  prepare_jacobian FAILED after $(round(time()-tp, digits=2)) s:")
    say("    " * first(sprint(showerror, e), 1500))
    exit(1)
end

J = copy(ED.jac_prototype(jac))
say(@sprintf("  structure = %s   size = %d x %d   nnz = %d  (%.4f%% dense)",
             jac.structure, size(J, 1), size(J, 2), nnz(J),
             100 * nnz(J) / (size(J, 1) * size(J, 2))))
NS = length(var_map) == 0 ? 0 : 0   # reported below from the block shape instead
say("  NOTE: :block_diagonal is what rx_traced_integrator.jl's blocksolve wants;")
say("        anything else means the per-cell 13x13 assumption does not hold.")

# --- 4. cost: fill vs one RHS evaluation ---------------------------------- #
say("\n---- cost ----")
jac(J, u, p, T0)                                  # warm
t1 = time(); for _ in 1:5; jac(J, u, p, T0); end
tj = (time() - t1) / 5
gC(u, p, T0, bufsC)                               # warm
t2 = time(); for _ in 1:5; gC(u, p, T0, bufsC); end
tr = (time() - t2) / 5
say(@sprintf("  symbolic jac! fill      %8.3f ms", 1000 * tj))
say(@sprintf("  one chemistry RHS eval  %8.3f ms", 1000 * tr))
say(@sprintf("  jac / RHS = %.2f x   (the FD block Jacobian costs ~13 colour evals;", tj / tr))
say( "                        under ~13x here means the analytic route wins)")

# --- 5. correctness: J*v vs a ForwardDiff JVP through the SAME RHS --------- #
say("\n---- correctness: J*v vs ForwardDiff JVP ----")
try
    v = randn(length(u))
    ud = ForwardDiff.Dual{:jvp}.(u, v)
    dud = gC(ud, p, T0, bufsC)
    ref = ForwardDiff.partials.(dud, 1)
    got = J * v
    num = norm(got .- ref); den = max(norm(ref), 1e-300)
    say(@sprintf("  ||Jv - ref|| / ||ref|| = %.3e   %s",
                 num / den, num / den <= 1e-10 ? "PASS" : "FAIL"))
    k = argmax(abs.(got .- ref))
    say(@sprintf("  worst component %d: sym=%.12e ref=%.12e", k, got[k], ref[k]))
catch e
    say("  ForwardDiff JVP unavailable through the :oop RHS:")
    say("    " * first(sprint(showerror, e), 600))
    say("  (falling back to a central finite difference)")
    v = randn(length(u)); h = 1e-7
    ref = (gC(u .+ h .* v, p, T0, bufsC) .- gC(u .- h .* v, p, T0, bufsC)) ./ (2h)
    got = J * v
    num = norm(got .- ref); den = max(norm(ref), 1e-300)
    say(@sprintf("  ||Jv - ref|| / ||ref|| = %.3e  (FD ref, ~1e-7 floor)  %s",
                 num / den, num / den <= 1e-5 ? "PASS" : "FAIL"))
end

say("\nPROBE_DONE")
