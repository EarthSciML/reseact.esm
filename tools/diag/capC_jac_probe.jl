#!/usr/bin/env julia
# ===========================================================================
# capC_jac_probe.jl -- does the SYMBOLIC BLOCK JACOBIAN survive the capacity build?
# ===========================================================================
# `ros23_step(..., jac = :sym)` needs EarthSciASTDiff's analytic band model, and
# the whole point of `jac = :sym` (adjoint_gradient.jl, blocker 2) is that :fd is
# only accurate to sqrt(eps) and :ad segfaults Enzyme-MLIR on a reverse sweep. A
# capacity build with no symbolic Jacobian would be a downgrade, so this checks
# `prepare_jacobian` accepts the surgered capacity document, still reports
# `block_diagonal`, and costs what a C-sized build should cost.
#
# The capacity document is an `AbstractDict`; `prepare_jacobian` dispatches on
# `EsmFile`/`FlattenedSystem`, so it goes in through `coerce_esm_file`.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
include(joinpath(@__DIR__, "_env.jl"))
using EarthSciAST, EarthSciIO, JSON3, Logging, Printf
const EA = EarthSciAST
include(joinpath(REPO, "prototypes", "reseact_3d_chem", "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
include(joinpath(REPO, "tools", "capacity_chem.jl")); using .CapacityChem
using EarthSciASTSplitter: split_system
using EarthSciASTDiff
say(s) = (println(s); flush(stdout))
const T0 = 5400.0
const CAPS = parse.(Int, split(get(ENV, "RESEACT_CAPS", "1024,4096"), ','))

f0 = reseact_forcing(joinpath(REPO, "prototypes", "reseact_3d_chem"); ndays = forcing_days_for(T0, T0 + 600))
ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, 6)))
merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
merged_param = Dict{String,Any}()
for (rawk, prov) in ff.providers
    k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
    EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
end
ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)

const CAPMP = Dict("NLON" => 70, "NLAT" => 45, "NLEV" => 6, "LON0" => 1, "LAT0" => 1)
t = time()
doc0 = Logging.with_logger(Logging.NullLogger()) do
    file = EA.load_path(joinpath(REPO, "reseact.esm"); metaparameters = CAPMP)
    flat = EA.flatten(file); pre = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre); promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    index_promoted_refs_by_loop!(EA.flattened_to_esm(parts[2]), promoted)
end
say(@sprintf("load+split (capacity) %.1f s", time() - t))

for C in CAPS
    cd_, meta = CapacityChem.capacity_doc(doc0, C; say = say)
    pa = CapacityChem.lane_buffers(meta, merged_param, C)
    ca = Dict{String,Any}(k => v for (k, v) in merged_const if k in meta.variables)
    ovc = Dict{String,Float64}(k => v for (k, v) in ov if k in meta.variables)
    tb = time()
    f, u0, p, _, vm = Logging.with_logger(Logging.NullLogger()) do
        EA.build_evaluator(cd_; form = :oop, parameter_overrides = ovc,
                           const_arrays = ca, param_arrays = pa)
    end
    say(@sprintf("  C=%-6d build_evaluator %7.1f s  (nstates=%d)", C, time() - tb, length(u0)))
    tj = time()
    jacE = Logging.with_logger(Logging.NullLogger()) do
        EarthSciASTDiff.prepare_jacobian(EA.coerce_esm_file(cd_); model_name = "Flattened",
            wrt = :states, build_kwargs = (; form = :oop, parameter_overrides = ovc,
                                             const_arrays = ca, param_arrays = pa))
    end
    say(@sprintf("  C=%-6d prepare_jacobian %6.1f s  structure=%s  oop=%s  nnz=%d",
                 C, time() - tj, jacE.structure, jacE.oop, length(jacE.prototype.nzval)))
end
say("CAPJAC_DONE")
