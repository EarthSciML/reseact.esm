#!/usr/bin/env julia
# ===========================================================================
# tools/origin_identity.jl -- prove that making the native-grid origin a
# metaparameter changed NOTHING at the default origin.
# ===========================================================================
# `reseact.esm` used to write the GEOS-FP 4x5 slice origin into 84 index
# expressions as the bare literals 29 / 14 (the origins) and 28 / 13 (the
# one-cell-back flank a face reads). Those became LAT0 / LON0 and LAT0-1 /
# LON0-1, which close to 29 / 14 by default.
#
# The `-1` sites are the reason this script exists. Metaparameter substitution
# is a NAME -> INTEGER replacement with no folding (EarthSciAST
# `_substitute_metaparams`), so `13 + gie` did not become `13 + gie` again --
# it became `(14 - 1) + gie`, an extra op node sitting in an index position
# that the affine-stencil lowering has to see through. If it does not, the
# stencil silently falls off the affine path. Only a value comparison catches
# that, so:
#
#   RESEACT_MODEL_A (default: a git snapshot of the pre-change model)
#   RESEACT_MODEL_B (default: the working-tree reseact.esm)
#
# are each built at the same grid and compared BITWISE on u0 and on one
# evaluation of both split halves' RHS. Anything but "identical" is a
# regression, not a rounding difference: the index arithmetic is exact.
#
# Env: RESEACT_GRID (default "7,7,8"), RESEACT_T0 (default 64800).
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)
haskey(ENV, "ESS_KERNEL_CLASS_MERGE_DISABLE") || (ENV["ESS_KERNEL_CLASS_MERGE_DISABLE"] = "1")
using SciMLBase, DiffEqCallbacks
using Printf, Logging

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const NLON, NLAT, NLEV = parse.(Int, split(get(ENV, "RESEACT_GRID", "7,7,8"), ","))
const T0 = parse(Float64, get(ENV, "RESEACT_T0", "64800"))
const A = get(ENV, "RESEACT_MODEL_A", joinpath(REPO, "_origin_before.esm"))
const B = get(ENV, "RESEACT_MODEL_B", joinpath(REPO, "reseact.esm"))

# One build -> (u0, du_transport, du_chemistry) at t = T0.
function probe(model)
    mp = Dict("NLON" => NLON, "NLAT" => NLAT, "NLEV" => NLEV)
    run, ff = Logging.with_logger(Logging.NullLogger()) do
        docs = prepare_split_docs(model; metaparameters = mp)
        f = reseact_forcing(CHEMDIR)
        f = merge(f, (; const_arrays = GridResize.slice_hybrid_coefs(f.const_arrays, NLEV)))
        (build_split_run(docs, (T0, T0 + 1.0); providers = f.providers,
                         parameters = f.parameters, const_arrays = f.const_arrays), f)
    end
    foreach(d -> d.materialize!(), run.dms)
    u = copy(run.u0)
    dus = [zero(u) for _ in run.funcs]
    for (i, f!) in enumerate(run.funcs)
        f!(dus[i], u, run.p, T0)
    end
    return (; u, dus, nvars = length(run.var_map))
end

say("grid $(NLON)x$(NLAT)x$(NLEV), t = $T0")
say("A: $A")
say("B: $B")
a = probe(A); say("  A built: $(length(a.u)) states")
b = probe(B); say("  B built: $(length(b.u)) states")

ok = true
check(lbl, x, y) = begin
    same = length(x) == length(y) && all(i -> x[i] === y[i], eachindex(x))
    d = length(x) == length(y) ? maximum(abs, x .- y; init = 0.0) : NaN
    say(@sprintf("  %-14s bitwise=%-5s  maxabs=%.3e", lbl, same, d))
    global ok = ok && same
end
check("u0", a.u, b.u)
for i in eachindex(a.dus)
    check("du part[$i]", a.dus[i], b.dus[i])
end
say(ok ? "IDENTICAL -- the origin metaparameters are a no-op at their defaults" :
         "DIFFERENT -- the rewrite changed the model; do not ship it")
exit(ok ? 0 : 1)
