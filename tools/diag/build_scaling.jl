#!/usr/bin/env julia
# ===========================================================================
# build_scaling.jl -- WHY DOES THE BUILD COST SCALE WITH THE GRID?
# ===========================================================================
# EarthSciAST pins GRID-INDEPENDENCE of the compiled IR as a property
# (test/grid_invariance_test.jl): a document is O(1) in the grid, so the IR
# must be too -- a fixed set of access kernels whose count, spine shapes,
# descriptor widths and CSE recipe counts are facts of the DOCUMENT. Only
# per-lane DATA may grow with N, and at most linearly.
#
# ReSEACT does not behave that way: `build_evaluator` went 725 s -> 1379 s and
# `prepare_jacobian` 597 s -> 1316 s for a 4x cell count. Either the property
# is being violated, or the cost lives outside the IR the property covers.
#
# The always-on `_CASCADE_TALLY` says which. Every array equation lands on one
# of:
#     :affine / :affine_fused_retry / :scan   -- the O(1) polyhedral build
#     :percell_acc                            -- the O(#cells) scalarize
#                                                fallback, merged after the
#                                                fact by the class merge
# A build that is O(N) is a build with `:percell_acc` in the tally (or one
# whose class-merge REPAIR counters are nonzero -- with direct emission on
# those are expected to be zero).
#
# This builds BOTH split parts at a sequence of grids IN ONE PROCESS and
# prints, per grid: the wall time of each phase, the cascade tally, and the
# per-cell derived cost. Fit the last column: flat => O(1) build, rising =>
# the fallback is what is being paid for.
#
#   BS_GRIDS   comma list of NLONxNLATxNLEV (default 6x6x8,9x9x8,12x12x8,18x12x8)
#   BS_PARTS   1,2 (default both)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
const EA = EarthSciAST
include(joinpath(@__DIR__, "_env.jl"))

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const GRIDS = String.(split(get(ENV, "BS_GRIDS", "6x6x8,9x9x8,12x12x8,18x12x8"), ','))
const PARTS = parse.(Int, String.(split(get(ENV, "BS_PARTS", "1,2"), ',')))

# Forcing is grid-independent at load; only the hybrid coefs are sliced per
# NLEV. Load it ONCE so the per-grid timings are build, not I/O.
const F0 = reseact_forcing(CHEMDIR; ndays = 1)

function one_grid(spec::AbstractString)
    nlon, nlat, nlev = parse.(Int, split(spec, 'x'))
    slice = native_slice(lon0 = 11, lat0 = 29, nlon = nlon, nlat = nlat, nlev = nlev)
    mp = slice.metaparameters
    say("\n" * "="^78)
    say(@sprintf("  GRID %s   cells=%d", spec, nlon * nlat * mp["NLEV"]))
    say("="^78)

    res = Dict{Symbol,Any}(:cells => nlon * nlat * mp["NLEV"], :spec => spec)
    Logging.with_logger(Logging.NullLogger()) do
        t = time()
        file = EA.load_path(MODEL; metaparameters = mp)
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        res[:t_load] = time() - t

        t = time()
        splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
        docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted)
                for pt in splitparts]
        res[:t_split] = time() - t

        ff = merge(F0, (; const_arrays =
                          GridResize.slice_hybrid_coefs(F0.const_arrays, mp["NLEV"])))
        merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
        merged_param = Dict{String,Any}()
        for (rawk, prov) in ff.providers
            k = String(rawk)
            fld = EA._provider_const_field(EA.provider_sample(prov, 5400.0), k)
            EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
        end
        ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
        merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters))

        for i in PARTS
            EA._reset_cascade_tally!()
            dm = EA.DiscreteMaterializer()
            t = time()
            fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
                parameter_overrides = ov, const_arrays = merged_const,
                param_arrays = merged_param, materialize_out = dm)
            res[Symbol("t_build$i")] = time() - t
            res[Symbol("tally$i")] = copy(EA._CASCADE_TALLY)
            res[Symbol("nstates$i")] = length(u0i)
        end
    end
    return res
end

const ROWS = Any[]
for g in GRIDS
    r = try
        one_grid(g)
    catch e
        say("  FAILED: " * first(split(sprint(showerror, e), '\n')))
        continue
    end
    push!(ROWS, r)
    for i in PARTS
        say(@sprintf("  part %d: build %8.2f s   %.4f s/cell   nstates=%d",
                     i, r[Symbol("t_build$i")],
                     r[Symbol("t_build$i")] / r[:cells], r[Symbol("nstates$i")]))
        tl = sort(collect(r[Symbol("tally$i")]), by = x -> String(x[1]))
        say("           tally: " * join(["$(k)=$(v)" for (k, v) in tl], "  "))
    end
    say(@sprintf("  load %.2f s   split %.2f s", r[:t_load], r[:t_split]))
end

say("\n" * "="^78)
say("  SCALING  (t/cell flat => O(1) build; rising => O(N) fallback)")
say("="^78)
say(@sprintf("  %-12s %7s %10s %10s %12s %12s", "grid", "cells", "load", "split",
             "build1", "build2"))
for r in ROWS
    say(@sprintf("  %-12s %7d %10.2f %10.2f %12.2f %12.2f", r[:spec], r[:cells],
                 r[:t_load], r[:t_split],
                 get(r, :t_build1, NaN), get(r, :t_build2, NaN)))
end
