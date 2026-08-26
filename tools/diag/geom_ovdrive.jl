#!/usr/bin/env julia
# ===========================================================================
# geom_ovdrive.jl -- DOES THE SETUP OVERLAP BROAD PHASE MAKE THE BUILD FLAT?
# ===========================================================================
# `_materialize_geom_array` now resolves a Phase-2a `overlap` join clause and
# DRIVES enumeration from its candidate set instead of filtering the full
# `#src x #tgt` product. ReSEACT's conservative regrid is the shape that
# motivated it: one 137,241 x NCOL weight matrix plus five contractions over the
# same product. If the drive works, `build_evaluator` stops tracking NCOL.
#
# Prints, per grid: build wall time, the overlap engagement counters
# (DRIVE / GATE_ONLY / NONE) and the sweep counters, then the slope table.
# Fit the slope over the grids AFTER the first (the first carries process JIT).
#
#   GO_GRIDS   comma list of NLONxNLATxNLEV (default 6x6x8,9x9x8,12x12x8,15x15x8)
#   GO_PART    1 | 2 (default 2, chemistry)
#   Set ESS_GEOM_OVERLAP_GATE_VERIFY=1 to additionally materialize every gated
#   array the OLD ungated way and assert `isequal` cell for cell.
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
const GRIDS = String.(split(get(ENV, "GO_GRIDS", "6x6x8,9x9x8,12x12x8,15x15x8"), ','))
const PART  = parse(Int, get(ENV, "GO_PART", "2"))
const F0 = reseact_forcing(CHEMDIR; ndays = 1)

function prep(spec)
    nlon, nlat, nlev = parse.(Int, split(spec, 'x'))
    slice = native_slice(lon0 = 11, lat0 = 29, nlon = nlon, nlat = nlat, nlev = nlev)
    mp = slice.metaparameters
    file = EA.load_path(MODEL; metaparameters = mp)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    sp = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in sp]
    ff = merge(F0, (; const_arrays =
                      GridResize.slice_hybrid_coefs(F0.const_arrays, mp["NLEV"])))
    mc = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    mparr = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, 5400.0), k)
        EA.provider_is_const(prov) ? (mc[k] = fld) : (mparr[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters))
    return docs, mc, mparr, ov, nlon * nlat * mp["NLEV"], nlon * nlat
end

const ROWS = Any[]
for g in GRIDS
    docs, mc, mparr, ovr, cells, cols = Logging.with_logger(Logging.NullLogger()) do
        prep(g)
    end
    EA._GEOM_OVERLAP_DRIVE[] = 0; EA._GEOM_OVERLAP_GATE_ONLY[] = 0
    EA._GEOM_OVERLAP_NONE[]  = 0
    EA._GEOM_SWEEP_FAST[] = 0;  EA._GEOM_SWEEP_REF[] = 0
    t = time()
    Logging.with_logger(Logging.NullLogger()) do
        dm = EA.DiscreteMaterializer()
        EA.build_evaluator(docs[PART]; form = :oop, parameter_overrides = ovr,
            const_arrays = mc, param_arrays = mparr, materialize_out = dm)
    end
    tb = time() - t
    say(@sprintf("  %-9s cols=%4d cells=%5d  build %8.2f s   overlap drive=%d gate_only=%d none=%d   sweep fast=%d ref=%d",
                 g, cols, cells, tb, EA._GEOM_OVERLAP_DRIVE[],
                 EA._GEOM_OVERLAP_GATE_ONLY[], EA._GEOM_OVERLAP_NONE[],
                 EA._GEOM_SWEEP_FAST[], EA._GEOM_SWEEP_REF[]))
    push!(ROWS, (spec = g, cols = cols, build = tb))
end

say("\n" * "="^70)
say("  SLOPE (exclude the first grid: it carries process JIT)")
say("="^70)
for k in 2:(length(ROWS) - 1)
    a, b = ROWS[k], ROWS[end]
    say(@sprintf("  %s -> %s : %.4f s/col", a.spec, b.spec,
                 (b.build - a.build) / (b.cols - a.cols)))
end
