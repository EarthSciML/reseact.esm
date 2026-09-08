#!/usr/bin/env julia
# ===========================================================================
# aktbl_ladder.jl -- HOW MUCH PER-BOX TABLE DOES THE AFFINE TIER MATERIALIZE?
# ===========================================================================
# `_derive_lane_repl` (stencil_affine.jl) does NOT decline a lane whose resolved
# index is non-affine across its box: it MATERIALIZES a dense per-box table
# (`_materialize_state_tbl` / `_materialize_pgather_tbl` / `_materialize_const_box`)
# by calling `_eval_recipe` once per box cell. That is O(cells) in both time and
# memory and emits ZERO extra `_compile` node lowerings -- exactly the signature
# BUILD_SCALING.md B3 is looking for.
#
# EarthSciAST already ships the attribution log for it (ESS_AK_TBL_DEBUG=1,
# `_AK_TBL_LOG`: rendered lane -> (#boxes, total entries)). This walks a grid
# ladder and prints total entries + per-lane top contributors, so the term is
# either proportional to cells or it is not.
#
#   AKT_GRIDS  comma list NLONxNLATxNLEV   (default 6x6x8,12x12x8,18x12x8)
#   AKT_PARTS  comma list                   (default 1)
#   AKT_TOP    rows per grid                (default 20)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
ENV["ESS_AK_TBL_DEBUG"] = "1"
using Printf, Logging, SHA
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
const EA = EarthSciAST
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const GRIDS = String.(split(get(ENV, "AKT_GRIDS", "6x6x8,12x12x8,18x12x8"), ','))
const PARTS = parse.(Int, String.(split(get(ENV, "AKT_PARTS", "1"), ',')))
const TOP   = parse(Int, get(ENV, "AKT_TOP", "20"))
const TSAMP = 5400.0
say("EarthSciAST from: " * String(pkgdir(EarthSciAST)))
const F0 = Logging.with_logger(Logging.NullLogger()) do
    reseact_forcing(CHEMDIR; ndays = 1)
end
rss_gb() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^30

for spec in GRIDS
    nlon, nlat, nlev = parse.(Int, split(spec, 'x'))
    slice = native_slice(lon0 = 11, lat0 = 29, nlon = nlon, nlat = nlat, nlev = nlev)
    mp = slice.metaparameters
    cells = nlon * nlat * mp["NLEV"]
    docs, mc, mparr, ov = Logging.with_logger(Logging.NullLogger()) do
        file = EA.load_path(MODEL; metaparameters = mp)
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        sp = split_system(flat, stencil_following_rule(flat); nparts = 2)
        d = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in sp]
        ff = merge(F0, (; const_arrays = GridResize.slice_hybrid_coefs(F0.const_arrays, mp["NLEV"])))
        c = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
        pa = Dict{String,Any}()
        for (rawk, prov) in ff.providers
            k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, TSAMP), k)
            EA.provider_is_const(prov) ? (c[k] = fld) : (pa[k] = fld)
        end
        o = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
        merge!(o, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters))
        (d, c, pa, o)
    end
    for i in PARTS
        EA._reset_ak_tbl_log!()
        EA._reset_cascade_tally!()
        EA._bench_reset!(); EA._BENCH_ON[] = true
        GC.gc(true)
        t = time()
        r = Logging.with_logger(Logging.NullLogger()) do
            dm = EA.DiscreteMaterializer()
            EA.build_evaluator(docs[i]; form = :oop, parameter_overrides = ov,
                const_arrays = mc, param_arrays = mparr, materialize_out = dm)
        end
        el = time() - t
        EA._BENCH_ON[] = false
        fi, u0i, pi, _, _ = r
        du = fi(u0i, pi, TSAMP)
        sha = bytes2hex(sha256(reinterpret(UInt8, vec(du))))
        PH = sort(collect(EA._BENCH_PHASE), by = x -> -x[2])
        L = EA._AK_TBL_LOG
        nb = sum(v[1] for v in values(L); init = 0)
        ne = sum(v[2] for v in values(L); init = 0)
        tl = sort(collect(EA._CASCADE_TALLY), by = x -> String(x[1]))
        say(@sprintf("\ngrid %-10s cells %7d part%d  build %8.2f s  nodes %9d  rss %.1f GB  sha %s",
                     spec, cells, i, el, EA._BENCH_COMPILE_CALLS[], rss_gb(), first(sha, 16)))
        say("   tally: " * join(["$(k)=$(v)" for (k, v) in tl], "  "))
        say(@sprintf("   cell_ckey calls: %d", get(EA._BENCH_PHASE_N, :cell_ckey, 0)))
        say("   PHASES (wall s, calls):")
        for (k, v) in PH
            say(@sprintf("     %-18s %9.3f s   gc %7.3f s   n=%d", String(k), v,
                         get(EA._BENCH_PHASE_GC, k, 0.0), get(EA._BENCH_PHASE_N, k, 0)))
        end
        say(@sprintf("   AK TABLES: distinct lanes %5d  boxes %8d  ENTRIES %12d  (%.3f entries/cell)",
                     length(L), nb, ne, ne / cells))
        for (k, v) in first(sort(collect(L), by = x -> -x[2][2]), TOP)
            say(@sprintf("     %10d entries  %6d boxes  %s", v[2], v[1], first(k, 110)))
        end
        r = nothing; fi = nothing; GC.gc(true)
    end
    docs = nothing; GC.gc(true)
end
