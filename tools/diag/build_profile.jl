#!/usr/bin/env julia
# ===========================================================================
# build_profile.jl -- WHICH HOST FUNCTION EATS THE CHEMISTRY BUILD?
# ===========================================================================
# `_CASCADE_TALLY` is byte-identical at 288 and 648 cells (affine=213,
# classmerge=22) yet the chemistry build still grows 203.3 s -> 260.5 s. So the
# grid dependence is NOT the routing; it is somewhere below the pinned
# grid-invariance property. A sampling profile over one build says where,
# without guessing.
#
# The standing hypothesis, from reading the code: `_InterpBilinearLaneSpec`
# (registered_functions.jl:1073) transposes a merged class's interp specs into
# PER-LANE COLUMNS --
#     for i in 1:Nx, j in 1:Ny
#         table_cols[i, j] = Float64[specs[l].table[i][j] for l in 1:L]
# -- which is Nx*Ny + Nx + Ny length-L vectors. ReSEACT's FastJX table is 61x23
# (the CONUS module carries f64[117936,61] and f64[117936,23] buffers), so that
# is 1,487 vectors x L: 1.40 GB of host allocation at CONUS lane counts, 5.6 GB
# at 4x. O(cells x table), built and content-hashed at build time.
#
# Prints the flat profile (self time) so the hypothesis is either at the top or
# it is wrong.
#
#   BP_GRID   NLONxNLATxNLEV (default 9x9x8)
#   BP_PART   1 | 2 (default 2, chemistry)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, Profile
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
const GRID  = get(ENV, "BP_GRID", "9x9x8")
const PART  = parse(Int, get(ENV, "BP_PART", "2"))

function setup(spec)
    nlon, nlat, nlev = parse.(Int, split(spec, 'x'))
    slice = native_slice(lon0 = 11, lat0 = 29, nlon = nlon, nlat = nlat, nlev = nlev)
    mp = slice.metaparameters
    F0 = reseact_forcing(CHEMDIR; ndays = 1)
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
    return docs, mc, mparr, ov, nlon * nlat * mp["NLEV"]
end

docs, mc, mparr, ov, cells = Logging.with_logger(Logging.NullLogger()) do
    setup(GRID)
end
say("grid=$GRID cells=$cells part=$PART -- profiling build_evaluator")

Profile.init(n = 20_000_000, delay = 0.005)
t = time()
Logging.with_logger(Logging.NullLogger()) do
    dm = EA.DiscreteMaterializer()
    @profile EA.build_evaluator(docs[PART]; form = :oop, parameter_overrides = ov,
        const_arrays = mc, param_arrays = mparr, materialize_out = dm)
end
say(@sprintf("build %.2f s", time() - t))

say("\n" * "="^78)
say("  FLAT PROFILE (self time, top 45)")
say("="^78)
Profile.print(; format = :flat, sortedby = :count, mincount = 20, maxdepth = 200)
say("\n" * "="^78)
say("  TREE (top of stack)")
say("="^78)
Profile.print(; format = :tree, maxdepth = 28, mincount = 60)
