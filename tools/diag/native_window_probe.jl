# ===========================================================================
# native_window_probe.jl -- does the model read only the native window it claims?
# ===========================================================================
# WHY. The forcing parameters are declared over the index sets gf_lon/gf_lat,
# whose sizes were the LITERALS 72/46 -- the 4x5 native extent -- while the
# arrays handed over at 2x2.5 are 144x91. If those declared sizes had folded
# into the emitted indexing, every non-4x5 run would have read the wrong cells
# and still produced a finite, plausible trajectory. This probe settles it
# empirically rather than by reading the emitter.
#
# METHOD. Build on a small box, evaluate the transport RHS, then write NaN into
# every cell of every forcing array OUTSIDE the halo window the model should be
# reading (native lon LON0..LON0+NLON+2, lat LAT0..LAT0+NLAT+1, generous by one
# on each side), and evaluate again. Identical => the model touched only the
# window. Anything else => it read elsewhere, and NaN makes that loud.
#
# MEASURED 2026-09-05, 6x6x8 box, t = 64800 s:
#
#   4x5    LON0=11 LAT0=29, native 72x46; kept lon 11:19 lat 29:36,
#          masked 3,790,800 cells => RHS bit-identical
#          (sha 4f820583fe4038b95f9b19825e0997c1ac721f0960938c202b2949cfd691aceb,
#          the same value tools/diag/res_switch_smoke.jl gates on).
#   2x2.5  LON0=22 LAT0=58, native 144x91; kept lon 22:30 lat 58:65,
#          masked 15,247,440 cells => RHS bit-identical
#          (sha 5cbdf571a2b7574166675f7e805a11a709244656064c6f6e1554fd0a07da3095).
#
# CONCLUSION. The runtime array shape drives the indexing; the declared index-set
# sizes are pure shape metadata for externally supplied arrays (they appear ONLY
# as F_* shape dims, never as an aggregate range -- 0 occurrences of
# `"from": "gf_lat"` in the model). They are now the metaparameters
# GF_NLON/GF_NLAT anyway, set from the resolution row, because a declaration
# that contradicts the data it describes is a trap even when nothing reads it.
#
# COROLLARY, and the reason this file is kept: the window is 3.6% of the native
# 4x5 grid and 2.3% of 0.25x0.3125, and the model provably needs nothing else --
# which is the measurement INDEX_RANGE_READS.md builds its case on.
#
#   julia --project=run-model-jl tools/diag/native_window_probe.jl          # 4x5
#   SMOKE_RES=2x2.5 julia --project=run-model-jl tools/diag/native_window_probe.jl
# ===========================================================================
# Does the built model read the NATIVE WINDOW it claims to, at a resolution whose
# native extent differs from the .esm's declared gf_lat/gf_lon index-set sizes?
#
# Method: build, evaluate the RHS, then NaN out every cell of every forcing array
# OUTSIDE the expected window and evaluate again. Same answer => the model only
# ever touched the window. NaN => it read somewhere else (or the declared sizes
# folded into the generated indexing).
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
using Logging, Printf, SHA
const EA = EarthSciAST

res = get(ENV, "SMOKE_RES", "4x5")
slice = native_slice(res = res, nlon = 6, nlat = 6, nlev = 8)
mp = slice.metaparameters; t0 = 64800.0
L0, B0 = slice.lon0, slice.lat0
NL, NB = mp["NLON"], mp["NLAT"]
println("res=$res  LON0=$L0 LAT0=$B0 NLON=$NL NLAT=$NB  native=$(slice.grid.nlon)x$(slice.grid.nlat)")

f1 = nothing; u0 = nothing; p = nothing; mpar = nothing
Logging.with_logger(Logging.NullLogger()) do
    global f1, u0, p, mpar
    file = EA.load_path(joinpath(REPO, "reseact.esm"); metaparameters = mp)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
    f0 = reseact_forcing(CHEMDIR; ndays = 1, res = res)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, mp["NLEV"])))
    mc = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    mpar = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, t0), k)
        EA.provider_is_const(prov) ? (mc[k] = fld) : (mpar[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters))
    fi, u0i, pi, _, _ = EA.build_evaluator(docs[1]; form = :oop,
        parameter_overrides = ov, const_arrays = mc, param_arrays = mpar)
    f1 = fi; u0 = u0i; p = pi
end
du0 = f1(u0, p, t0)
println("  before  sha ", bytes2hex(sha256(reinterpret(UInt8, vec(du0)))),
        "  finite=", all(isfinite, du0))

# Window, generous by one on every side: lon LON0-0 .. LON0+NLON+2, lat LAT0 .. LAT0+NLAT+1.
lonwin = max(1, L0):min(slice.grid.nlon, L0 + NL + 2)
latwin = max(1, B0):min(slice.grid.nlat, B0 + NB + 1)
println("  keeping native lon $(first(lonwin)):$(last(lonwin))  lat $(first(latwin)):$(last(latwin))")
function maskout!(mpar, lonwin, latwin, nlat_native)
    n = 0
    for (k, a) in mpar
        a isa AbstractArray || continue
        nd = ndims(a)
        (nd == 3 || nd == 4) || continue          # [t,lat,lon] or [t,lev,lat,lon]
        latax = nd - 1; lonax = nd
        size(a, latax) == nlat_native || (println("  SKIP $k dims $(size(a))"); continue)
        for I in CartesianIndices(a)
            if !(I[lonax] in lonwin && I[latax] in latwin)
                a[I] = NaN; n += 1
            end
        end
    end
    return n
end
nmask = maskout!(mpar, lonwin, latwin, slice.grid.nlat)
println("  masked $nmask cells outside the window")
du1 = f1(u0, p, t0)
println("  after   sha ", bytes2hex(sha256(reinterpret(UInt8, vec(du1)))),
        "  finite=", all(isfinite, du1))
println(du0 == du1 ? "  WINDOW OK: identical" : "  WINDOW MISMATCH: differs (nan count $(count(isnan, du1)))")
