#!/usr/bin/env julia
# ===========================================================================
# build_res_window.jl -- WHY DOES BUILD TIME SCALE WITH *RESOLUTION*?
# ===========================================================================
# `build_scaling.jl` answered the other axis: how the build scales with the
# MODEL grid (cells). This one holds the model grid FIXED and varies the
# GEOS-FP native resolution, because the 0.25x0.3125 box325 arm (slurm
# 10395008) built in 1932.7 s against 330.5 s at 2x2.5 with an IDENTICAL cell
# count (25x13x72, nstates=304200) -- a 5.85x tax that the model grid cannot
# explain and that the compiled program does not show (the traced helper
# census is character-for-character the same at both resolutions).
#
# Two candidate mechanisms, and they need different fixes, so separate them:
#
#   (a) the build's own initial `provider_sample` of all 15 discrete providers,
#       which decodes each netCDF file WHOLE. Measured standalone by
#       decode_cost.jl: 6.95 s at 4x5, 18.04 s at 2x2.5, 1052.87 s at 0.25.
#   (b) everything downstream having to carry those whole-globe arrays --
#       7.8 GB resident at 0.25 vs 0.12 GB at 2x2.5 -- through
#       `build_evaluator`, as allocation and GC pressure.
#
# The third arm is the FIX, applied in memory rather than in the reader: slice
# every forcing array down to the halo-inclusive window the model actually
# reads (lon `LON0 .. LON0+NLON+1`, lat `LAT0 .. LAT0+NLAT`) and rebase LON0 /
# LAT0 onto it. `native_window_probe.jl` already showed the transport RHS is
# bit-identical when everything outside that window is NaN, and the emitted
# indexing is driven by the runtime array shape, not by GF_NLON/GF_NLAT -- so
# this is a rebasing and nothing more. It is what a windowed READER would hand
# over (INDEX_RANGE_READS.md steps 1-2), minus the decode saving, which arm
# (a) already prices separately.
#
#   BRW_RES    comma list of resolutions   (default 2x2.5,0.25x0.3125)
#   BRW_GRID   NLONxNLATxNLEV              (default 6x6x8 -- small on purpose:
#              the model-grid term is what build_scaling.jl measures, and here
#              it is a constant to be held down, not the signal)
#   BRW_ARMS   comma list of full,win      (default both)
#   BRW_PARTS  1,2                         (default both)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, SHA
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
const RESL  = String.(split(get(ENV, "BRW_RES", "2x2.5,0.25x0.3125"), ','))
const ARMS  = String.(split(get(ENV, "BRW_ARMS", "full,win"), ','))
const PARTS = parse.(Int, String.(split(get(ENV, "BRW_PARTS", "1,2"), ',')))
const NLON, NLAT, NLEV = parse.(Int, split(get(ENV, "BRW_GRID", "6x6x8"), 'x'))
const TSAMP = 5400.0

rss_gb() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^30
gc_s()   = Base.gc_time_ns() / 1e9

# --- the window the model actually reads, and the rebased metaparameters ----
# `native_slice` documents the extent in its own guards: lon `lon0 .. lon0+nlon+1`
# (west halo at `lon0`, east halo at `lon0+nlon+1`) and lat `lat0 .. lat0+nlat`
# (south flank at `lat0`). Taking that sub-box makes native index `lon0+k` the
# new index `k+1`, so the rebased origin is ONE, not zero: the emitted subscript
# is `LON0 + gi` with `gi` running from 0 (the west halo), so LON0 = 1 puts that
# first read on the first column of the windowed array. LON0 = 0 is what the
# first attempt used and it fails loudly --
# `E_TREEWALK_PGATHER_OOB: 'GEOSFP.PS' index 0 out of range [1, 8] on dim 3`.
# The DEGREE parameters are untouched: the box has not moved, only its indexing.
function window_arrays(param::AbstractDict, slice)
    lo, la = slice.lon0, slice.lat0
    ilon = lo:(lo + NLON + 1)
    ilat = la:(la + NLAT)
    out = Dict{String,Any}()
    for (k, v) in param
        a = v::Array{Float64}
        # The loader hands over the netCDF axis order with the record bracket
        # prepended: (record, lat, lon) for a surface field and
        # (record, lev, lat, lon) for a 3-D one -- lon is LAST. Verified against
        # decode_cost.jl, which prints these shapes at every resolution
        # ((2,46,72) / (2,72,46,72) at 4x5, (2,721,1152) / (2,72,721,1152) at
        # 0.25). Only the two trailing axes are windowed; records and levels are
        # kept whole because the model reads all of both.
        out[k] = ndims(a) == 3 ? Array{Float64}(a[:, ilat, ilon]) :
                 ndims(a) == 4 ? Array{Float64}(a[:, :, ilat, ilon]) :
                 error("$k has $(ndims(a)) dims; expected 3 or 4")
    end
    return out
end

function one_arm(resname::AbstractString, arm::AbstractString)
    slice = native_slice(res = resname, nlon = NLON, nlat = NLAT, nlev = NLEV)
    mp = copy(slice.metaparameters)
    say("\n" * "="^78)
    say(@sprintf("  RES %-12s ARM %-5s  model grid %dx%dx%d  native %dx%d",
                 resname, arm, NLON, NLAT, NLEV, slice.grid.nlon, slice.grid.nlat))
    say("="^78)
    r = Dict{Symbol,Any}(:res => resname, :arm => arm)

    Logging.with_logger(Logging.NullLogger()) do
        t = time(); g0 = gc_s()
        f0 = reseact_forcing(CHEMDIR; ndays = 1, res = resname)
        ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV)))
        r[:t_forcing] = time() - t

        # --- phase A: the build's own initial sample of every provider -------
        merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
        merged_param = Dict{String,Any}()
        t = time(); g = gc_s()
        for (rawk, prov) in ff.providers
            k = String(rawk)
            fld = EA._provider_const_field(EA.provider_sample(prov, TSAMP), k)
            EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
        end
        r[:t_sample] = time() - t; r[:gc_sample] = gc_s() - g
        r[:mb_param] = sum(length(v) * 8 / 1e6 for v in values(merged_param); init = 0.0)
        r[:rss_sample] = rss_gb()

        if arm == "win"
            t = time()
            merged_param = window_arrays(merged_param, slice)
            mp["LON0"] = 1; mp["LAT0"] = 1
            mp["GF_NLON"] = NLON + 2; mp["GF_NLAT"] = NLAT + 1
            r[:t_window] = time() - t
            r[:mb_param_win] = sum(length(v) * 8 / 1e6 for v in values(merged_param); init = 0.0)
            GC.gc(true)
            r[:rss_win] = rss_gb()
        end

        # --- phase B: document load / flatten / split (grid-only, no forcing) -
        t = time()
        file = EA.load_path(MODEL; metaparameters = mp)
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
        docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted)
                for pt in splitparts]
        r[:t_doc] = time() - t

        ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
        merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters))

        # --- phase C: build_evaluator, the part that carries the arrays ------
        dms = Any[]
        for i in PARTS
            EA._reset_cascade_tally!()
            dm = EA.DiscreteMaterializer()
            push!(dms, dm)
            t = time(); g = gc_s()
            fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
                parameter_overrides = ov, const_arrays = merged_const,
                param_arrays = merged_param, materialize_out = dm)
            r[Symbol("t_build$i")] = time() - t
            r[Symbol("gc_build$i")] = gc_s() - g
            r[Symbol("nstates$i")] = length(u0i)
            r[Symbol("tally$i")]   = copy(EA._CASCADE_TALLY)
            r[Symbol("rss_build$i")] = rss_gb()
            # ACCEPTANCE, not timing: a rebasing that is off by one still builds
            # and still reports the same cascade tally -- it just reads the wrong
            # native cells, which would make every number in this probe a
            # measurement of the wrong model. Hash the RHS at the build's own
            # base point so the `win` arm has to reproduce the `full` arm bit for
            # bit. (`native_window_probe.jl` already settled the PHYSICS claim
            # that nothing outside the window is read; this settles that THIS
            # slice-and-rebase implements that window correctly.)
            du = fi(u0i, pi, TSAMP)
            r[Symbol("sha$i")] = bytes2hex(sha256(reinterpret(UInt8, vec(du))))
            r[Symbol("finite$i")] = all(isfinite, du)
        end

        # --- phase D: the discrete-cadence materialization ------------------
        # The driver runs this INSIDE its timed BUILD region
        # (`foreach(d -> d.materialize!(), dms)`, adjoint_gradient.jl) and the
        # first version of this probe did not, which is the one structural
        # difference left between the two. It is also the only build phase that
        # is per-CELL *and* reads the forcing buffers, so it is the candidate
        # for the ~590 s of the box325 build delta that decode does not explain
        # and that a 6x6x8 model grid does not reproduce at all.
        t = time(); g = gc_s()
        foreach(d -> d.materialize!(), dms)
        r[:t_materialize] = time() - t; r[:gc_materialize] = gc_s() - g
        r[:rss_materialize] = rss_gb()
        r[:g0] = g0
    end
    return r
end

const ROWS = Any[]
for resname in RESL, arm in ARMS
    row = try
        one_arm(resname, arm)
    catch e
        say("  FAILED: " * first(split(sprint(showerror, e), '\n')))
        say("    " * sprint(showerror, e))
        continue
    end
    push!(ROWS, row)
    say(@sprintf("  forcing ctor     %8.2f s", row[:t_forcing]))
    say(@sprintf("  provider sample  %8.2f s   (gc %.2f s)   kept %8.1f MB   rss %.1f GB",
                 row[:t_sample], row[:gc_sample], row[:mb_param], row[:rss_sample]))
    haskey(row, :t_window) && say(@sprintf("  window slice     %8.2f s   kept %8.1f MB   rss %.1f GB",
                 row[:t_window], row[:mb_param_win], row[:rss_win]))
    say(@sprintf("  doc load+split   %8.2f s", row[:t_doc]))
    for i in PARTS
        say(@sprintf("  build_evaluator%d %8.2f s   (gc %.2f s)   nstates=%d   rss %.1f GB",
                     i, row[Symbol("t_build$i")], row[Symbol("gc_build$i")],
                     row[Symbol("nstates$i")], row[Symbol("rss_build$i")]))
        tl = sort(collect(row[Symbol("tally$i")]), by = x -> String(x[1]))
        say("        tally: " * join(["$(k)=$(v)" for (k, v) in tl], "  "))
        sha = row[Symbol("sha$i")]
        ref = row[:arm] == "full" ? nothing :
              findfirst(x -> x[:res] == row[:res] && x[:arm] == "full", ROWS)
        vs  = ref === nothing ? "" :
              ROWS[ref][Symbol("sha$i")] == sha ? "   == full arm  RHS IDENTICAL" :
                                                  "   != full arm  RHS DIFFERS"
        say(@sprintf("        rhs sha %s  finite=%s%s", first(sha, 16),
                     row[Symbol("finite$i")], vs))
    end
    say(@sprintf("  materialize!     %8.2f s   (gc %.2f s)   rss %.1f GB",
                 row[:t_materialize], row[:gc_materialize], row[:rss_materialize]))
    GC.gc(true)
end

say("\n" * "="^78)
say("  SUMMARY   (model grid fixed at $(NLON)x$(NLAT)x$(NLEV); only the native grid moves)")
say("="^78)
say(@sprintf("  %-12s %-5s %9s %9s %9s %9s %9s %9s %9s", "res", "arm",
             "sample", "param_MB", "doc", "build1", "build2", "matrlz", "total"))
for r in ROWS
    tot = r[:t_forcing] + r[:t_sample] + get(r, :t_window, 0.0) + r[:t_doc] +
          sum(get(r, Symbol("t_build$i"), 0.0) for i in PARTS) + r[:t_materialize]
    say(@sprintf("  %-12s %-5s %9.2f %9.1f %9.2f %9.2f %9.2f %9.2f %9.2f",
                 r[:res], r[:arm], r[:t_sample],
                 get(r, :mb_param_win, r[:mb_param]), r[:t_doc],
                 get(r, :t_build1, NaN), get(r, :t_build2, NaN),
                 r[:t_materialize], tot))
end
