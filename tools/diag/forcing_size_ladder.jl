#!/usr/bin/env julia
# ===========================================================================
# forcing_size_ladder.jl -- ISOLATE ARRAY SIZE FROM WORK.
# ===========================================================================
# `build_res_window.jl` varies the native resolution, which moves THREE things
# at once: the decode cost, the resident array size, and the model's own
# geometry parameters. That is enough to price the decode (A1) but not enough
# to say WHY the ~590 s residual of the box325 build exists, because a
# resolution change is not a controlled experiment.
#
# This probe is the controlled one. It decodes ONCE, always at 2x2.5, and then
# EMBEDS that same 2x2.5 halo window into a synthetic native array of whatever
# shape the rung asks for, rebasing LON0/LAT0 onto the embedding offset. So
# across the whole ladder:
#
#   * the model grid is identical            (same cells, same equations)
#   * the model geometry is identical        (same degree parameters)
#   * the VALUES the model reads are identical (bit for bit -- the RHS hash
#     is the acceptance test, and it must not move across the ladder)
#   * the number of gathers is identical
#   * ONLY the size of the array being gathered from changes.
#
# That is precisely "hold the access count fixed and scale the array", which
# separates a LOCALITY mechanism from a WORK mechanism. If build time grows
# along this ladder, the cost is the array, not the resolution and not the
# decode -- and windowed reads fix it. If it is flat, the residual is somewhere
# else entirely and the locality hypothesis is dead.
#
# Two stages, selected by FSL_STAGES:
#   build  time `build_evaluator` (the ~590 s residual lives inside the driver's
#          BUILD region, and build_evaluator is its per-cell forcing reader)
#   loop   compile the transport RHS with Reactant and time it. This is the
#          question that matters more than the build: does a per-step RHS pay
#          for gathering out of a 7.8 GB buffer instead of a 3.5 MB one?
#
# Env:
#   FSL_GRID    NLONxNLATxNLEV                 (default 6x6x8)
#   FSL_RUNGS   comma list of rung names       (default win,r1,r2,r4,r8)
#               win = the halo window only; rN = N x the 2x2.5 native grid on
#               each axis (r8 == the 0.25x0.3125 shape, 1152x721)
#   FSL_STAGES  build,loop                     (default build)
#   FSL_PARTS   1,2                            (default 1 -- part 1 is the
#               transport half, the only one holding whole-globe arrays)
#   FSL_REPS    timing reps for the loop stage (default 30)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, SHA, Statistics
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
const EA = EarthSciAST
include(joinpath(@__DIR__, "_env.jl"))

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL  = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const NLON, NLAT, NLEV = parse.(Int, split(get(ENV, "FSL_GRID", "6x6x8"), 'x'))
const RUNGS  = String.(split(get(ENV, "FSL_RUNGS", "win,r1,r2,r4,r8"), ','))
const STAGES = Set(String.(split(get(ENV, "FSL_STAGES", "build"), ',')))
const PARTS  = parse.(Int, String.(split(get(ENV, "FSL_PARTS", "1"), ',')))
const REPS   = parse(Int, get(ENV, "FSL_REPS", "30"))
const TSAMP  = 5400.0
const BASERES = "2x2.5"                      # decoded once, for every rung

# Reactant is loaded UNCONDITIONALLY: `RX.@compile` is resolved when `one_rung`
# is macroexpanded (i.e. at definition), so `RX` must already be a Module even
# on a build-only run -- binding it to `nothing` fails there with
# "type Nothing has no field @compile", which is how the first sbatch died.
using Reactant
const RX = Reactant

rss_gb() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^30
gc_s()   = Base.gc_time_ns() / 1e9
faults() = (s = split(read("/proc/self/stat", String)); (min = parse(Int, s[10]),
                                                        maj = parse(Int, s[12])))

# --- the ladder -------------------------------------------------------------
# `win` is the halo-inclusive window and nothing else -- what a windowed READER
# would hand over. `rN` scales the 2x2.5 native grid by N on each axis; r8 is
# 1152x721, the 0.25x0.3125 shape, to within one cell (144*8 = 1152, 91*8 = 728
# vs 721; use the real 0.25 lat extent so the byte count matches exactly).
const NLON0, NLAT0 = 144, 91
function rung_shape(name::AbstractString)
    name == "win" && return (NLON + 2, NLAT + 1)
    name == "r1"  && return (NLON0, NLAT0)
    name == "r8"  && return (1152, 721)          # the real 0.25x0.3125 grid
    m = match(r"^r(\d+)$", name)
    m === nothing && error("unknown rung '$name'")
    n = parse(Int, m.captures[1])
    return (NLON0 * n, NLAT0 * n)
end

# Embed the decoded 2x2.5 halo window into a zeros array of the rung's native
# shape, at an offset that keeps the window at roughly the same FRACTION of the
# globe it occupies at 2x2.5 (so the strides between a gather and its neighbour
# scale the way they really would). Returns the new arrays and the rebased
# (LON0, LAT0).
#
# Layout, verified by decode_cost.jl at every resolution: the loader prepends
# the record bracket, so a surface field is (record, lat, lon) and a 3-D field
# is (record, lev, lat, lon) -- lon LAST. Only the two trailing axes move.
function embed_arrays(param::AbstractDict, slice, (nlon_t, nlat_t))
    lo, la = slice.lon0, slice.lat0
    wlon = lo:(lo + NLON + 1)                 # NLON+2 wide (west + east halo)
    wlat = la:(la + NLAT)                     # NLAT+1 tall (south flank)
    olon = clamp(round(Int, (lo - 1) * nlon_t / NLON0) + 1, 1, nlon_t - (NLON + 1))
    olat = clamp(round(Int, (la - 1) * nlat_t / NLAT0) + 1, 1, nlat_t - NLAT)
    dlon = olon:(olon + NLON + 1)
    dlat = olat:(olat + NLAT)
    out = Dict{String,Any}()
    for (k, v) in param
        a = v::Array{Float64}
        if ndims(a) == 3
            size(a, 3) == NLON0 && size(a, 2) == NLAT0 ||
                error("$k is $(size(a)); expected (rec, $NLAT0, $NLON0)")
            b = zeros(Float64, size(a, 1), nlat_t, nlon_t)
            b[:, dlat, dlon] = a[:, wlat, wlon]
        elseif ndims(a) == 4
            size(a, 4) == NLON0 && size(a, 3) == NLAT0 ||
                error("$k is $(size(a)); expected (rec, lev, $NLAT0, $NLON0)")
            b = zeros(Float64, size(a, 1), size(a, 2), nlat_t, nlon_t)
            b[:, :, dlat, dlon] = a[:, :, wlat, wlon]
        else
            error("$k has $(ndims(a)) dims; expected 3 or 4")
        end
        out[k] = b
    end
    return out, olon, olat
end

function one_rung(rung::AbstractString)
    nlon_t, nlat_t = rung_shape(rung)
    slice = native_slice(res = BASERES, nlon = NLON, nlat = NLAT, nlev = NLEV)
    mp = copy(slice.metaparameters)
    say("\n" * "="^78)
    say(@sprintf("  RUNG %-4s  synthetic native %dx%d   model grid %dx%dx%d   (data: %s)",
                 rung, nlon_t, nlat_t, NLON, NLAT, NLEV, BASERES))
    say("="^78)
    r = Dict{Symbol,Any}(:rung => rung, :nlon_t => nlon_t, :nlat_t => nlat_t)

    Logging.with_logger(Logging.NullLogger()) do
        f0 = reseact_forcing(CHEMDIR; ndays = 1, res = BASERES)
        ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV)))
        merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
        raw_param = Dict{String,Any}()
        t = time()
        for (rawk, prov) in ff.providers
            k = String(rawk)
            fld = EA._provider_const_field(EA.provider_sample(prov, TSAMP), k)
            EA.provider_is_const(prov) ? (merged_const[k] = fld) : (raw_param[k] = fld)
        end
        r[:t_sample] = time() - t

        t = time()
        merged_param, olon, olat = embed_arrays(raw_param, slice, (nlon_t, nlat_t))
        r[:t_embed] = time() - t
        mp["LON0"] = olon; mp["LAT0"] = olat
        mp["GF_NLON"] = nlon_t; mp["GF_NLAT"] = nlat_t
        r[:mb_param] = sum(length(v) * 8 / 1e6 for v in values(merged_param); init = 0.0)
        raw_param = nothing
        GC.gc(true)
        r[:rss_pre] = rss_gb()
        say(@sprintf("  forcing: %8.1f MB resident   LON0=%d LAT0=%d   rss %.1f GB",
                     r[:mb_param], olon, olat, r[:rss_pre]))

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

        fo = Dict{Int,Any}()
        for i in PARTS
            EA._reset_cascade_tally!()
            dm = EA.DiscreteMaterializer()
            f0b = faults(); t = time(); g = gc_s()
            fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
                parameter_overrides = ov, const_arrays = merged_const,
                param_arrays = merged_param, materialize_out = dm)
            r[Symbol("t_build$i")] = time() - t
            r[Symbol("gc_build$i")] = gc_s() - g
            f1b = faults()
            r[Symbol("minflt$i")] = f1b.min - f0b.min
            r[Symbol("majflt$i")] = f1b.maj - f0b.maj
            r[Symbol("nstates$i")] = length(u0i)
            r[Symbol("tally$i")] = copy(EA._CASCADE_TALLY)
            r[Symbol("rss_build$i")] = rss_gb()
            t = time(); dm.materialize!(); r[Symbol("t_mat$i")] = time() - t
            du = fi(u0i, pi, TSAMP)
            r[Symbol("sha$i")] = bytes2hex(sha256(reinterpret(UInt8, vec(du))))
            r[Symbol("finite$i")] = all(isfinite, du)
            fo[i] = (f = fi, u0 = u0i, p = pi)
        end

        # ---- stage: loop ----------------------------------------------------
        # The compiled transport RHS takes every forcing buffer as a real XLA
        # INPUT (`rhs_with_buffers` / `forcing_buffers`), so at r8 the program is
        # handed 7.8 GB of operands to gather ~19k window cells out of. Whether
        # that costs anything PER CALL is the loop question, and it is not
        # answerable from the build.
        if "loop" in STAGES && 1 in PARTS
            fi = fo[1].f
            g4 = EA.rhs_with_buffers(fi)
            hb = EA.forcing_buffers(fi)
            t = time()
            db = map(RX.ConcreteRArray, hb)
            r[:t_upload] = time() - t
            r[:nbuf] = length(hb)
            r[:mb_dev] = sum(length(v) * 8 / 1e6 for v in values(hb); init = 0.0)
            pd = NamedTuple{keys(fo[1].p)}(map(RX.ConcreteRNumber, values(fo[1].p)))
            uR = RX.ConcreteRArray(copy(fo[1].u0)); tR = RX.ConcreteRNumber(TSAMP)
            gT(u, pp, tt, bb) = g4(u, pp, tt, bb)
            # `sync = true` so the timing measures the EXECUTION and not the
            # enqueue, and the driver's XLA:CPU race workaround so the program
            # is the one the production loop runs (see blocker 4).
            copts = RX.CompileOptions(; sync = true,
                xla_debug_options = (; xla_cpu_prefer_vector_width = 128))
            t = time()
            c = RX.@compile compile_options = copts gT(uR, pd, tR, db)
            r[:t_compile] = time() - t
            for _ in 1:3; c(uR, pd, tR, db); end
            ts = Float64[]
            for _ in 1:REPS
                t0 = time(); c(uR, pd, tR, db); push!(ts, time() - t0)
            end
            r[:ms_med] = 1e3 * median(ts); r[:ms_min] = 1e3 * minimum(ts)
            r[:rss_loop] = rss_gb()
            # the per-refresh device push, which also scales with the array
            t = time(); EA.sync_forcing!(db, hb); r[:t_sync] = time() - t
        end
    end
    return r
end

const ROWS = Any[]
for rung in RUNGS
    row = try
        one_rung(rung)
    catch e
        say("  FAILED: " * first(split(sprint(showerror, e), '\n')))
        say("    " * sprint(showerror, e))
        continue
    end
    push!(ROWS, row)
    say(@sprintf("  provider sample  %8.2f s (2x2.5, identical every rung)   embed %.2f s",
                 row[:t_sample], row[:t_embed]))
    say(@sprintf("  doc load+split   %8.2f s", row[:t_doc]))
    for i in PARTS
        say(@sprintf("  build_evaluator%d %8.2f s   (gc %.2f s)   minflt %d  majflt %d   rss %.1f GB",
                     i, row[Symbol("t_build$i")], row[Symbol("gc_build$i")],
                     row[Symbol("minflt$i")], row[Symbol("majflt$i")],
                     row[Symbol("rss_build$i")]))
        tl = sort(collect(row[Symbol("tally$i")]), by = x -> String(x[1]))
        say("        tally: " * join(["$(k)=$(v)" for (k, v) in tl], "  "))
        sha = row[Symbol("sha$i")]
        vs = isempty(ROWS) || ROWS[1] === row ? "   (reference)" :
             ROWS[1][Symbol("sha$i")] == sha ? "   == rung $(ROWS[1][:rung])  RHS IDENTICAL" :
                                               "   != rung $(ROWS[1][:rung])  RHS DIFFERS -- LADDER INVALID"
        say(@sprintf("        rhs sha %s  finite=%s%s   materialize! %.2f s",
                     first(sha, 16), row[Symbol("finite$i")], vs, row[Symbol("t_mat$i")]))
    end
    if haskey(row, :ms_med)
        say(@sprintf("  device buffers   %d bufs  %8.1f MB   upload %.2f s   sync_forcing! %.2f s",
                     row[:nbuf], row[:mb_dev], row[:t_upload], row[:t_sync]))
        say(@sprintf("  @compile gT      %8.2f s", row[:t_compile]))
        say(@sprintf("  gT per call      median %8.3f ms   min %8.3f ms   rss %.1f GB",
                     row[:ms_med], row[:ms_min], row[:rss_loop]))
    end
    GC.gc(true)
end

say("\n" * "="^78)
say("  SUMMARY   model grid $(NLON)x$(NLAT)x$(NLEV); SAME data, SAME gathers,")
say("            only the size of the array being gathered from moves")
say("="^78)
say(@sprintf("  %-5s %11s %10s %9s %9s %9s %10s %10s %9s", "rung", "native",
             "param_MB", "doc", "build1", "build2", "compile", "gT_ms", "sync_s"))
for r in ROWS
    say(@sprintf("  %-5s %5dx%-5d %10.1f %9.2f %9.2f %9.2f %10.2f %10.3f %9.2f",
                 r[:rung], r[:nlon_t], r[:nlat_t], r[:mb_param], r[:t_doc],
                 get(r, :t_build1, NaN), get(r, :t_build2, NaN),
                 get(r, :t_compile, NaN), get(r, :ms_med, NaN), get(r, :t_sync, NaN)))
end
say("FSL_DONE")
