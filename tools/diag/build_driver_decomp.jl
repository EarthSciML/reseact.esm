#!/usr/bin/env julia
# ===========================================================================
# build_driver_decomp.jl -- decompose the DRIVER's own BUILD region, faithfully.
# ===========================================================================
# `build_res_window.jl` at BRW_GRID=25x13x72 (slurm 10404098) accounts for only
# 1297.2 s of the 0.25x0.3125 driver's 1932.7 s BUILD, and it settled two
# suspects by measurement: `build_evaluator` is FLAT in the native grid at the
# box325 model grid (159.6 s at 0.25 against 168.2 s at 2x2.5 -- the fine grid
# is if anything FASTER), and `materialize!` is 0.01 s. So the ~590 s residual
# is not per-cell, not locality, and not the materializer.
#
# What is left are the two things the probe does NOT copy from the driver:
#
#   1. `ndays`. The driver builds its forcing with
#      `NDAYS = forcing_days_for(T0, T_END)` -- FOUR days for a 48 h window --
#      where the probe hardcodes `ndays = 1`. A provider is constructed over a
#      4x longer cadence, and every URL it can resolve is a daily file that may
#      not be in the cache yet. At 0.25 one day is 9.4 GB.
#   2. `inspect`. With RESEACT_SHARDS > 0 the driver hands part 2 an
#      `EA.BuildInspection()`; the probe passes `nothing`.
#
# and one thing no probe can copy: WHETHER THE FILES WERE ALREADY THERE. The
# driver run (10395008, started 2026-09-06 20:13:28) shares its cache with
# 10395007, which started 20:12:59 on another node and was in ITS build for the
# next 110 minutes; the nineteen 0.25 blobs in the cache carry mtimes from
# 20:26:12 to 20:48:40, i.e. INSIDE that build window. So the driver's BUILD may
# simply have contained the download of the GEOS-FP files, against a shared
# /projects cache and a concurrent second consumer.
#
# This probe replicates the driver's BUILD region statement for statement, with
# `ndays` and `inspect` as knobs, and reports the cache's blob mtimes before and
# after so a download inside the timed region cannot hide.
#
#   BDD_RES      resolution row              (default 0.25x0.3125)
#   BDD_GRID     NLONxNLATxNLEV              (default 25x13x72 -- box325)
#   BDD_NDAYS    forcing days                (default 4 = the 48 h driver's)
#   BDD_INSPECT  1/0, BuildInspection on part 2 (default 1 = the driver's)
#   BDD_NMACRO   only used to derive the default NDAYS if BDD_NDAYS is unset
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, SHA, Dates
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
const RES   = get(ENV, "BDD_RES", "0.25x0.3125")
const NLON, NLAT, NLEV = parse.(Int, split(get(ENV, "BDD_GRID", "25x13x72"), 'x'))
const NDAYS = parse(Int, get(ENV, "BDD_NDAYS", "4"))
const INSPECT = get(ENV, "BDD_INSPECT", "1") == "1"
# Resident-memory BALLAST, in GB, held live for the whole run. The in-run
# refresh at 0.25 costs 1231.3 s/call (10395008's own timer) against 1018-1027 s
# for the identical decode in this probe -- +21% for the same work. The driver
# differs from this probe in exactly one gross way: it is a 62-88 GB process
# with eight 5.5 GB shard workers co-resident, where this probe is ~16 GB alone
# on the node. Fixed-size transfers in that process are 9.7x-28x slower
# (`T.step.read` 11.89 ms and `T.step.upload` 22.90 ms for the SAME 2.4 MB state
# vector that costs 1.23 / 0.81 ms at 2x2.5), which says the process is paying
# for its own residency. Ballast turns that into a controlled variable: same
# node, same files, same decode, only resident memory moves.
const BALLAST_GB = parse(Float64, get(ENV, "BDD_BALLAST_GB", "0"))
const T0 = 5400.0

rss_gb() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^30
gc_s()   = Base.gc_time_ns() / 1e9

# Cache census: how many blobs, how many bytes, and the newest mtime. Taken
# before and after the timed region; a blob whose mtime lands inside it was
# DOWNLOADED by this process (or by a concurrent consumer of the same cache),
# which is the difference between "the build is slow" and "the build was
# waiting on the network".
const BLOBDIR = joinpath(get(ENV, "EARTHSCIDATADIR",
                             joinpath(homedir(), ".earthsciml")), "v1", "blobs")
function blob_census()
    n = 0; b = 0; newest = 0.0
    isdir(BLOBDIR) || return (n = 0, gb = 0.0, newest = 0.0)
    for (root, _, files) in walkdir(BLOBDIR), f in files
        p = joinpath(root, f)
        st = try stat(p) catch; continue end
        n += 1; b += st.size; newest = max(newest, st.mtime)
    end
    return (n = n, gb = b / 2^30, newest = newest)
end
fmt_t(x) = x == 0 ? "-" : Dates.format(Dates.unix2datetime(x), "yyyy-mm-ddTHH:MM:SS")

const SLICE = native_slice(res = RES, nlon = NLON, nlat = NLAT, nlev = NLEV)
const MP = SLICE.metaparameters
say("="^78)
say(@sprintf("  build_driver_decomp  res=%s grid=%dx%dx%d native %dx%d  ndays=%d inspect=%s",
             RES, NLON, NLAT, NLEV, SLICE.grid.nlon, SLICE.grid.nlat, NDAYS, INSPECT))
say("="^78)
let c = blob_census()
    say(@sprintf("  cache BEFORE: %d blobs  %.1f GB  newest %s", c.n, c.gb, fmt_t(c.newest)))
end
const T_START = time()
# Touched (not just allocated) so the pages are really resident, and kept in a
# const binding so the GC cannot reclaim it.
const BALLAST = if BALLAST_GB > 0
    n = round(Int, BALLAST_GB * 2^30 / 8)
    t = time(); v = Vector{Float64}(undef, n); fill!(v, 1.0)
    say(@sprintf("  ballast: %.1f GB resident (%.1f s to touch)   rss %.1f GB",
                 BALLAST_GB, time() - t, rss_gb()))
    v
else
    Float64[]
end

merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global merged_param, discrete
    t = time()
    file = EA.load_path(MODEL; metaparameters = MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in splitparts]
    say(@sprintf("  doc load+split   %8.2f s", time() - t))

    t = time()
    f0 = reseact_forcing(CHEMDIR; ndays = NDAYS, res = RES)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV)))
    say(@sprintf("  forcing ctor     %8.2f s   (ndays=%d)", time() - t, NDAYS))

    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    tsamp = time(); gsamp = gc_s()
    for (rawk, prov) in sort(collect(ff.providers), by = first)
        k = String(rawk)
        t = time(); s = EA.provider_sample(prov, T0); dt = time() - t
        fld = EA._provider_const_field(s, k)
        mb = fld isa AbstractArray ? length(fld) * 8 / 1e6 : 0.0
        say(@sprintf("      sample %-22s %8.2f s   %9.1f MB  %s", k, dt, mb,
                     EA.provider_is_const(prov) ? "(const)" : ""))
        EA.provider_is_const(prov) ? (merged_const[k] = fld) :
            (merged_param[k] = fld; discrete[k] = prov)
    end
    say(@sprintf("  provider sample  %8.2f s   (gc %.2f s)  %d providers  %.1f MB kept  rss %.1f GB",
                 time() - tsamp, gc_s() - gsamp, length(ff.providers),
                 sum(length(v) * 8 / 1e6 for v in values(merged_param); init = 0.0), rss_gb()))

    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))

    binsp = INSPECT ? EA.BuildInspection() : nothing
    dms = Any[]
    for i in 1:2
        dm = EA.DiscreteMaterializer(); push!(dms, dm)
        t = time(); g = gc_s()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dm,
            inspect = (INSPECT && i == 2) ? binsp : nothing)
        say(@sprintf("  build_evaluator%d %8.2f s   (gc %.2f s)   nstates=%d   rss %.1f GB",
                     i, time() - t, gc_s() - g, length(u0i), rss_gb()))
    end
    t = time(); foreach(d -> d.materialize!(), dms)
    say(@sprintf("  materialize!     %8.2f s   rss %.1f GB", time() - t, rss_gb()))
end
say(@sprintf("BUILD %.2f s   (the driver's own line; discrete_providers=%d)",
             time() - tb, length(discrete)))
let c = blob_census()
    say(@sprintf("  cache AFTER : %d blobs  %.1f GB  newest %s", c.n, c.gb, fmt_t(c.newest)))
    say(c.newest > T_START ?
        "  *** A BLOB WAS WRITTEN DURING THIS RUN -- the build contained a DOWNLOAD ***" :
        "  no blob written during this run -- the cache was warm, this is decode only")
end
say("BDD_DONE")
