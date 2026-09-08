#!/usr/bin/env julia
# ===========================================================================
# build_prof_delta.jl -- WHERE DO THE PER-CELL BUILD SECONDS GO?
# ===========================================================================
# BUILD_SCALING.md B3 leaves ~0.0058 s/cell in `build_evaluator` unattributed:
# node lowerings are now flat in the grid (PR #273) while build TIME is not, so
# the cost is not proportional to IR volume. A sampling profile at TWO grids,
# differenced function by function, names it directly: whatever is O(cells)
# shows up as a self-time DELTA proportional to the cell ratio; whatever is
# O(1) cancels.
#
#   BPD_GRIDS  comma list NLONxNLATxNLEV   (default 6x6x8,18x12x8)
#   BPD_PART   1 | 2                        (default 1)
#   BPD_TOP    rows to print                (default 45)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, Profile, SHA
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
const EA = EarthSciAST
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const GRIDS = String.(split(get(ENV, "BPD_GRIDS", "6x6x8,18x12x8"), ','))
const PART  = parse(Int, get(ENV, "BPD_PART", "1"))
const TOP   = parse(Int, get(ENV, "BPD_TOP", "45"))
const TSAMP = 5400.0

say("EarthSciAST from: " * String(pkgdir(EarthSciAST)))
const F0 = Logging.with_logger(Logging.NullLogger()) do
    reseact_forcing(CHEMDIR; ndays = 1)
end

function setup(spec)
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
    ff = merge(F0, (; const_arrays = GridResize.slice_hybrid_coefs(F0.const_arrays, mp["NLEV"])))
    mc = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    mparr = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, TSAMP), k)
        EA.provider_is_const(prov) ? (mc[k] = fld) : (mparr[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters))
    return docs, mc, mparr, ov, nlon * nlat * mp["NLEV"]
end

# ---- flat self-time aggregation -------------------------------------------
function flat_self(delay)
    data, lidict = Profile.retrieve()
    self = Dict{String,Int}()
    total = Dict{String,Int}()
    i = 1
    n = length(data)
    while i <= n
        j = i
        while j <= n && data[j] != 0
            j += 1
        end
        bt = @view data[i:j-1]
        # metadata block trails the backtrace in >=1.8: skip leading meta ips
        frames = Any[]
        for ip in bt
            fs = get(lidict, ip, nothing)
            fs === nothing && continue
            for f in fs
                f.from_c && continue
                push!(frames, f)
            end
        end
        if !isempty(frames)
            key(f) = string(f.func) * " @ " * string(basename(String(f.file))) * ":" * string(f.line)
            self[key(frames[1])] = get(self, key(frames[1]), 0) + 1
            seen = Set{String}()
            for f in frames
                k = key(f)
                k in seen && continue
                push!(seen, k)
                total[k] = get(total, k, 0) + 1
            end
        end
        i = j + 1
    end
    return Dict(k => v * delay for (k, v) in self), Dict(k => v * delay for (k, v) in total)
end

const DELAY = 0.002
Profile.init(n = 60_000_000, delay = DELAY)

results = Any[]
for spec in GRIDS
    docs, mc, mparr, ov, cells = Logging.with_logger(Logging.NullLogger()) do
        setup(spec)
    end
    GC.gc(true)
    Profile.clear()
    EA._reset_cascade_tally!()
    EA._bench_reset!(); EA._BENCH_ON[] = true
    t = time(); a0 = Base.gc_live_bytes()
    r = Logging.with_logger(Logging.NullLogger()) do
        dm = EA.DiscreteMaterializer()
        @profile EA.build_evaluator(docs[PART]; form = :oop, parameter_overrides = ov,
            const_arrays = mc, param_arrays = mparr, materialize_out = dm)
    end
    el = time() - t
    EA._BENCH_ON[] = false
    fi, u0i, pi, _, vmi = r
    du = fi(u0i, pi, TSAMP)
    sha = bytes2hex(sha256(reinterpret(UInt8, vec(du))))
    self, tot = flat_self(DELAY)
    nsamp = sum(values(self))
    push!(results, (; spec, cells, el, self, tot, sha,
                    nodes = EA._BENCH_COMPILE_CALLS[], nstates = length(u0i),
                    rss = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^30))
    say(@sprintf("grid %-10s cells %7d  build %8.2f s  sampled %8.2f s  nodes %9d  nstates %8d  rss %.1f GB  sha %s",
                 spec, cells, el, nsamp, EA._BENCH_COMPILE_CALLS[], length(u0i),
                 results[end].rss, first(sha, 16)))
    docs = nothing; r = nothing; fi = nothing
    GC.gc(true)
end

if length(results) >= 2
    A, B = results[1], results[end]
    say("\n" * "="^110)
    say(@sprintf("  SELF-TIME DELTA  %s (%d cells) -> %s (%d cells)   ratio %.2fx",
                 A.spec, A.cells, B.spec, B.cells, B.cells / A.cells))
    say("="^110)
    keys_all = union(keys(A.self), keys(B.self))
    rows = [(k, get(A.self, k, 0.0), get(B.self, k, 0.0)) for k in keys_all]
    sort!(rows, by = r -> -(r[3] - r[2]))
    say(@sprintf("  %10s %10s %10s  %s", "A_self_s", "B_self_s", "delta_s", "function @ file:line"))
    for (k, a, b) in first(rows, TOP)
        say(@sprintf("  %10.2f %10.2f %10.2f  %s", a, b, b - a, k))
    end
    say("\n  --- CUMULATIVE (any frame on stack) delta, top $(TOP) ---")
    keys_all = union(keys(A.tot), keys(B.tot))
    rows = [(k, get(A.tot, k, 0.0), get(B.tot, k, 0.0)) for k in keys_all]
    sort!(rows, by = r -> -(r[3] - r[2]))
    say(@sprintf("  %10s %10s %10s  %s", "A_tot_s", "B_tot_s", "delta_s", "function @ file:line"))
    for (k, a, b) in first(rows, TOP)
        say(@sprintf("  %10.2f %10.2f %10.2f  %s", a, b, b - a, k))
    end
end
