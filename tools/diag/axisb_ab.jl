#!/usr/bin/env julia
# ===========================================================================
# axisb_ab.jl -- A/B the CONTRACTION-LOOP-vs-AFFINE tier ordering
# ===========================================================================
# build_scaling.jl reports `percell_loop=14` for ReSEACT split part 1. Those 14
# are plain column sums (`dp_col`, `pbl_wt`, `pbl_sum_*`). They are NOT declining
# the affine lane: they are never OFFERED to it, because
# `_compile_arrayop_equation!` tests `use_contraction_loop` FIRST and gates the
# whole affine block on `!use_contraction_loop`.
#
# Arms (env is read at build time, so all arms run in ONE process):
#   base    default
#   noloop  ESS_CONTRACTION_LOOP_MIN=1000000 -- the contraction loop can never
#           fire, so the affine tier sees these equations. If they land on
#           `affine`, the ordering is the bug.
#
# Per arm: cascade tally, build wall time, node lowerings (_BENCH_COMPILE_CALLS),
# spine templates, and sha256(du) at the build's own base point.
#
#   AB_GRIDS  comma list of NLONxNLATxNLEV  (default 6x6x8)
#   AB_PARTS  1,2                            (default 1)
#   AB_ARMS   comma list                     (default base,noloop)
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
const GRIDS = String.(split(get(ENV, "AB_GRIDS", "6x6x8"), ','))
const PARTS = parse.(Int, String.(split(get(ENV, "AB_PARTS", "1"), ',')))
const ARMS  = String.(split(get(ENV, "AB_ARMS", "base,noloop"), ','))
const TSAMP = 5400.0

say("EarthSciAST from: " * String(pkgdir(EarthSciAST)))

const F0 = reseact_forcing(CHEMDIR; ndays = 1)

function apply_arm!(arm)
    delete!(ENV, "ESS_CONTRACTION_LOOP_MIN")
    delete!(ENV, "ESS_CONTRACTION_LOOP")
    delete!(ENV, "ESS_STENCIL_DISABLE")
    delete!(ENV, "ESS_LANE_AFFINE_KEY_DISABLE")
    delete!(ENV, "ESS_STATE_BOX_DISABLE")
    if arm == "noloop"
        ENV["ESS_CONTRACTION_LOOP_MIN"] = "1000000"
    elseif arm == "unrollref"
        ENV["ESS_CONTRACTION_LOOP"] = "0"
    elseif arm == "percellref"
        ENV["ESS_STENCIL_DISABLE"] = "1"
    elseif arm == "deltakey"
        # the PRE-FIX signature: state lanes keyed by Delta = slot - output slot
        ENV["ESS_LANE_AFFINE_KEY_DISABLE"] = "1"
    elseif arm == "lanekey"
        # the default (post-fix) signature -- explicit arm name for the ladder
    elseif arm == "statebox_off"
        ENV["ESS_STATE_BOX_DISABLE"] = "1"
    elseif arm == "pre"
        # the WHOLE pre-fix affine tier: Delta-keyed cuts AND the dense per-box
        # state slot tables. This is the baseline every post-fix number is
        # compared against, and the RHS must be bit-identical to it.
        ENV["ESS_LANE_AFFINE_KEY_DISABLE"] = "1"
        ENV["ESS_STATE_BOX_DISABLE"] = "1"
    elseif arm != "base"
        error("unknown arm $arm")
    end
end

const ROWS = Any[]
const LON0 = parse(Int, get(ENV, "AB_LON0", "11"))
const LAT0 = parse(Int, get(ENV, "AB_LAT0", "29"))
for spec in GRIDS
  try
    nlon, nlat, nlev = parse.(Int, split(spec, 'x'))
    slice = native_slice(lon0 = LON0, lat0 = LAT0, nlon = nlon, nlat = nlat, nlev = nlev)
    mp = slice.metaparameters
    say("\n" * "="^78); say(@sprintf("  GRID %s  cells=%d", spec, nlon*nlat*mp["NLEV"])); say("="^78)
    local docs, ov, merged_const, merged_param
    Logging.with_logger(Logging.NullLogger()) do
        file = EA.load_path(MODEL; metaparameters = mp)
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
        docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted)
                for pt in splitparts]
        ff = merge(F0, (; const_arrays = GridResize.slice_hybrid_coefs(F0.const_arrays, mp["NLEV"])))
        merged_const = Dict{String,Any}(String(k) => v for (k,v) in ff.const_arrays)
        merged_param = Dict{String,Any}()
        for (rawk, prov) in ff.providers
            k = String(rawk)
            fld = EA._provider_const_field(EA.provider_sample(prov, TSAMP), k)
            EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
        end
        ov = Dict{String,Float64}(String(k) => Float64(v) for (k,v) in ff.parameters)
        merge!(ov, Dict{String,Float64}(k => Float64(v) for (k,v) in slice.parameters))
    end
    for arm in ARMS
        apply_arm!(arm)
        r = Dict{Symbol,Any}(:spec => spec, :arm => arm, :cells => nlon*nlat*mp["NLEV"])
        Logging.with_logger(Logging.NullLogger()) do
            for i in PARTS
                EA._reset_cascade_tally!()
                EA._bench_reset!(); EA._BENCH_ON[] = true
                dm = EA.DiscreteMaterializer()
                GC.gc(true)
                a0 = Base.gc_num().allocd + Base.gc_num().total_allocd
                t = time()
                fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
                    parameter_overrides = ov, const_arrays = merged_const,
                    param_arrays = merged_param, materialize_out = dm)
                r[Symbol("t$i")] = time() - t
                r[Symbol("alloc$i")] = (Base.gc_num().allocd + Base.gc_num().total_allocd) - a0
                EA._BENCH_ON[] = false
                r[Symbol("nodes$i")] = EA._BENCH_COMPILE_CALLS[]
                r[Symbol("spines$i")] = EA._BENCH_BRANCH_TEMPLATES[]
                r[Symbol("tally$i")] = copy(EA._CASCADE_TALLY)
                r[Symbol("nstates$i")] = length(u0i)
                du = fi(u0i, pi, TSAMP)
                r[Symbol("sha$i")] = bytes2hex(sha256(reinterpret(UInt8, vec(du))))
                r[Symbol("finite$i")] = all(isfinite, du)
                r[Symbol("norm$i")] = sqrt(sum(abs2, du))
            end
        end
        push!(ROWS, r)
        say(@sprintf("\n  arm %-11s grid %s", arm, spec))
        for i in PARTS
            tl = sort(collect(r[Symbol("tally$i")]), by = x -> String(x[1]))
            say(@sprintf("    part%d  build %8.2f s  alloc %9.1f MiB  nodes %10d  spines %8d  nstates %d",
                         i, r[Symbol("t$i")], r[Symbol("alloc$i")]/2^20,
                         r[Symbol("nodes$i")], r[Symbol("spines$i")], r[Symbol("nstates$i")]))
            say("           tally: " * join(["$(k)=$(v)" for (k,v) in tl], "  "))
            say(@sprintf("           sha %s  finite=%s  |du|=%.17g",
                         r[Symbol("sha$i")], r[Symbol("finite$i")], r[Symbol("norm$i")]))
        end
    end
  catch e
    say("  GRID $spec FAILED: " * first(split(sprint(showerror, e), '\n')))
  end
end
apply_arm!("base")

say("\n" * "="^78); say("  SUMMARY"); say("="^78)
for i in PARTS
    say("  part $i")
    ref = Dict{String,String}()
    for r in ROWS
        s = r[Symbol("sha$i")]
        base = get!(ref, r[:spec] * "|" * String(first(ARMS)), s)
        haskey(ref, r[:spec]) || (ref[r[:spec]] = s)
        eq = ref[r[:spec]] == s ? "IDENTICAL" : "*** DIFFERS ***"
        say(@sprintf("    %-10s %-11s build %8.2f s  nodes %10d  %s",
                     r[:spec], r[:arm], r[Symbol("t$i")], r[Symbol("nodes$i")], eq))
    end
end
