#!/usr/bin/env julia
# ===========================================================================
# plan_index_census.jl -- HOW MUCH OF THE COMPILED MODULE IS INDEX DATA?
# ===========================================================================
# One CONUS ROS23 step (logs/hlodump-10105946, f64[85176] entry) compiles to
# 356.7 MB of buffer, of which 126.4 MB is CONSTANT -- and 86.2 MB of THAT is
# s64: pure gather/scatter index vectors, 249 allocations. Addresses, not
# physics. Every byte scales linearly with the cell count, and XLA hashes,
# folds, buffer-assigns and serialises all of it on every compile.
#
# The index vectors come from `_build_oop_desc_vectors` (tree_walk/oop.jl),
# which materialises ONE length-L host vector per access descriptor:
#     _AK_STATE_AFFINE   -> out .+ delta                (AFFINE in the lane)
#     _AK_STATE_TBL_BOX  -> conn[box-addressed]         (arbitrary)
#     _AK_FORCING_BOX    -> off + strided midx          (AFFINE, usually)
#     _AK_ARR_TBL_BOX    -> conn[box-addressed]         (arbitrary)
#     _AK_CONST_*        -> a length-L Float64 lane vector
# Reactant's `getindex_linear` already turns a CONSTANT-STRIDE run into a
# `stablehlo.slice` (O(1) in the module) and everything else into a
# `stablehlo.gather` with an O(L) s64 constant operand. So the question that
# decides whether an O(1) module is reachable is: WHAT FRACTION OF THE INDEX
# DATA IS ALREADY AFFINE, and what does the residue look like?
#
# This reflects on the built `:oop` closure's documented `:acc_plans` field
# (oop.jl pins those field names for external tooling) and classifies every
# per-lane vector:
#     stride     -- constant-stride run; Reactant emits a slice, costs nothing
#     affine-ish -- piecewise stride (a few runs); expressible as a few slices
#     arbitrary  -- a real permutation; needs an O(L) s64 constant
#     splat      -- every lane equal; ONE broadcast would do
# The structure is a fact of the DOCUMENT, so a small grid answers it.
#
#   PIC_GRIDS  comma list of NLONxNLATxNLEV (default 6x6x8,9x9x8)
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
const GRIDS = String.(split(get(ENV, "PIC_GRIDS", "6x6x8,9x9x8"), ','))
const F0 = reseact_forcing(CHEMDIR; ndays = 1)

# How many maximal constant-stride runs does `v` decompose into? 1 => Reactant
# emits a single slice. `n` runs => n slices + a concat, still O(1) in L when n
# is small and independent of L.
function nruns(v::Vector{Int})
    length(v) <= 1 && return 1
    n = 1; d = v[2] - v[1]
    @inbounds for i in 3:length(v)
        di = v[i] - v[i-1]
        di == d || (n += 1; d = di)
    end
    return n
end
allequal_(v) = isempty(v) || all(==(first(v)), v)

mutable struct Tally
    n::Int; bytes::Int
end
Tally() = Tally(0, 0)
bump!(t::Tally, L, w) = (t.n += 1; t.bytes += L * w; nothing)

function census_plan!(acc, plan, K)
    plan.vectorizable || return
    for (i, g) in enumerate(plan.gathers)
        isempty(g) && continue
        L = length(g); r = nruns(g)
        k = r == 1 ? :idx_stride : r <= 8 ? :idx_fewruns : :idx_arbitrary
        bump!(acc[k], L, 8)
        # UNCAPPED. The decisive question is not "is it affine" but "does the RUN
        # COUNT grow with the grid": a vector with r runs lowers to r slices +
        # a concat, so r independent of L is O(1) in the grid no matter how far
        # from a single stride it is. Record (r, L) pairs so runs-per-lane can be
        # compared across grid sizes.
        push!(acc[:idx_rl], (r, L))
    end
    for f in plan.forc
        isempty(f) && continue
        L = length(f); r = nruns(f)
        bump!(acc[r == 1 ? :forc_stride : r <= 8 ? :forc_fewruns : :forc_arbitrary], L, 8)
    end
    for c in plan.consts
        isempty(c) && continue
        L = length(c)
        bump!(acc[allequal_(c) ? :const_splat :
                  length(unique(c)) * 8 <= L ? :const_lowcard : :const_field], L, 8)
    end
    for g in plan.ghost
        isempty(g) && continue
        bump!(acc[allequal_(g) ? :ghost_splat : :ghost_field], length(g), 1)
    end
    L = length(plan.out_slots)
    L == 0 || bump!(acc[nruns(plan.out_slots) == 1 ? :out_stride : :out_arbitrary], L, 8)
    for sp in plan.sub_plans; census_plan!(acc, sp, K); end
    for rp in plan.red_plan;  census_plan!(acc, rp, K); end
end

const KEYS = [:idx_stride, :idx_fewruns, :idx_arbitrary, :forc_stride, :forc_fewruns,
              :forc_arbitrary, :const_splat, :const_lowcard, :const_field,
              :ghost_splat, :ghost_field, :out_stride, :out_arbitrary]

# `build_evaluator(form=:oop)` returns an `_OopRHS` wrapper (fields rhs /
# buffers / buffer_index); the walker closure carrying the documented
# `:acc_plans` / `:acc_kernels` fields is the inner `.rhs`.
_inner(f) = hasfield(typeof(f), :acc_plans) ? f :
            hasfield(typeof(f), :rhs) ? _inner(getfield(f, :rhs)) :
            error("no acc_plans reachable; fields = $(fieldnames(typeof(f)))")

function census(f0, label)
    f = _inner(f0)
    plans = getfield(f, :acc_plans)
    kerns = getfield(f, :acc_kernels)
    acc = Dict{Symbol,Any}(k => Tally() for k in KEYS)
    acc[:idx_runs_hist] = Dict{Int,Int}()
    acc[:idx_rl] = Tuple{Int,Int}[]
    for (p, K) in zip(plans, kerns); census_plan!(acc, p, K); end
    say("\n  --- $label : $(length(plans)) acc kernels, " *
        "$(count(p -> p.vectorizable, plans)) vectorized ---")
    tot = sum(acc[k].bytes for k in KEYS)
    for k in KEYS
        t = acc[k]
        t.n == 0 && continue
        say(@sprintf("      %-16s %6d vecs  %9.2f MB  (%4.1f%%)", k, t.n,
                     t.bytes / 1e6, 100 * t.bytes / max(tot, 1)))
    end
    say(@sprintf("      %-16s %6s      %9.2f MB", "TOTAL", "", tot / 1e6))
    bakeable = acc[:idx_arbitrary].bytes + acc[:forc_arbitrary].bytes +
               acc[:const_field].bytes + acc[:out_arbitrary].bytes
    say(@sprintf("      MUST be O(L) constants: %.2f MB   collapsible to O(1): %.2f MB",
                 bakeable / 1e6, (tot - bakeable) / 1e6))
    rl = acc[:idx_rl]
    if !isempty(rl)
        rs = [r for (r, _) in rl]; ls = [l for (_, l) in rl]
        srt = sort(rs)
        say(@sprintf("      index RUNS: n=%d min=%d median=%d max=%d | lane len: min=%d median=%d max=%d",
                     length(rs), minimum(rs), srt[cld(end, 2)], maximum(rs),
                     minimum(ls), sort(ls)[cld(end, 2)], maximum(ls)))
        say(@sprintf("      runs/lane median=%.4f -> run-length form is %.1fx smaller overall",
                     sort([r / l for (r, l) in rl])[cld(end, 2)],
                     sum(ls) / max(sum(rs), 1)))
        # The run count of the WORST offenders, which set the floor.
        top = sort(rl, by = x -> -x[1])[1:min(end, 8)]
        say("      worst (runs/len): " * join(["$(r)/$(l)" for (r, l) in top], " "))
    end
    return tot
end

function one_grid(spec)
    nlon, nlat, nlev = parse.(Int, split(spec, 'x'))
    slice = native_slice(lon0 = 11, lat0 = 29, nlon = nlon, nlat = nlat, nlev = nlev)
    mp = slice.metaparameters
    say("\n" * "="^78)
    say(@sprintf("  GRID %s   cells=%d", spec, nlon * nlat * mp["NLEV"]))
    say("="^78)
    Logging.with_logger(Logging.NullLogger()) do
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
        for i in 1:2
            dm = EA.DiscreteMaterializer()
            t = time()
            fi, = EA.build_evaluator(docs[i]; form = :oop, parameter_overrides = ov,
                const_arrays = mc, param_arrays = mparr, materialize_out = dm)
            bt = time() - t
            census(fi, "part $i (build $(round(bt, digits=1)) s)")
        end
    end
end

for g in GRIDS
    try
        one_grid(g)
    catch e
        say("  FAILED $g: " * first(split(sprint(showerror, e), '\n')))
    end
end
