#!/usr/bin/env julia
# ===========================================================================
# jac_gather_runs.jl -- ARE THE BAND-JACOBIAN GATHERS AFFINE?
# ===========================================================================
# The CONUS ROS23 module carries 86.2 MB of s64 gather-index constants. The
# emitter's own plan layer accounts for only part of it: `plan_index_census.jl`
# puts the CHEMISTRY plan at 0.82 MB per 288 cells (97.5% of it stride, few-run
# or splat, i.e. collapsible), which extrapolates to ~19 MB at CONUS. The rest
# comes from RxSymBlockJac, which is NOT the emitter:
#
#   gather_uj  -> `up[plan.ugather]`, ugather is length NJ (537,264 at CONUS,
#                 4.3 MB) and appears once per use site
#   block_jac  -> `dp[gl[q]]` for each of the 69 used (row,col) block entries,
#                 each a length-NC vector (3.6 MB total at CONUS)
#
# Both are inverted permutations built by `block_jac_plan`, so whether they cost
# anything depends entirely on their RUN STRUCTURE: Reactant lowers a
# constant-stride run to a `stablehlo.slice` (free) and everything else to a
# gather carrying its whole index vector as a constant. `block_jac_plan`'s own
# comment warns the cells "may appear in ANY order, and on ReSEACT they are not
# lexicographic" -- but that is about the state-name ordering, not necessarily
# about these vectors. Measure rather than assume.
#
#   JGR_GRID  NLONxNLATxNLEV (default 6x6x8)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using EarthSciASTDiff
using Reactant
const EA = EarthSciAST
include(joinpath(@__DIR__, "_env.jl"))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
include(joinpath(RXDIR, "rx_sym_block_jac.jl")); using .RxSymBlockJac
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const GRID  = get(ENV, "JGR_GRID", "6x6x8")

function nruns(v::AbstractVector{Int})
    length(v) <= 1 && return 1
    n = 1; d = v[2] - v[1]
    @inbounds for i in 3:length(v)
        di = v[i] - v[i-1]
        di == d || (n += 1; d = di)
    end
    return n
end

nlon, nlat, nlev = parse.(Int, split(GRID, 'x'))
slice = native_slice(lon0 = 11, lat0 = 29, nlon = nlon, nlat = nlat, nlev = nlev)
mp = slice.metaparameters
say("grid=$GRID cells=$(nlon*nlat*mp["NLEV"])")

jacE = var_map = nothing
Logging.with_logger(Logging.NullLogger()) do
    global jacE, var_map
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
    bkw = (; form = :oop, parameter_overrides = ov,
             const_arrays = mc, param_arrays = mparr)
    dm = EA.DiscreteMaterializer()
    _, _, _, _, var_map = EA.build_evaluator(docs[2]; materialize_out = dm, bkw...)
    jacE = EarthSciASTDiff.prepare_jacobian(sp[2]; wrt = :states, build_kwargs = bkw)
end

PLAN = block_jac_plan(jacE; runner_names = first.(sort(collect(var_map), by = last)))
say("  $PLAN")

ug = PLAN.ugather
say(@sprintf("  ugather:  len=%d  runs=%d  (%.1fx compressible)  bytes=%.2f MB",
             length(ug), nruns(ug), length(ug) / nruns(ug), 8 * length(ug) / 1e6))

runs = Int[]; lens = Int[]
for cs in 1:PLAN.NS, rs in 1:PLAN.NS, g in PLAN.blocks[rs, cs]
    push!(runs, nruns(g)); push!(lens, length(g))
end
if isempty(runs)
    say("  no block gathers")
else
    say(@sprintf("  blocks:   n=%d  len=%d each  runs: min=%d median=%d max=%d",
                 length(runs), first(lens), minimum(runs),
                 sort(runs)[cld(end, 2)], maximum(runs)))
    say(@sprintf("            total %.2f MB -> run-length %.2f MB (%.1fx)",
                 8 * sum(lens) / 1e6, 8 * 2 * sum(runs) / 1e6,
                 sum(lens) / max(2 * sum(runs), 1)))
    say("  runs histogram: " *
        join(["$(k)=>$(v)" for (k, v) in sort(collect(
            Dict(r => count(==(r), runs) for r in unique(runs))))][1:min(end, 12)], " "))
end


# ---------------------------------------------------------------------------
# The BAND MODEL's own lane plan -- the last unattributed candidate.
#
# The CONUS ros_step module carries 86.2 MB of s64 gather indices. The
# chemistry source plan accounts for ~19 MB (extrapolated from 0.82 MB at 288
# cells) and RxSymBlockJac's gathers, measured above, are single stride runs
# and so cost nothing. The band model is what is left: it goes through the SAME
# `build_evaluator` and so carries its own `_OopAccPlan`, over NJ states rather
# than N -- 537,264 vs 85,176 at CONUS, 6.3x.
#
# If its index vectors have chemistry's character (median run count 1) the
# run-length proposal captures the whole module. If they have transport's
# (dense permutations) it captures a fifth of it.
# ---------------------------------------------------------------------------
_inner(f) = hasfield(typeof(f), :acc_plans) ? f :
            hasfield(typeof(f), :rhs) ? _inner(getfield(f, :rhs)) :
            error("no acc_plans reachable; fields = $(fieldnames(typeof(f)))")

function plan_census(f0, label)
    f = _inner(f0)
    plans = getfield(f, :acc_plans)
    runs = Int[]; lens = Int[]; bytes = 0
    walk(p) = begin
        p.vectorizable || return
        for g in p.gathers
            isempty(g) && continue
            push!(runs, nruns(g)); push!(lens, length(g)); bytes += 8*length(g)
        end
        for v in p.forc
            isempty(v) && continue
            push!(runs, nruns(v)); push!(lens, length(v)); bytes += 8*length(v)
        end
        for sp in p.sub_plans; walk(sp); end
        for rp in p.red_plan;  walk(rp); end
    end
    for p in plans; walk(p); end
    say("\n  --- $label : $(length(plans)) acc kernels, $(length(runs)) index vectors ---")
    if isempty(runs)
        say("      no index vectors"); return
    end
    sr = sort(runs)
    say(@sprintf("      raw index bytes %.2f MB", bytes/1e6))
    say(@sprintf("      runs: min=%d median=%d max=%d | len: min=%d median=%d max=%d",
                 minimum(runs), sr[cld(end,2)], maximum(runs),
                 minimum(lens), sort(lens)[cld(end,2)], maximum(lens)))
    say(@sprintf("      run-length form is %.1fx smaller  (single-run vectors: %d of %d)",
                 sum(lens)/max(sum(runs),1), count(==(1), runs), length(runs)))
end

plan_census(jacE.fJ!, "BAND MODEL plan (NJ=$(PLAN.NJ) states)")
