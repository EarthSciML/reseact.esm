#!/usr/bin/env julia
# ===========================================================================
# p10_ue_census.jl -- WHY does the transport RHS still assemble the extended
#                     observed buffer `ue` with ESS_OOP_SSA=1 on?
# ===========================================================================
# The 2026-09-05 root cause (b): the CONUS transport step VJP carries 362
# zeroing dynamic-update-slices plus 394 copy-insertion copies of the 2 MB
# extended buffer per call. A refined census of the dumped RHS reverse
# (logs/p7-rhs-13x7x72-10375981) says the `ue`-shaped traffic is ~60% of the
# reverse's element writes, and the 91 `broadcast_dynamic-update-slice` fusions
# (22.35 M elements) run SCALAR.
#
# EarthSciAST already has the machinery to not route stencil reads through
# `ue`: ESS_OOP_SSA redirects a consumer's class-to-class read to the PRODUCER
# VALUE and, when nothing reads a producer's slots off `ue` any more, skips its
# scatter entirely. This probe builds ONLY split part 1 (transport) and prints
# `oop_ssa_stats` plus the per-blocker breakdown, so we know which of the two
# vetoes is holding: the GLOBAL `dynamic` flag (any non-vectorizable acc plan
# anywhere) or per-producer residual `ue` reads.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 6 6 8)
#   P10_PART                 split part (default 1 = transport)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using EarthSciAST, EarthSciIO, JSON3, Printf, Logging
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
const EA = EarthSciAST
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))
get!(ENV, "ESS_OOP_SSA", "1")

const PART  = parse(Int, get(ENV, "P10_PART", "1"))
const RES   = get(ENV, "RESEACT_RES", "4x5")
_envi(k) = haskey(ENV, "RESEACT_$k") ? parse(Int, ENV["RESEACT_$k"]) : nothing
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(res = RES, lon0 = _envi("LON0"), lat0 = _envi("LAT0"),
                           nlon = something(_envi("NLON"), 6), nlat = something(_envi("NLAT"), 6),
                           nlev = _env("NLEV", 8))
const GRID_MP = SLICE.metaparameters
const T0 = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
say("P10 grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$(GRID_MP["NLEV"]) part=$PART ESS_OOP_SSA=$(get(ENV,"ESS_OOP_SSA",""))")

fo = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo
    file = EA.load_path(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    doc = index_promoted_refs_by_loop!(EA.flattened_to_esm(parts[PART]), promoted)
    f0 = reseact_forcing(CHEMDIR; ndays = 1, res = SLICE.res)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, GRID_MP["NLEV"])))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    merged_param = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    dm = EA.DiscreteMaterializer()
    fi, _, _, _, _ = EA.build_evaluator(doc; form = :oop,
        parameter_overrides = ov, const_arrays = merged_const,
        param_arrays = merged_param, materialize_out = dm)
    dm.materialize!()
    fo = fi
end
say(@sprintf("BUILD %.1f s", time() - tb))

st = EA.oop_ssa_stats(fo)
say("P10_SSA " * repr(st))

# ---- the per-blocker breakdown, reaching into the closure ------------------
r = fo.rhs
plans = r.acc_plans
say(@sprintf("P10_ACC   state-RHS acc kernels %d   vectorizable %d   PER-CELL %d",
             length(plans), count(p -> p.vectorizable, plans), count(p -> !p.vectorizable, plans)))
ml = r.mat_levels
say("P10_MAT   levels=$(length(ml))")
tot_kern = 0; tot_pc = 0; tot_scal = 0; tot_scan = 0
for (li, lv) in enumerate(ml)
    global tot_kern, tot_pc, tot_scal, tot_scan
    scalars, kernels, mplans, scans = lv
    npc = count(p -> !p.vectorizable, mplans)
    tot_kern += length(kernels); tot_pc += npc
    tot_scal += length(scalars); tot_scan += length(scans)
    say(@sprintf("  level %d: kernels %3d (per-cell %d)  scalars %4d  scans %3d  out_slots %d",
                 li, length(kernels), npc, length(scalars), length(scans),
                 sum(length(p.out_slots) for p in mplans; init = 0)))
end
say(@sprintf("P10_MATTOT kernels=%d per-cell=%d scalars=%d scans=%d", tot_kern, tot_pc, tot_scal, tot_scan))
say(@sprintf("P10_SIZE  n_states=%d n_total=%d", r.n_states, r.n_total))
# who is marked skip
let sk = 0, npid = 0
    for ks in r.ssa.mat, k in ks
        k.pid != 0 && (npid += 1)
        k.skip && (sk += 1)
    end
    say(@sprintf("P10_SKIP  producers with pid %d   skip=%d", npid, sk))
end
# The per-blocker attribution (ess-oop-ssa ghost extension): which read surface
# is holding each surviving producer scatter alive. Absent on builds from before
# the attribution landed.
if haskey(st, :blockers)
    say("P10_GHOST edges=$(st.n_ghost_edges) fast=$(st.n_ghost_fast) " *
        "elems=$(st.elems_ghost_edges)/$(st.elems_ghost_fast)")
    haskey(st, :n_sub_edges) &&
        say("P10_SUBE  edges=$(st.n_sub_edges) fast=$(st.n_sub_fast) " *
            "elems=$(st.elems_sub_edges)/$(st.elems_sub_fast)")
    say("P10_BLK   " * join(["$k=$(v)" for (k, v) in pairs(st.blockers)], " "))
    say("P10_BLKO  " * join(["$k=$(v)" for (k, v) in pairs(st.blockers_only)], " "))
end
say("P10_DONE")
