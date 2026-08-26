#!/usr/bin/env julia
# ===========================================================================
# jac_source_layout.jl -- does `source_layout` delete the redundant build,
#                         and is the evaluator BIT-IDENTICAL without it?
# ===========================================================================
# `prepare_jacobian` used to run a COMPLETE `build_evaluator` of the SOURCE
# model and throw the RHS away, keeping only `length(u0)`, `p` and `var_map`
# (jac_phase_split.jl measured that build at 45% of the call at 288 cells and
# 51% at 648). EarthSciASTDiff now takes `source_layout = (u0, p, var_map)`
# and skips it.
#
# This runs BOTH forms on the same split part the adjoint driver uses, with the
# driver's own overrides, and (a) times them, (b) compares every observable
# field of the two JacobianEvaluators. The comparison is the point: a layout
# from the wrong build would put every band in the wrong cell with no visible
# symptom, so "same wall clock" is worthless without "same evaluator".
#
#   JSL_GRIDS  comma list of NLONxNLATxNLEV (default 6x6x8,9x9x8)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, SparseArrays
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using EarthSciASTDiff
include(joinpath(REPO, "tools", "reactant_handoff", "rx_sym_block_jac.jl"))
using .RxSymBlockJac
const EA = EarthSciAST
const ED = EarthSciASTDiff
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const GRIDS = String.(split(get(ENV, "JSL_GRIDS", "6x6x8,9x9x8"), ','))
const F0 = reseact_forcing(CHEMDIR; ndays = 1)

fingerprint(j) = (umap = j.umap, scatter = j.scatter, colnames = j.colnames,
                  rownames = j.rownames, structure = j.structure, wrt = j.wrt,
                  nentries = length(j.entries), colptr = j.prototype.colptr,
                  rowval = j.prototype.rowval, nzval = j.prototype.nzval,
                  pattern = j.pattern, nuj = length(j.uj))

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
        docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted)
                for pt in sp]
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

        # The build the DRIVER already has in hand (part 1's layout, which the
        # driver asserts equals part 2's var_map). NOT charged to either timing.
        t0 = time()
        _, u0d, pd, _, vmd = EA.build_evaluator(docs[1]; bkw...)
        _, _, _, _, vmd2 = EA.build_evaluator(docs[2]; bkw...)
        say(@sprintf("  driver-side builds (both parts)   %8.2f s", time() - t0))
        say("  part1 var_map == part2 var_map : $(vmd == vmd2)")

        # Does the layout the driver holds match the one prepare_jacobian would
        # have rebuilt from the RAW FlattenedSystem? (docs[] goes through
        # `index_promoted_refs_by_loop!`; prepare_jacobian does not.)
        t0 = time()
        _, u0r, pr, _, vmr = ED._build_eval(sp[2], nothing; bkw...)
        say(@sprintf("  the REDUNDANT build, standalone   %8.2f s", time() - t0))
        say("  raw var_map == driver var_map  : $(vmr == vmd)   " *
            "length(u0) $(length(u0r)) vs $(length(u0d))   " *
            "keys(p) equal: $(keys(pr) == keys(pd))")

        t0 = time(); jA = ED.prepare_jacobian(sp[2]; wrt = :states, build_kwargs = bkw)
        tA = time() - t0
        t0 = time(); jB = ED.prepare_jacobian(sp[2]; wrt = :states, build_kwargs = bkw,
                                              source_layout = (u0 = u0d, p = pd,
                                                               var_map = vmd))
        tB = time() - t0
        fA = fingerprint(jA); fB = fingerprint(jB)
        for k in keys(fA)
            eq = getproperty(fA, k) == getproperty(fB, k)
            eq || say("  FIELD DIFFERS: $k")
        end
        say("  evaluators identical (all $(length(keys(fA))) fields): $(fA == fB)")

        # ...and they evaluate to the same numbers at a real point.
        J1 = copy(jA.prototype); J2 = copy(jB.prototype)
        u = copy(u0d); jA(J1, u, nothing, 5400.0); jB(J2, u, nothing, 5400.0)
        say(@sprintf("  max |J_rebuild - J_layout| = %.3e   (nnz=%d)",
                     maximum(abs.(J1.nzval .- J2.nzval); init = 0.0), nnz(J1)))

        say(@sprintf("  prepare_jacobian  BEFORE %8.2f s   AFTER %8.2f s",  tA, tB))
        say(@sprintf("  saved %8.2f s = %.1f%% of the call", tA - tB, 100 * (tA - tB) / tA))

        # The driver's own guard, run on both: `block_jac_plan` checks the
        # Jacobian's state ordering against the names the RUNNER compiles, and
        # `validate_plan` reproduces the host Jacobian through the gather plan.
        # Both must pass for the layout form exactly as they do for the rebuild.
        rn = first.(sort(collect(vmd), by = last))
        for (lbl, j) in (("BEFORE", jA), ("AFTER ", jB))
            pl = block_jac_plan(j; runner_names = rn)
            w = validate_plan(pl, j, u0d, pd, 5400.0)
            say(@sprintf("  %s  %s   validate_plan worst relative %.3e  %s",
                         lbl, pl, w, w <= 1e-12 ? "PASS" : "FAIL"))
        end

    end
end

for g in GRIDS
    try
        one_grid(g)
    catch e
        for l in split(sprint(showerror, e), '\n')[1:min(end, 8)]; say("    " * l); end
    end
end
