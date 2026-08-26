#!/usr/bin/env julia
# ===========================================================================
# jac_phase_split.jl -- WHERE DO prepare_jacobian's 592 s GO?
# ===========================================================================
# `prepare_jacobian` is the single largest setup item in the CONUS adjoint
# (592 s at 13x7x72, 1316 s at 26x14x72 -- 2.22x for 4x the cells) and its body
# (EarthSciASTDiff/src/assemble.jl:146) is five separable phases:
#
#   sysview + jacobian_bands   symbolic differentiation, BAND-level, so it is a
#                              fact of the document and should be grid-independent
#   _evaluation_document       lower the bands into a runnable ESM document
#   build_evaluator(doc)       build the BAND model                 <- a full build
#   _build_eval(src)           build the SOURCE model               <- a full build
#                              ...used ONLY for `vm0`, `p0`, `length(u00)`, all
#                              three of which the caller already has in hand
#   _scatter_pairs + scatter   enumerate every (band, cell) pair: three
#                              interpolated String cell names each, plus an
#                              `_eval_cidx` that rebuilds an expression tree
#                              per cell per column index. O(cells x bands).
#
# This runs the same five phases with a timer on each, at two grids, so the
# split is measured rather than argued. If `_build_eval(src)` is what it looks
# like, it is a redundant build that an optional kwarg deletes outright.
#
#   JPS_GRIDS  comma list of NLONxNLATxNLEV (default 6x6x8,9x9x8)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, SparseArrays
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using EarthSciASTDiff
const EA = EarthSciAST
const ED = EarthSciASTDiff
include(joinpath(@__DIR__, "_env.jl"))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const GRIDS = String.(split(get(ENV, "JPS_GRIDS", "6x6x8,9x9x8"), ','))
const F0 = reseact_forcing(CHEMDIR; ndays = 1)

tick(lbl, f) = (t = time(); r = f(); dt = time() - t;
                say(@sprintf("      %-26s %8.2f s", lbl, dt)); (r, dt))

function one_grid(spec)
    nlon, nlat, nlev = parse.(Int, split(spec, 'x'))
    slice = native_slice(lon0 = 11, lat0 = 29, nlon = nlon, nlat = nlat, nlev = nlev)
    mp = slice.metaparameters
    say("\n" * "="^78)
    say(@sprintf("  GRID %s   cells=%d", spec, nlon * nlat * mp["NLEV"]))
    say("="^78)
    times = Dict{String,Float64}()
    Logging.with_logger(Logging.NullLogger()) do
        file = EA.load_path(MODEL; metaparameters = mp)
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        sp = split_system(flat, stencil_following_rule(flat); nparts = 2)
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

        src = sp[2]
        (svsrc, t1) = tick("sysview", () -> ED._sv_src(src, nothing))
        sv, srcd = svsrc
        (entries, t2) = tick("jacobian_bands", () -> ED.jacobian_bands(sv; wrt = :states))
        (docp, t3) = tick("_evaluation_document",
                          () -> ED._evaluation_document(sv, entries; cse = true,
                                                        cse_min_nodes = 12))
        doc, _ = docp
        (bandb, t4) = tick("build_evaluator(BAND doc)",
                           () -> EA.build_evaluator(doc; model_name = "JacobianEval", bkw...))
        fJ!, u0j, pj, _, vmj = bandb
        (srcb, t5) = tick("_build_eval(SOURCE)  <-- ?",
                          () -> ED._build_eval(srcd, nothing; bkw...))
        _, u00, p0, _, vm0 = srcb
        (pairs, t6) = tick("_scatter_pairs", () -> ED._scatter_pairs(sv, entries))
        (_, t7) = tick("scatter map (dicts+sort)", function ()
            colslot, ncol, _ = ED._colmap(vm0, p0, :states)
            n = length(u00)
            items = Tuple{Int,Int,Int}[]
            for (k, rowname, colname, jname) in pairs
                cs = colslot(colname); cs == 0 && continue
                push!(items, (vmj[jname], vm0[rowname], cs))
            end
            I = [it[2] for it in items]; J = [it[3] for it in items]
            pattern = sparse(I, J, trues(length(I)), n, ncol, |)
            proto = SparseMatrixCSC(n, ncol, copy(pattern.colptr), copy(pattern.rowval),
                                    zeros(length(pattern.rowval)))
            sc = Tuple{Int,Int}[]
            for (slot, r, c) in items
                rng = proto.colptr[c]:(proto.colptr[c+1]-1)
                push!(sc, (slot, rng[searchsortedfirst(view(proto.rowval, rng), r)]))
            end
            length(sc)
        end)
        say(@sprintf("      %-26s %8.2f s   (nbands=%d npairs=%d nstates=%d)",
                     "TOTAL", t1+t2+t3+t4+t5+t6+t7, length(entries), length(pairs),
                     length(u00)))
        say(@sprintf("      redundant SOURCE build = %.1f%% of the total",
                     100 * t5 / (t1+t2+t3+t4+t5+t6+t7)))
        times["src_build"] = t5
        times["total"] = t1+t2+t3+t4+t5+t6+t7
    end
    return times
end

for g in GRIDS
    try
        one_grid(g)
    catch e
        say("  FAILED $g: " * first(split(sprint(showerror, e), '\n')))
        for l in split(sprint(showerror, e), '\n')[1:min(end,5)]; say("    " * l); end
    end
end
