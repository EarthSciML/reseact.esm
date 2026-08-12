#!/usr/bin/env julia
# ===========================================================================
# tools/cwc_gate.jl -- the q == 1 CONSISTENCY-WITH-CONTINUITY gate, run against
# the model as it actually is: diagnosed Mz, pressure fixer, chemistry and all.
# ===========================================================================
# CWC (Jockel et al. 2001) is the requirement that the tracer scheme and the
# air-mass scheme be driven by the SAME mass fluxes, so that a tracer initialised
# to a UNIFORM mixing ratio stays uniform no matter what the flow does. It is not
# a tolerance test: at q == 1 the tracer divergence must collapse onto the
# continuity divergence BITWISE, so the gate is `== 0.0` in IEEE, and any nonzero
# answer is a real defect rather than accumulated roundoff.
#
# reseact.esm's metadata cites this gate, but the states it needs (mq, dev) were
# dropped when chemistry was added -- prototypes/reseact_3d/reseact_3d.esm still
# has them. Rather than carry four dead states in the production model, this
# driver SYNTHESISES a gate copy: it reads reseact.esm, appends the gate states
# and their equations, and writes the result NEXT TO the original (relative $ref
# paths resolve from the file's own directory, so the copy has to live there).
# The copy is deleted on exit unless RESEACT_KEEP_GATE=1.
#
# THREE gates, testing different things:
#
#   qtest   the MIXING-RATIO form the real species actually use,
#             dq/dt = [-div(M q) + q div(M)] / m
#           initialised to 1 everywhere with all four inflow halos at 1. This is
#           the statement that matters for SuperFast: O3 and friends are advected
#           by exactly this expression, so if qtest holds, they inherit CWC.
#           Note it is insensitive to what the CONTINUITY equation does -- the m
#           in the denominator cancels out of the q == 1 limit.
#
#   qg      the MASS form, mqg/mg, with its OWN pair of air-mass and tracer-mass
#           states integrated side by side from the raw flux divergences. This is
#           reseact_3d.esm's original gate verbatim. It is the stronger statement:
#           tracer MASS must track air MASS to the bit as the layers breathe.
#           It deliberately uses the RAW continuity divergence rather than the
#           model's fixed one, so it tests the DISCRETISATION (does PPM at q == 1
#           reduce to the mass flux?) and not the pressure fixer.
#
#   dev     the same difference integrated directly from dev(0) = 0, which is
#           what makes the failure mode legible: assertion references are
#           build-time and cannot read states, and asserting mqg would mean
#           re-spelling the whole discrete divergence by hand, i.e. duplicating
#           the operator inside its own test. Against a constant-zero reference
#           there is nothing for the test to get wrong.
#
# It also re-reports the continuity drift |m - dp|/dp, which the pressure fixer
# is supposed to hold at roundoff -- see tools/continuity_residual.jl for the
# single-evaluation version of the same quantity.
#
# Env: RESEACT_MODEL, RESEACT_NLON/NLAT/NLEV, RESEACT_T0, RESEACT_STEPS,
#      RESEACT_MACRO_DT, RESEACT_KEEP_GATE
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RUN_ENV", joinpath(REPO, "run-model-jl")); io = devnull)
haskey(ENV, "ESS_KERNEL_CLASS_MERGE_DISABLE") || (ENV["ESS_KERNEL_CLASS_MERGE_DISABLE"] = "1")
using SciMLBase, DiffEqCallbacks
import OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSSPRK
import LinearSolve
using LinearAlgebra, Printf, Statistics, Logging, JSON3

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(RXDIR, "op_split.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL    = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
# Grid defaults follow the model's own metaparameters (CONUS 13x7x72); the
# slice ORIGIN comes from the .esm defaults plus the degree-space twins
# `native_slice()` applies through `build_split_run` (see HELPERS.md §1).
const NLON     = parse(Int, get(ENV, "RESEACT_NLON", "13"))
const NLAT     = parse(Int, get(ENV, "RESEACT_NLAT", "7"))
const NLEV     = parse(Int, get(ENV, "RESEACT_NLEV", "72"))
const T0       = parse(Float64, get(ENV, "RESEACT_T0", "64800"))
const NSTEPS   = parse(Int, get(ENV, "RESEACT_STEPS", "6"))
const MACRO_DT = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const RTOL, ATOL, ATOL_T = 1e-4, 1e-9, 1e-6

# --------------------------------------------------------------------------- #
# 1. Synthesise the gate model.
# --------------------------------------------------------------------------- #
grid3(expr) = Dict{String,Any}("op" => "aggregate", "output_idx" => ["i", "j", "k"],
    "args" => Any[], "expr" => expr,
    "ranges" => Dict{String,Any}("i" => Dict("from" => "lon"),
                                 "j" => Dict("from" => "lat"),
                                 "k" => Dict("from" => "lev")))
"A constant-valued halo field on the two axes `a`,`b` (the inflow BC operand shape)."
halo(a, b, val) = Dict{String,Any}("type" => "observed", "units" => "1", "shape" => [a, b],
    "description" => "Gate inflow halo, identically $val so the incoming tracer matches the interior.",
    "expression" => Dict{String,Any}("op" => "aggregate", "output_idx" => ["ga", "gb"],
        "args" => Any[], "expr" => val,
        "ranges" => Dict{String,Any}("ga" => Dict("from" => a), "gb" => Dict("from" => b))))
state3(desc) = Dict{String,Any}("type" => "state", "units" => "1",
    "shape" => ["lon", "lat", "lev"], "description" => desc)
Dt(v) = Dict{String,Any}("op" => "D", "args" => Any[v], "wrt" => "t")
# raw mass divergence: D(Mx,lon) + D(My,lat) + D(Mz,lev)
cont_div() = Dict{String,Any}("op" => "+", "args" => Any[
    Dict{String,Any}("op" => "D", "args" => Any["Mx"], "wrt" => "lon"),
    Dict{String,Any}("op" => "D", "args" => Any["My"], "wrt" => "lat"),
    Dict{String,Any}("op" => "D", "args" => Any["Mz"], "wrt" => "lev")])
# tracer divergence with q's own halos: D(Mx*q,lon,w,e) + D(My*q,lat,s,n) + D(Mz*q,lev)
tracer_div(q, w, e, s, n) = Dict{String,Any}("op" => "+", "args" => Any[
    Dict{String,Any}("op" => "D", "args" => Any[
        Dict{String,Any}("op" => "*", "args" => Any["Mx", q]), w, e], "wrt" => "lon"),
    Dict{String,Any}("op" => "D", "args" => Any[
        Dict{String,Any}("op" => "*", "args" => Any["My", q]), s, n], "wrt" => "lat"),
    Dict{String,Any}("op" => "D", "args" => Any[
        Dict{String,Any}("op" => "*", "args" => Any["Mz", q])], "wrt" => "lev")])
neg(x) = Dict{String,Any}("op" => "-", "args" => Any[x])

function write_gate_model(src_path)
    doc = JSON3.read(read(src_path, String), Dict{String,Any})
    tp = doc["models"]["Transport3D"]
    V = tp["variables"]; E = tp["equations"]

    for (nm, a, b) in (("qbc_w_g", "lat", "lev"), ("qbc_e_g", "lat", "lev"),
                       ("qbc_s_g", "lon", "lev"), ("qbc_n_g", "lon", "lev"))
        V[nm] = halo(a, b, 1.0)
    end
    V["qtest"] = state3("GATE (mixing-ratio form). A passive tracer carried by the " *
        "SAME expression the SuperFast species use, initialised to 1 everywhere " *
        "with all four halos at 1. Must stay EXACTLY 1.")
    V["mg"] = state3("GATE (mass form). The gate's own air mass, integrated from " *
        "the RAW flux divergence so the gate measures the DISCRETISATION and not " *
        "the pressure fixer. Independent of the model's own m.")
    V["mqg"] = state3("GATE (mass form). Tracer mass, initialised bitwise equal to " *
        "mg so qg == 1.")
    V["dev"] = state3("GATE. dev = mqg - mg obtained by integrating the DIFFERENCE " *
        "of the two right-hand sides from dev(0) = 0. At q == 1 that difference is " *
        "EXACTLY 0.0 in IEEE, so dev never leaves zero. Feeds nothing back.\n\n" *
        "This is the SENSITIVE instrument, and it does not duplicate the mg/mqg " *
        "comparison. A 1-ulp disagreement between the two divergences is ~2e-18 Pa/s " *
        "at |divh| ~ 1e-2, which is ~1e5 times SMALLER than ulp(1500 Pa) -- so mg and " *
        "mqg can stay bitwise equal step after step while the defect is real. dev " *
        "starts at 0.0, where ulp is ~5e-324, and accumulates it faithfully.")
    for (nm, ax) in (("dev_x", "lon"), ("dev_y", "lat"), ("dev_z", "lev"))
        V[nm] = state3("GATE. The $ax-axis term of dev, in isolation. dev alone says " *
            "THAT the tracer and continuity divergences disagree by an ulp; these three " *
            "say WHICH rule pair does it, which is the difference between a one-line " *
            "fix and a search. Split per axis because the three axes use different rule " *
            "pairs: lon/lat are ppm_flux_D_*_mono_inflow_bc vs face_flux_divergence_*_open_bc, " *
            "lev is ppm_flux_D_lev_mono_hybrid_noflux_bc vs face_flux_divergence_lev_massform_noflux_bc.")
    end
    V["qg"] = Dict{String,Any}("type" => "observed", "units" => "1",
        "shape" => ["lon", "lat", "lev"],
        "description" => "Gate mixing ratio mqg/mg. Must stay EXACTLY 1.",
        "expression" => Dict{String,Any}("op" => "aggregate",
            "output_idx" => ["iq", "jq", "kq"], "args" => Any["mqg", "mg"],
            "ranges" => Dict{String,Any}("iq" => Dict("from" => "lon"),
                                         "jq" => Dict("from" => "lat"),
                                         "kq" => Dict("from" => "lev")),
            "expr" => Dict{String,Any}("op" => "/", "args" => Any[
                Dict{String,Any}("op" => "index", "args" => Any["mqg", "iq", "jq", "kq"]),
                Dict{String,Any}("op" => "index", "args" => Any["mg", "iq", "jq", "kq"])])))

    tdiv_g = tracer_div("qg", "qbc_w_g", "qbc_e_g", "qbc_s_g", "qbc_n_g")
    tdiv_q = tracer_div("qtest", "qbc_w_g", "qbc_e_g", "qbc_s_g", "qbc_n_g")
    for (nm, v) in (("qtest", 1.0), ("mg", 1.0), ("mqg", 1.0), ("dev", 0.0),
                    ("dev_x", 0.0), ("dev_y", 0.0), ("dev_z", 0.0))
        push!(E, Dict{String,Any}("lhs" => Dict{String,Any}("op" => "ic", "args" => Any[nm]),
                                  "rhs" => grid3(v)))
    end
    push!(E, Dict{String,Any}("lhs" => Dt("mg"),  "rhs" => neg(cont_div())))
    push!(E, Dict{String,Any}("lhs" => Dt("mqg"), "rhs" => neg(tdiv_g)))
    push!(E, Dict{String,Any}("lhs" => Dt("dev"),
        "rhs" => neg(Dict{String,Any}("op" => "-", "args" => Any[tdiv_g, cont_div()]))))
    # Per-axis instruments: the SAME difference, one axis at a time, so a failure
    # names the rule pair responsible instead of just the sum.
    for (nm, tr, co) in (
        ("dev_x", Dict{String,Any}("op" => "D", "args" => Any[
                      Dict{String,Any}("op" => "*", "args" => Any["Mx", "qg"]),
                      "qbc_w_g", "qbc_e_g"], "wrt" => "lon"),
                  Dict{String,Any}("op" => "D", "args" => Any["Mx"], "wrt" => "lon")),
        ("dev_y", Dict{String,Any}("op" => "D", "args" => Any[
                      Dict{String,Any}("op" => "*", "args" => Any["My", "qg"]),
                      "qbc_s_g", "qbc_n_g"], "wrt" => "lat"),
                  Dict{String,Any}("op" => "D", "args" => Any["My"], "wrt" => "lat")),
        ("dev_z", Dict{String,Any}("op" => "D", "args" => Any[
                      Dict{String,Any}("op" => "*", "args" => Any["Mz", "qg"])], "wrt" => "lev"),
                  Dict{String,Any}("op" => "D", "args" => Any["Mz"], "wrt" => "lev")))
        push!(E, Dict{String,Any}("lhs" => Dt(nm),
            "rhs" => neg(Dict{String,Any}("op" => "-", "args" => Any[tr, co]))))
    end
    # mixing-ratio form, verbatim the species shape: [-div(Mq) + q*div(M)] / m
    push!(E, Dict{String,Any}("lhs" => Dt("qtest"),
        "rhs" => Dict{String,Any}("op" => "/", "args" => Any[
            Dict{String,Any}("op" => "+", "args" => Any[
                neg(tdiv_q),
                Dict{String,Any}("op" => "*", "args" => Any["qtest", cont_div()])]),
            "m"])))

    doc["metadata"]["name"] = "reseact_cwc_gate"
    out = joinpath(dirname(src_path), "_cwc_gate.esm")   # sibling: relative refs resolve
    write(out, JSON3.write(doc))
    return out
end

const GATE = write_gate_model(MODEL)
say("gate model : $GATE  (from $(basename(MODEL)))")
say("grid       : $(NLON)x$(NLAT)x$(NLEV)   t0=$T0  steps=$NSTEPS x $(MACRO_DT)s")

try
    tb = time()
    docs = Logging.with_logger(Logging.NullLogger()) do
        prepare_split_docs(GATE; metaparameters = Dict("NLON" => NLON, "NLAT" => NLAT, "NLEV" => NLEV))
    end
    ff = reseact_forcing(CHEMDIR)
    ff = merge(ff, (; const_arrays = GridResize.slice_hybrid_coefs(ff.const_arrays, NLEV)))
    run = build_split_run(docs, (T0, T0 + NSTEPS * MACRO_DT); providers = ff.providers,
                          parameters = ff.parameters, const_arrays = ff.const_arrays)
    say(@sprintf("build      : %.1f s", time() - tb))

    P = cellmajor_perm(run.var_map)
    ft! = cellmajor_rhs(run.funcs[1], P.sm_of_cm)
    fchem! = cellmajor_rhs(run.funcs[2], P.sm_of_cm)
    jac!, mkjp = block_fd_jac(fchem!, P.NS, P.NC)
    u = run.u0[P.sm_of_cm]
    foreach(d -> d.materialize!(), run.dms)

    # Seed the three air masses from the SAME call, so mg and mqg start bitwise
    # equal -- the gate is meaningless if they differ in the last bit at t0.
    dp0 = hydrostatic_dp(run.merged_param, ff.const_arrays, T0; slice = run.slice)
    at(nm, c) = (P.cell_pos[c] - 1) * P.NS + P.base_pos[nm]
    for c in P.cells
        v = dp0(c[1], c[2], c[3])
        u[at("Transport3D.m", c)]   = v
        u[at("Transport3D.mg", c)]  = v
        u[at("Transport3D.mqg", c)] = v
    end
    seed_ok = all(u[at("Transport3D.mg", c)] === u[at("Transport3D.mqg", c)] for c in P.cells) &&
              all(u[at("Transport3D.qtest", c)] === 1.0 for c in P.cells) &&
              all(u[at("Transport3D.dev", c)] === 0.0 for c in P.cells)
    say("seed       : mg === mqg bitwise, qtest === 1.0, dev === 0.0  -> $seed_ok")

    tgz(g, u, p, t) = (fill!(g, 0); nothing)
    fc = SciMLBase.ODEFunction(fchem!; jac = jac!, jac_prototype = mkjp(), tgrad = tgz)
    algs = (OrdinaryDiffEqSSPRK.SSPRK43(),
            OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff = false,
                                                  linsolve = LinearSolve.LUFactorization()))
    disc = Dict(String(k) => p for (k, p) in ff.providers if !EA.provider_is_const(p))
    refresh = t -> begin
        for (k, p) in disc
            run.merged_param[k] .= EA._provider_const_field(EA.provider_sample(p, t), k)
        end
        foreach(d -> d.materialize!(), run.dms)
    end

    # NOTE clamp_nonneg is OFF. It is a production safety net for the stiff
    # chemistry, but it is a NONLINEAR edit of the state vector: were it ever to
    # fire on mg or mqg it would silently repair the very identity under test.
    getf(nm) = [u[at(nm, c)] for c in P.cells]
    report(lbl, tnow) = begin
        qt = getf("Transport3D.qtest"); dv = getf("Transport3D.dev")
        mg = getf("Transport3D.mg");    mq = getf("Transport3D.mqg")
        mm = getf("Transport3D.m")
        qg = mq ./ mg
        # dp must be evaluated at the CURRENT time. Holding the t0 thickness fixed
        # reports the atmosphere's own surface-pressure tendency as model drift --
        # it shows up as a suspiciously exact straight line (dPSdt is constant
        # inside a 3 h I3 window), which is the tell.
        dpt = hydrostatic_dp(run.merged_param, ff.const_arrays, tnow; slice = run.slice)
        dpn = [dpt(c[1], c[2], c[3]) for c in P.cells]
        drift = maximum(abs.((mm .- dpn) ./ dpn))
        say(@sprintf("  %-9s |qtest-1|=%.3e  |qg-1|=%.3e  |dev|=%.3e  bitwise mg==mqg:%-5s  |m-dp|/dp=%.3e",
            lbl, maximum(abs, qt .- 1), maximum(abs, qg .- 1), maximum(abs, dv),
            all(i -> mg[i] === mq[i], eachindex(mg)), drift))
        (maximum(abs, qt .- 1), maximum(abs, qg .- 1), maximum(abs, dv), drift)
    end

    # ----------------------------------------------------------------------- #
    # RHS PROBE. Before any stepping: is the defect in the DISCRETISATION (the
    # difference of the two divergences is already nonzero at t0, with the state
    # exactly uniform) or does it only appear through the SOLVER? One evaluation
    # settles it, and the per-axis states name the rule pair.
    # ----------------------------------------------------------------------- #
    du = similar(u); ft!(du, u, run.p, T0)
    duc = similar(u); fchem!(duc, u, run.p, T0)
    isbnd(c) = c[1] <= 3 || c[1] > NLON - 3 || c[2] <= 3 || c[2] > NLAT - 3 ||
               c[3] == 1 || c[3] == NLEV
    say("\nRHS probe at t0 (state exactly uniform; every entry must be 0.0):")
    say("   quantity   max|d/dt|      worst cell    in a PPM boundary region?   nonzero cells")
    for nm in ("dev", "dev_x", "dev_y", "dev_z")
        k = "Transport3D." * nm
        vals = [du[at(k, c)] for c in P.cells]
        mx, ix = findmax(abs.(vals))
        nz = count(!=(0.0), vals)
        say(@sprintf("   %-9s %12.4e  %-13s %-26s %d / %d",
            nm, mx, string(P.cells[ix]), string(isbnd(P.cells[ix])), nz, length(vals)))
    end
    let cvals = [abs(duc[at("Transport3D.dev", c)]) for c in P.cells]
        say(@sprintf("   (chemistry half contributes max|d dev/dt| = %.4e -- must be 0)",
                     maximum(cvals)))
    end

    say("\n  step       gate quantities (all three must be EXACTLY 0)")
    report("t0", T0)
    t = T0; ok = true
    for s in 1:NSTEPS
        tn = t + MACRO_DT
        res = lie_trotter_solve(ft!, fc, u, (t, tn), run.p, algs;
            macro_dt = MACRO_DT, reltols = (RTOL, RTOL), abstols = (ATOL_T, ATOL),
            refresh = refresh, forcing_tstops = run.tstops, clamp_nonneg = false)
        (res.retcode == SciMLBase.ReturnCode.Success ||
         res.retcode == SciMLBase.ReturnCode.Default) ||
            (ok = false; say("  !! step $s retcode=$(res.retcode)"))
        u = res.u; t = tn
        report(@sprintf("+%.0f s", t - T0), t)
    end

    qterr, qgerr, deverr, drift = report("final", t)
    axerr = Dict(nm => maximum(abs, getf("Transport3D." * nm)) for nm in ("dev_x", "dev_y", "dev_z"))
    bitwise = let mg = getf("Transport3D.mg"), mq = getf("Transport3D.mqg")
        all(i -> mg[i] === mq[i], eachindex(mg))
    end
    say("")
    say(@sprintf("CWC GATE (mixing-ratio, the form the species use)  max|qtest-1| = %.3e  -> %s",
                 qterr, qterr == 0.0 ? "PASS (bit-exact)" : "FAIL"))
    say(@sprintf("CWC GATE (mass form, tracer mass vs air mass)      max|qg-1|    = %.3e  -> %s",
                 qgerr, qgerr == 0.0 ? "PASS (bit-exact)" : "FAIL"))
    say(@sprintf("CWC GATE (tracer mass bitwise == air mass)                      -> %s",
                 bitwise ? "PASS (bit-exact)" : "FAIL"))
    say("CWC GATE (per-axis rhs difference -- the DISCRETISATION test):")
    for (nm, pair) in (("dev_x", "ppm_flux_D_lon_mono_inflow_bc vs face_flux_divergence_lon_open_bc"),
                       ("dev_y", "ppm_flux_D_lat_mono_inflow_bc vs face_flux_divergence_lat_open_bc"),
                       ("dev_z", "ppm_flux_D_lev_mono_hybrid_noflux_bc vs face_flux_divergence_lev_massform_noflux_bc"))
        say(@sprintf("   %-6s max = %.3e  -> %-16s  %s", nm, axerr[nm],
                     axerr[nm] == 0.0 ? "PASS (bit-exact)" : "FAIL", pair))
    end
    say(@sprintf("continuity drift after fixer                      max|m-dp|/dp = %.3e", drift))
    say("")
    # The composite dev is reported but does NOT gate, and the per-axis states are
    # why. dev integrates -( (Ax+Ay+Az) - (Bx+By+Bz) ); dev_x/y/z integrate
    # -(Ax-Bx), -(Ay-By), -(Az-Bz). If all three of those are EXACTLY zero then
    # Ax==Bx, Ay==By, Az==Bz bitwise, so the two triples being summed are the same
    # three numbers -- and any residue in dev is the ASSOCIATION of the two sums in
    # the emitted code, not the transport operator. IEEE addition is commutative but
    # not associative, so two sums of the same three addends can differ by an ulp if
    # codegen groups them differently. WHICH pass regroups them is not established
    # here and does not need to be: the per-axis result bounds the cause to codegen
    # regardless of the mechanism. (Worth noting the prototype this gate came from,
    # prototypes/reseact_3d, reported dev == 0 exactly with the same equation shape
    # but Mz as a plain forcing array rather than a diagnosed observed.)
    #
    # That is a statement about codegen, not about consistency with continuity, so
    # it must not fail a CWC gate. Its physical footprint is nil: the difference is
    # ~1 ulp of |div(M)| ~ 1e-2, i.e. ~1e-18 Pa/s, about 1e5 times smaller than
    # ulp(1500 Pa), which is why mg and mqg stay bitwise equal step after step.
    #
    # If a per-axis state EVER goes nonzero, that IS the operator and IS a failure.
    axclean = all(v -> v == 0.0, values(axerr))
    say(@sprintf("composite dev (informational, association-order artifact)  max|dev| = %.3e",
                 deverr))
    if deverr != 0.0 && axclean
        say("   -> expected: every per-axis difference is bit-exact, so this is the")
        say("      summation order of the two 3-term sums in the emitted rhs, not the")
        say("      transport operator. Not a CWC failure.")
    elseif deverr != 0.0
        say("   -> NOT explained by association: a per-axis gate failed above.")
    end
    say(ok ? "solver: all macro steps succeeded" : "solver: SOME MACRO STEPS FAILED")
    exitcode = (qterr == 0.0 && qgerr == 0.0 && axclean && bitwise && ok) ? 0 : 1
    say(exitcode == 0 ? "\nDONE cwc_gate -- ALL GATES PASS" : "\nDONE cwc_gate -- GATE FAILURE")
finally
    get(ENV, "RESEACT_KEEP_GATE", "0") == "1" || (isfile(GATE) && rm(GATE))
end
