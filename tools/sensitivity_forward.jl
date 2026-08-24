#!/usr/bin/env julia
# ===========================================================================
# sensitivity_forward.jl -- d(objective)/d(parameter) by FORWARD-mode Enzyme
# through the compiled Reactant macro step, chained across a macro-step window.
# ===========================================================================
# This is Phase 2 of DIFFERENTIABILITY_PLAN.md: the first REAL parameter
# sensitivity of ReSEACT. It exists because of ONE measured fact --
#
#     FORWARD mode crosses a `Reactant.@trace while` region EXACTLY;
#     REVERSE mode does not (it dies with `MLIR pass pipeline "all" failed`).
#
# -- and a macro step IS two such while regions (the SSPRK43 transport loop and
# the ROS23 chemistry loop in rx_traced_integrator.jl). So the whole-simulation
# adjoint has to wait for Phase 3's hand-written stage VJP, but a JVP w.r.t. a
# HANDFUL of parameters is available today at a cost proportional to the number
# of parameters -- one forward pass per parameter, which is why this driver
# takes a short list and not the whole vector.
#
# WHAT IS COMPILED. Structurally this is run_reseact_reactant.jl with one extra
# layer: the same two `:oop` split halves over the same live GEOS-FP buffers,
# the same `adv` Lie-Trotter macro step with `t0`/`t1`/`dt` as RUNTIME args, but
# `@compile`d through `Enzyme.autodiff(ForwardWithPrimal, ...)` so ONE call
# returns the primal step AND its directional derivative. The parameters under
# test are peeled out of the `p` NamedTuple into a tuple argument `ths`, and
# merged back in inside the trace, so BOTH their values and the tangent seed
# `dths` are ordinary runtime inputs:
#
#     (u, du, ths, dths, t0, t1, dtT, ddtT, dtC, ddtC, bufs...)
#         -> (u', du', dtT', ddtT', dtC', ddtC', counters...)
#
# ONE @compile therefore serves every parameter (a one-hot `dths` selects the
# direction) AND every finite-difference evaluation (perturb `ths`, zero the
# seed). That matters: the compile is the expensive part, and it is paid once.
#
# CHAINING. The tangent is loop-carried exactly like the state. `du` rides
# alongside `u`, and -- this is easy to miss -- so do the two step-size tangents
# `ddtT`/`ddtC`: the adaptive controller carries `dt` across macro steps, `dt`
# is a function of `u` and `p`, so it has a derivative, and dropping it would
# quietly bias the chain. The parameters are held fixed across the window (one
# physical value each), so the seed is re-fed unchanged every step and the
# accumulated `du` is the total derivative at the end of the window.
#
# WHY THE PARAMETERS CAN BE RUNTIME ARGUMENTS AT ALL. Measured: `p` under
# Reactant is a real XLA input, not a trace-time constant -- the same compiled
# program fed new parameter values produces new answers. That is what makes the
# finite-difference cross-check affordable: J(theta+h) and J(theta-h) reuse the
# SAME compiled program, no rebuild and no recompile.
#
# THE ACCEPTANCE CHECK IS THE POINT. A forward sensitivity that is never
# compared to a finite difference is a plausible-looking number. This driver
# always runs a central-difference step-size STUDY (several h, not one) through
# the identical compiled program with the tangent seed set to zero, and reports
# the whole table. Three failure modes it is built to catch:
#
#   * ZERO-GRADIENT-BY-CONSTANT-FOLDING. If a parameter were folded at build
#     time, changing it at runtime would do nothing and BOTH the AD and the FD
#     would return 0 -- agreeing perfectly and meaning nothing. So a zero AD
#     derivative is reported as a FAILURE, not a pass. (`const_tier.jl`
#     deliberately does not fold parameter-only subexpressions for precisely
#     this reason; this is the check that it held.)
#   * BRANCH FLIPS. The step-size controller's accept/reject test and the
#     `clamp_nonneg` max() are piecewise. AD differentiates the branch taken at
#     `theta`; an fd perturbation may cross into another one. The accept/reject
#     counters are compared against the unperturbed run for every fd evaluation
#     and flagged, so a disagreement can be attributed instead of guessed at.
#     That check is NECESSARY, not sufficient: equal counts do not prove the
#     same branch was taken everywhere, and it says nothing at all about the
#     elementwise kinks (`max(un, 0)`, `max(|u|, 1e-9)` in the FD Jacobian step,
#     the controller's q clamps), which is the reason the fd/AD agreement
#     measured here FLOORS at ~5e-7 relative instead of falling to roundoff.
#     The floor is flat across four decades of h -- not the h^2 of truncation
#     error and not the 1/h of cancellation -- which is what a small, fixed set
#     of kinked states looks like. Differentiating a program with kinks is what
#     was asked for; the number is the derivative of THIS program, and the fd
#     disagrees with it by the measure of how kinked the program is.
#   * A MIS-UNPACKED AD RESULT. Enzyme's `ForwardWithPrimal` returns a pair; get
#     its order backwards and the chain silently carries the primal in the
#     tangent slot and still produces plausible numbers. The order is settled
#     empirically on a two-second toy compile before anything expensive runs,
#     and a zero-seed pass through the full window must return an exactly zero
#     tangent.
#
# OBJECTIVE. Default: domain-mean SURFACE O3 (level 1 -- `slice_hybrid_coefs`
# keeps the FIRST NLEV entries of the hybrid table, i.e. the surface end) at the
# END of the window, in ppb. It is expressed as a weight vector `w` with
# J(u) = dot(w, u) and grad J = w, so swapping in a nonlinear objective means
# supplying its gradient and nothing else changes -- the chain rule is applied
# once, on the host, at the end: dJ = dot(gradJ(u_end), du_end).
#
# COST. Forward mode roughly doubles the traced program, so expect the @compile
# and the per-step cost to be ~2x run_reseact_reactant.jl's. Total window passes
# = 1 (baseline) + n_params (AD) + 2 * n_params * n_h (fd). Develop at a SMALL
# grid; RESEACT_SENS_SKIPFD=1 drops the fd study once it has been established.
#
# Env (the grid/window knobs are run_reseact_reactant.jl's, verbatim):
#   RESEACT_MODEL / LABEL / LON0 / LAT0 / NLON / NLAT / NLEV / T0 / MACRO_DT
#   RESEACT_DT0T / RESEACT_DT0C / RESEACT_RXENV
#   RESEACT_SENS_PARAM   COMMA-SEPARATED parameters to differentiate w.r.t.
#                        (default "NEIRegrid.scale,Transport3D.tau_pblmix")
#   RESEACT_SENS_NMACRO  macro steps in the window (default 3)
#   RESEACT_SENS_OBJ     "<state prefix>:<scope>", scope = surf | all | lev<K>
#                        (default "SuperFast.O3:surf")
#   RESEACT_SENS_FD      comma-separated RELATIVE fd steps; the absolute step is
#                        h = rel * max(|theta|, 1). Default 1e-2 .. 1e-7.
#   RESEACT_SENS_SKIPFD  =1 to report only the AD numbers (no acceptance check)
#   RESEACT_SENS_CSV     write the fd study table here
# ===========================================================================
import Pkg
const REPO = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using SciMLBase
using LinearAlgebra, Printf, Statistics, Logging, Random
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using Reactant
const EA = EarthSciAST
const RX = Reactant
const Enzyme = Reactant.Enzyme      # Enzyme is not a direct dep of the run env
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL     = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL     = get(ENV, "RESEACT_LABEL", "reseact-sens")
const T0        = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const MACRO_DT  = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const NMACRO    = parse(Int, get(ENV, "RESEACT_SENS_NMACRO", "3"))
const DT0T      = parse(Float64, get(ENV, "RESEACT_DT0T", "15.0"))
const DT0C      = parse(Float64, get(ENV, "RESEACT_DT0C", "0.5"))
const PARAMS    = String.(split(get(ENV, "RESEACT_SENS_PARAM",
                    "NEIRegrid.scale,Transport3D.tau_pblmix,NEIRegrid.g0"), ','))
const OBJSPEC   = get(ENV, "RESEACT_SENS_OBJ", "SuperFast.O3:surf")
const SKIPFD    = get(ENV, "RESEACT_SENS_SKIPFD", "0") == "1"
const SENSCSV   = get(ENV, "RESEACT_SENS_CSV", "")
const FDREL     = [parse(Float64, s) for s in
                   split(get(ENV, "RESEACT_SENS_FD", "1e-2,1e-3,1e-4,1e-5,1e-6,1e-7"), ',')]
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 13), nlat = _env("NLAT", 7),
                           nlev = _env("NLEV", 72))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const T_END    = T0 + NMACRO * MACRO_DT
const NDAYS    = forcing_days_for(T0, T_END)
const RTOL, ATOL_T, ATOL_C = 1e-4, 1e-6, 1e-9

say("=== $LABEL : FORWARD parameter sensitivity ($(basename(MODEL))) " *
    "grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF T0=$(round(Int,T0)) " *
    "nmacro=$NMACRO macro_dt=$(round(Int,MACRO_DT)) ===")
say("    d[$OBJSPEC]/d[$(join(PARAMS, ", "))] at t=$(round(Int,T_END)) s")

# --------------------------------------------------------------------------- #
# 1. Build -- identical to run_reseact_reactant.jl.
# --------------------------------------------------------------------------- #
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff
    file = EA.load_path(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
    f0 = reseact_forcing(CHEMDIR; ndays = NDAYS)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld
            discrete[k] = prov
        end
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part 2 var_map != part 1")
        end
    end
end
foreach(d -> d.materialize!(), dms)
say(@sprintf("BUILD %.2f s   nstates=%d  discrete_providers=%d",
    time() - tb, length(u0), length(discrete)))

# Seed m(0) from the real GEOS-FP surface pressure (species-major state).
let dp0 = hydrostatic_dp(merged_param, ff.const_arrays, T0; slice = SLICE)
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dp0(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
                      parse(Int, mm.captures[3]))
    end
end

# THE DEFAULT BASE POINT IS DEGENERATE FOR VALIDATION. Every SuperFast field in
# the default IC is EXACTLY spatially uniform, which puts u0 on the switching
# surface of essentially every PPM monotonicity limiter -- and the guard is
# quadratic in the perturbation, so the fd study below cannot see it (the long
# note in tools/rx_adjoint_check.jl has the measurement). RESEACT_SENS_UJITTER
# moves the base point off the uniform field using the SAME seeded stream that
# tools/rx_adjoint_check.jl and tools/adjoint_gradient.jl use, so "the jittered
# base point" is literally the same point in all three drivers and their
# gradients are directly comparable.
const UJIT = parse(Float64, get(ENV, "RESEACT_SENS_UJITTER", "0"))
if UJIT > 0
    u0 .*= (1 .+ UJIT .* randn(Random.MersenneTwister(31337), length(u0)))
    say(@sprintf("  base point jittered by %.0e relative (seed 31337)", UJIT))
end

# --------------------------------------------------------------------------- #
# 2. Every parameter under test must be a RUNTIME SCALAR entry of `p`. Anything
#    else is a build-time constant here and would return an unconditional zero
#    that the fd check could not distinguish from a true zero sensitivity.
#    NOTE, because it is a trap and it cost this driver a design revision:
#    `NEIRegrid.F_NO` ... `F_FORM` are NOT split fractions. They are the whole
#    NEI2016 source-grid emission FIELDS (12US1, ~137k cells), wired in as CONST
#    provider arrays and collapsed at build time by the conservative regrid --
#    deliberately, see split_common.jl's note on why the regrid must not be
#    per-step RHS IR. They never appear in the runtime `p` (measured: `p` here is
#    49 scalars and nothing else), so they are not differentiable through this
#    route at all, and this check rejects them rather than reporting a zero that
#    a finite difference would "confirm". The nearest runtime emissions knobs
#    that DO exist are `NEIRegrid.scale` (uniform multiplier) and `NEIRegrid.g0`
#    (the pressure-thickness -> column-mass divisor).
# --------------------------------------------------------------------------- #
const PSYMS = Tuple(Symbol.(PARAMS))
const NP = length(PSYMS)
let scal = sort([String(k) for k in keys(p) if getfield(p, k) isa Number])
    for (nm, s) in zip(PARAMS, PSYMS)
        s in keys(p) || error("""parameter "$nm" is not an entry of the runtime `p`.
            It is structural (folded at build time), a const array, or misspelled --
            differentiating w.r.t. it here would return an unconditional zero.
            Runtime scalar parameters available:\n    $(join(scal, "\n    "))""")
        getfield(p, s) isa Number ||
            error("parameter \"$nm\" is a $(typeof(getfield(p, s))), not a scalar; " *
                  "this driver seeds scalar tangents only")
    end
    say(@sprintf("  %d of %d runtime scalar parameters selected", NP, length(scal)))
end
const THETA0 = Tuple(Float64(getfield(p, s)) for s in PSYMS)
for (nm, th) in zip(PARAMS, THETA0)
    say(@sprintf("    %-28s = %.10g", nm, th))
end

include(joinpath(RXDIR, "rx_native_patch.jl"))       # AFTER using Reactant/EarthSciAST
include(joinpath(RXDIR, "rx_traced_integrator.jl"))

host_bufs = [EA.forcing_buffers(fo[i]) for i in 1:2]
g4 = [EA.rhs_with_buffers(fo[i]) for i in 1:2]
P = cellmajor_perm(var_map)
const NS = P.NS; const NC = P.NC; const N = P.N
masks = RxTracedIntegrator.species_masks(var_map, NS, NC)

# --------------------------------------------------------------------------- #
# 3. The objective, as a weight vector: J(u) = dot(w, u), grad J = w.
#    A nonlinear objective needs only its gradient swapped in below; the chain
#    rule is applied once, on the host, to the final (u, du).
# --------------------------------------------------------------------------- #
function objective_weights(var_map, spec::AbstractString)
    parts = split(spec, ':')
    length(parts) == 2 || error("RESEACT_SENS_OBJ must be \"<prefix>:<scope>\", got \"$spec\"")
    pre, scope = String(parts[1]), String(parts[2])
    rx = Regex("^" * replace(pre, "." => "\\.") * "\\[(\\d+),(\\d+),(\\d+)\\]\$")
    keep = if scope == "all"
        _ -> true
    elseif scope == "surf"
        k -> k == 1                       # level 1 is the SURFACE (see header)
    elseif startswith(scope, "lev")
        kk = parse(Int, scope[4:end]); k -> k == kk
    else
        error("unknown objective scope \"$scope\" (want surf | all | lev<K>)")
    end
    idxs = Int[]
    for (nm, idx) in var_map
        mm = match(rx, nm)
        mm === nothing && continue
        keep(parse(Int, mm.captures[3])) && push!(idxs, idx)
    end
    isempty(idxs) && error("objective \"$spec\" selected no states")
    w = zeros(Float64, length(var_map))
    w[idxs] .= 1 / length(idxs)
    return w, length(idxs)
end
const _OBJ = objective_weights(var_map, OBJSPEC)
const WOBJ = _OBJ[1]
const NOBJ = _OBJ[2]
objective(u) = dot(WOBJ, u)
objective_grad(_u) = WOBJ
say("  objective: mean over $NOBJ states matching $OBJSPEC")

# --------------------------------------------------------------------------- #
# 4. The traced macro step, with the parameters under test peeled out into a
#    tuple so their values AND their tangent seed are runtime arguments.
#    `merge` keeps `p`'s key order because every PSYM is already one of its keys.
# --------------------------------------------------------------------------- #
const CTRL_T = RxTracedIntegrator.pictrl_ssprk43()
const CTRL_C = RxTracedIntegrator.pictrl_ros23()
step_core = let gT = g4[1], gC = g4[2], NS = NS, NC = NC, masks = masks
    (uR, ths, t0R, t1R, dtTR, dtCR, bufsT, bufsC, p0) -> begin
        pR = merge(p0, NamedTuple{PSYMS}(ths))
        fT = (u, t, dtc, aux) -> RxTracedIntegrator.ssprk43_step(
            (uu, tt) -> gT(uu, aux.p, tt, aux.bufs), u, t, dtc, ATOL_T, RTOL)
        uT, _, dtTend, naT, nrT = RxTracedIntegrator.adaptive_solve(
            fT, uR, t0R, t1R, dtTR, CTRL_T, (p = pR, bufs = bufsT); clamp_nonneg = true)
        fC = (u, t, dtc, aux) -> RxTracedIntegrator.ros23_step(
            (uu, tt) -> gC(uu, aux.p, tt, aux.bufs), u, t, dtc, NS, NC, masks,
            ATOL_C, RTOL; unrolled = true)
        uC, _, dtCend, naC, nrC = RxTracedIntegrator.adaptive_solve(
            fC, uT, t0R, t1R, dtCR, CTRL_C, (p = pR, bufs = bufsC); clamp_nonneg = true)
        return (uC, dtTend, dtCend, naT, nrT, naC, nrC)
    end
end

# Two things have to be true before the expensive compile is worth starting, and
# both are cheap to settle on a toy: which slot of `autodiff(ForwardWithPrimal,
# ...)` holds the derivative, and whether `Duplicated` accepts a TUPLE of active
# scalars (which is how the parameter vector rides in). d/dx(2x+3y) with seed
# (1,0) is 2, with seed (0,1) is 3, and the primal at (3,5) is 21 -- all
# distinguishable.
const FWP_DERIV_FIRST = let
    isdefined(Enzyme, :ForwardWithPrimal) ||
        error("this Enzyme has no ForwardWithPrimal; the driver needs primal+tangent " *
              "from ONE traced program")
    probe = (xy, dxy) -> begin
        r = Enzyme.autodiff(Enzyme.ForwardWithPrimal, ab -> 2.0 * ab[1] + 3.0 * ab[2],
                            Enzyme.Duplicated(xy, dxy))
        (r[1], r[2])
    end
    xy = (RX.ConcreteRNumber(3.0), RX.ConcreteRNumber(5.0))
    s1 = (RX.ConcreteRNumber(1.0), RX.ConcreteRNumber(0.0))
    s2 = (RX.ConcreteRNumber(0.0), RX.ConcreteRNumber(1.0))
    xp = RX.@compile sync=true probe(xy, s1)
    a1, b1 = xp(xy, s1); a2, b2 = xp(xy, s2)
    f = (Float64(a1), Float64(b1), Float64(a2), Float64(b2))
    if f == (2.0, 21.0, 3.0, 21.0)
        true
    elseif f == (21.0, 2.0, 21.0, 3.0)
        false
    else
        error("ForwardWithPrimal tuple probe returned $f; expected the derivative " *
              "(2 then 3) and the primal (21) in some order")
    end
end
say("  Enzyme.ForwardWithPrimal returns " *
    (FWP_DERIV_FIRST ? "(derivative, primal)" : "(primal, derivative)") *
    "; Duplicated over an NTuple of active scalars works")

# ONE compiled program serves every parameter (one-hot seed) and every finite-
# difference evaluation (perturbed `ths`, zero seed). Forward mode does not
# touch the primal, so fd and AD are compared on identical arithmetic and any
# disagreement is a real derivative disagreement, not two different programs.
jvp_step = (u, du, ths, dths, t0, t1, dtT, ddtT, dtC, ddtC, bufsT, bufsC, p0) -> begin
    r = Enzyme.autodiff(Enzyme.ForwardWithPrimal, step_core,
        Enzyme.Duplicated(u, du), Enzyme.Duplicated(ths, dths),
        Enzyme.Const(t0), Enzyme.Const(t1),
        Enzyme.Duplicated(dtT, ddtT), Enzyme.Duplicated(dtC, ddtC),
        Enzyme.Const(bufsT), Enzyme.Const(bufsC), Enzyme.Const(p0))
    d, v = FWP_DERIV_FIRST ? (r[1], r[2]) : (r[2], r[1])
    return (v[1], d[1], v[2], d[2], v[3], d[3], v[4], v[5], v[6], v[7])
end

_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_devt(t::Tuple) = map(RX.ConcreteRNumber, t)
dev_bufs = [map(RX.ConcreteRArray, host_bufs[i]) for i in 1:2]
push_forcing!() = for i in 1:2; EA.sync_forcing!(dev_bufs[i], EA.forcing_buffers(fo[i])); end
push_forcing!()
PRd = _dev(p)

say("  @compile of ONE forward-JVP macro step (t0/t1/dt, the parameter values " *
    "and the tangent seed are all runtime args, so this compile serves every " *
    "parameter AND every fd evaluation)...")
tc = time()
xjvp = RX.@compile sync=true jvp_step(
    RX.ConcreteRArray(copy(u0)), RX.ConcreteRArray(zeros(Float64, length(u0))),
    _devt(THETA0), _devt(ntuple(_ -> 0.0, NP)),
    RX.ConcreteRNumber(T0), RX.ConcreteRNumber(T0 + MACRO_DT),
    RX.ConcreteRNumber(DT0T), RX.ConcreteRNumber(0.0),
    RX.ConcreteRNumber(DT0C), RX.ConcreteRNumber(0.0),
    dev_bufs[1], dev_bufs[2], PRd)
say(@sprintf("TRACED JVP @compile: %.1f s", time() - tc))
try; report_patch_stats(); catch; end

# --------------------------------------------------------------------------- #
# 5. The macro-step window. State AND tangent stay on device between calls.
# --------------------------------------------------------------------------- #
function refresh_forcing(t)
    for (k, prov) in discrete
        merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), dms)
    push_forcing!()
end

# Stop grid: macro lattice UNION the GEOS-FP cadence boundaries, half-open on
# the right -- run_reseact_reactant.jl's fencepost, kept verbatim so the window
# this differentiates is the window that runner would simulate.
const TSTOPS = sort!(unique!(Float64[t for prov in values(discrete)
                                     for t in EarthSciIO.refresh_times(prov)]))
const FSTOPS = Set(round(t; digits = 6) for t in TSTOPS if T0 + 1e-6 < t <= T_END + 1e-6)
const STOPS  = sort!(unique!(vcat(collect((T0 + MACRO_DT):MACRO_DT:(T_END - 1e-9)),
                                  collect(FSTOPS), Float64[T_END])))
say(@sprintf("  %d macro stops over [%.0f, %.0f] s, %d of them GEOS-FP boundaries",
             length(STOPS), T0, T_END, length(FSTOPS)))

"""
    run_window(thetas, seeds) -> (J, dJ, trace)

One pass over the window at parameter values `thetas` (an NP-tuple) with tangent
seed `seeds`. A one-hot seed gives that parameter's AD sensitivity; an all-zero
seed gives a pure primal solve for the finite differences, through the identical
compiled program. `trace` records the per-step accept/reject counts and the
carried step sizes -- the evidence for (or against) an fd perturbation having
crossed a controller branch.
"""
function run_window(thetas::NTuple{M,Float64}, seeds::NTuple{M,Float64}) where {M}
    refresh_forcing(T0)                       # make every pass independent
    UR  = RX.ConcreteRArray(copy(u0))
    dUR = RX.ConcreteRArray(zeros(Float64, length(u0)))
    thR = _devt(thetas); dthR = _devt(seeds)
    dtT = RX.ConcreteRNumber(DT0T); ddtT = RX.ConcreteRNumber(0.0)
    dtC = RX.ConcreteRNumber(DT0C); ddtC = RX.ConcreteRNumber(0.0)
    tcur = T0; trace = NamedTuple[]
    for tnext in STOPS
        tnext <= tcur + 1e-9 && continue
        r = xjvp(UR, dUR, thR, dthR, RX.ConcreteRNumber(tcur), RX.ConcreteRNumber(tnext),
                 dtT, ddtT, dtC, ddtC, dev_bufs[1], dev_bufs[2], PRd)
        UR, dUR = r[1], r[2]
        dtT, ddtT = r[3], r[4]
        dtC, ddtC = r[5], r[6]
        push!(trace, (t = tnext, naT = Float64(r[7]), nrT = Float64(r[8]),
                      naC = Float64(r[9]), nrC = Float64(r[10]),
                      dtT = Float64(dtT), dtC = Float64(dtC)))
        tcur = tnext
        round(tnext; digits = 6) in FSTOPS && refresh_forcing(tnext)
    end
    uh = Array(UR); duh = Array(dUR)
    all(isfinite, uh) || error("non-finite state at thetas=$thetas")
    all(isfinite, duh) || error("non-finite TANGENT at thetas=$thetas")
    return objective(uh), dot(objective_grad(uh), duh), trace
end

steps_sig(tr) = join((@sprintf("%d/%d,%d/%d", Int(r.naT), Int(r.nrT), Int(r.naC), Int(r.nrC))
                      for r in tr), " ")
onehot(j) = ntuple(i -> i == j ? 1.0 : 0.0, NP)
bump(j, h) = ntuple(i -> i == j ? THETA0[i] + h : THETA0[i], NP)

# --- the zero-seed pass: the fd baseline AND a sanity check ---------------- #
# A zero tangent seed must propagate to an exactly zero tangent through the whole
# window. If it does not, something in the JVP is producing 0*Inf -> NaN and no
# derivative below can be trusted.
J_BASE = 0.0; TR_BASE = NamedTuple[]
let r0 = run_window(THETA0, ntuple(_ -> 0.0, NP))
    J0z, dJ0z, tr0 = r0[1], r0[2], r0[3]
    dJ0z == 0.0 || error("a zero tangent seed produced a nonzero tangent ($dJ0z) after " *
                         "$(length(tr0)) macro steps -- a 0*Inf/NaN is leaking into the " *
                         "forward sweep; no derivative here is trustworthy")
    isfinite(J0z) && J0z != 0.0 ||
        error("zero-seed pass returned objective $J0z; expected a nonzero primal")
    global J_BASE = J0z
    global TR_BASE = tr0
    say(@sprintf("  zero-seed pass OK: primal J=%.12g, tangent exactly 0", J0z))
    say("  base accept/reject per macro step (T then C): $(steps_sig(tr0))")
end

# --------------------------------------------------------------------------- #
# 6. One forward pass per parameter. This is the cost model of the whole phase:
#    forward sensitivity is O(n_params), which is why the list is short.
# --------------------------------------------------------------------------- #
AD = zeros(Float64, NP)
for j in 1:NP
    ta = time()
    Jj, dJj, _ = run_window(THETA0, onehot(j))
    AD[j] = dJj
    Jj == J_BASE || say(@sprintf("  NOTE: seeded primal %.17g != zero-seed primal %.17g (reldiff %.3e) -- forward mode should not move the primal",
                                 Jj, J_BASE, abs(Jj - J_BASE) / abs(J_BASE)))
    say(@sprintf("AD  d[%s]/d[%s] = %.12e     [%.1f s]", OBJSPEC, PARAMS[j], dJj, time() - ta))
    dJj == 0.0 && error("""AD sensitivity is EXACTLY zero for $(PARAMS[j]).
        That is the signature of a parameter that never reaches the objective at
        runtime -- constant-folded at build time, or reaching the answer only through
        a discrete closed function (interp.searchsorted / datetime.*), both of which
        return zero partials by contract. Not a pass.""")
end

# --- a STRUCTURAL cross-check that finite differences cannot fake ---------- #
# In reseact.esm every NEI emission rate is literally
#     X_emis = C_X * Exy_X / days_in_month * (g0 / delp) * scale * diurnal * dow
# (see the five *_emis observeds in models.NEIRegrid), and `g0` appears NOWHERE
# else in the model. So the emission field is exactly proportional to BOTH
# `scale` and `g0`, and by the chain rule through that single product
#     scale * dJ/dscale  ==  g0 * dJ/dg0
# identically -- an EXACT algebraic identity between two independently computed
# forward sweeps, with no finite difference anywhere in it. It checks the
# chaining and the seeding in a way an fd comparison cannot, because an error
# that corrupted both sweeps the same way would still break the ratio.
let is = findfirst(==("NEIRegrid.scale"), PARAMS), ig = findfirst(==("NEIRegrid.g0"), PARAMS)
    if is !== nothing && ig !== nothing
        lhs = THETA0[is] * AD[is]; rhs = THETA0[ig] * AD[ig]
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
        say(@sprintf("  structural identity  scale*dJ/dscale = g0*dJ/dg0 :  %.12e vs %.12e   rel=%.3e  %s",
                     lhs, rhs, rel, rel <= 1e-12 ? "OK" : "MISMATCH"))
    end
end

# --------------------------------------------------------------------------- #
# 7. Central finite differences, with a step-size STUDY. A single h and no
#    sensitivity check is not evidence: too large and truncation error
#    dominates, too small and cancellation does, and only a plateau across
#    several decades shows you are between the two.
# --------------------------------------------------------------------------- #
rows = NamedTuple[]
if !SKIPFD
    for j in 1:NP
        scale = max(abs(THETA0[j]), 1.0)
        say("  central finite differences for $(PARAMS[j]) " *
            "(same compiled program, zero seed; h = rel * $(scale)):")
        say("        h          J(theta+h)          J(theta-h)        fd derivative       rel err vs AD  branch")
        for rel in FDREL
            h = scale * rel
            Jp, _, trp = run_window(bump(j, h), ntuple(_ -> 0.0, NP))
            Jm, _, trm = run_window(bump(j, -h), ntuple(_ -> 0.0, NP))
            fd = (Jp - Jm) / (2h)
            err = abs(fd - AD[j]) / abs(AD[j])
            same = steps_sig(trp) == steps_sig(TR_BASE) && steps_sig(trm) == steps_sig(TR_BASE)
            push!(rows, (j = j, h = h, Jp = Jp, Jm = Jm, fd = fd, err = err, same = same))
            say(@sprintf("  %10.3e  %.15g  %.15g  %.12e   %.3e   %s",
                         h, Jp, Jm, fd, err, same ? "same" : "MOVED"))
        end
    end
    bests = [(rj = [r for r in rows if r.j == j]; rj[argmin([r.err for r in rj])]) for j in 1:NP]
    for j in 1:NP
        b = bests[j]
        say(@sprintf("BEST fd agreement for %-28s h=%.3e  fd=%.12e  ad=%.12e  rel=%.3e  %s",
                     PARAMS[j], b.h, b.fd, AD[j], b.err, b.err <= 1e-6 ? "PASS" : "FAIL"))
    end
    allpass = all(b.err <= 1e-6 for b in bests)
    say(allpass ?
        "ACCEPTANCE PASS: every parameter's forward AD matches central fd to <= 1e-6 relative." :
        "ACCEPTANCE FAIL: at least one parameter's best fd/AD relative disagreement exceeds 1e-6.")
    if !isempty(SENSCSV)
        open(SENSCSV, "w") do io
            println(io, "param,objective,theta0,ad,h,J_plus,J_minus,fd,rel_err,same_branch")
            for r in rows
                println(io, join((PARAMS[r.j], OBJSPEC, THETA0[r.j], AD[r.j], r.h,
                                  r.Jp, r.Jm, r.fd, r.err, r.same), ","))
            end
        end
        say("wrote $SENSCSV")
    end
end

for j in 1:NP
    say(@sprintf("RESULT label=%s param=%s theta0=%.10g obj=%s nmacro=%d window_s=%.0f grid=%dx%dx%d nstates=%d J=%.10g dJ_dtheta=%.10e",
                 LABEL, PARAMS[j], THETA0[j], OBJSPEC, NMACRO, T_END - T0,
                 GRID_MP["NLON"], GRID_MP["NLAT"], NLEV_EFF, N, J_BASE, AD[j]))
    # elasticity: the dimensionless (dJ/J) / (dtheta/theta), which is what a
    # "a 10% emissions cut buys you X%" statement actually needs.
    say(@sprintf("       elasticity (theta/J)*dJ/dtheta = %+.6f  [a +1%% change in %s moves %s by %+.6f%%]",
                 THETA0[j] * AD[j] / J_BASE, PARAMS[j], OBJSPEC,
                 THETA0[j] * AD[j] / J_BASE))
end
say("DONE $LABEL")
