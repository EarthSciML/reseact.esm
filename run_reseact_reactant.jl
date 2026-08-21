#!/usr/bin/env julia
# ===========================================================================
# run_reseact_reactant.jl -- the traced (Reactant/XLA) twin of run_reseact.jl.
# ===========================================================================
# Same default run: a ONE-WEEK CONUS simulation, 13x7x72 (6,552 cells, 85,176
# states), macro-stepped at 300 s, over the same GEOS-FP meteorology. Same
# Lie-Trotter scheme (SSPRK43 transport, then Rosenbrock23 chemistry with a block
# Jacobian). The difference is entirely in who does the arithmetic: both adaptive
# loops run inside `Reactant.@trace while` -> `stablehlo.while` regions, so each
# half's RHS graph appears exactly ONCE in the compiled module however many steps
# the solve takes, and there is no host<->device round trip per RHS evaluation.
#
# THE MACRO-STEP LOOP IS THE POINT. This runner used to compile the WHOLE window
# as one XLA program and take a single Lie-Trotter step over it. That is the right
# shape for a 60 s validation window and the wrong shape for anything longer: at a
# week it means a 604,800 s splitting interval with the meteorology frozen at the
# start of the run. It would return a number, and the number would not be a
# simulation. So the structure here is
#
#   compile ONCE  ->  call 2016 times  ->  refresh forcing between calls
#
# Three properties of the traced program make that work, none of them accidental:
#
#   * `t0`, `t1`, `dt0T`, `dt0C` are RUNTIME arguments (ConcreteRNumber) and both
#     adaptive loops are `stablehlo.while` regions, so the compiled program does
#     not encode the window. ONE @compile serves every macro step, whatever its
#     length -- the compile cost is paid once and amortized over the whole run.
#   * the forcing buffers are REAL XLA INPUTS (`rhs_with_buffers` / B2 form), and
#     EarthSciAST guarantees that an in-place `copyto!` into those device arrays
#     between calls is observed on the next call with NO RETRACE. That is what
#     makes a discrete-cadence refresh survive compilation.
#   * the state stays on device between calls: `u` is fed straight back in, and so
#     are the final `dt`s, so the step-size controller carries across macro steps
#     exactly as it does within one.
#
# The forcing refresh needs a DiscreteMaterializer. Both halves get one here, and
# a refresh is merged_param .= provider_sample -> dm.materialize!() ->
# sync_forcing! to device.
#
# The stop grid follows `lie_trotter_solve`'s, INCLUDING its fencepost: the
# forcing-boundary set is half-open on the right, because a boundary that lands on
# a macro-step edge would otherwise be excluded from the step that ends there AND
# the step that starts there, and never refresh at all. See op_split.jl and
# HELPERS.md §1.
#
# PERFORMANCE, MEASURED OVER A FULL 24 h RUN, SO NOBODY HAS TO GUESS. At CONUS
# 13x7x72: build 718 s, @compile 557 s (paid ONCE for the whole run), solve
# 5,923 s for 288 macro steps. The native runner does the same 24 h in 2,143 s,
# so Reactant is ~2.8x SLOWER on CPU for this model. It is here because it is the
# route to a DIFFERENTIABLE, device-portable simulation, not because it is faster
# today -- see DIFFERENTIABILITY_PLAN.md.
#
# Do not extrapolate the step cost from a short window: it is strongly diurnal,
# ~9.6 s/step pre-dawn against ~35 s/step at midday (20.6 s/step averaged over the
# day), because the photochemistry is what makes the chemistry half stiff. A
# 3-macro-step sanity window sits in the dark AND carries the first-step warm-up,
# and gets the right answer for the wrong reasons.
#
# Agreement with the native arm over 24 h (289 aligned records), RE-MEASURED
# 2026-08-20 with the XLA:CPU race workaround ON (both sides pinned to the same
# pre-rename EarthSciAST; tools/diag/compare_traced_native.py):
#
#   column     race active      race fixed     change
#   O3_mean       6.3e-5        1.199e-7       525x BETTER
#   O3_min        1.6e-3        2.116e-4       7.6x better
#   OH_max        1.1e-3        1.072e-3       unchanged
#   air mass      5.2e-12       4.914e-12      unchanged
#   cos_sza       exactly 0     exactly 0
#
# THE RACE WAS THE DOMINANT ERROR SOURCE ON O3_mean, by two and a half orders of
# magnitude. The old 6.3e-5 was NOT "adaptive-solver path divergence" as this
# comment used to claim -- it was mostly the race, and with the race fixed the
# two independent implementations agree to 1.2e-7. That is the justification for
# defaulting RESEACT_RXFIX on: it costs ~6% of wall clock and buys 525x on the
# headline species.
#
# OH_max is the honest exception -- unchanged at ~1.1e-3, so THAT one really is
# solver-path divergence (OH is a fast radical and this is a max over cells, the
# most path-sensitive diagnostic here). Still far inside the 5e-2 validation
# tolerance.
#
# Env: the same as run_reseact.jl (RESEACT_MODEL / LABEL / LON0 / LAT0 / NLON /
# NLAT / NLEV / T0 / SOLVE_SECS / MACRO_DT / CSV / ZARR / OUT_EVERY /
# FLUSH_EVERY), plus
#   RESEACT_DT0T / RESEACT_DT0C  initial transport/chemistry dt (15.0 / 0.5)
#   RESEACT_RXENV                Julia env WITH Reactant (default run-model-jl)
#   RESEACT_RXJAC                fd (default) | ad | sym -- the chemistry block
#                                Jacobian. sym is EarthSciASTDiff's analytic one
#                                and needs EarthSciASTDiff in the project (use
#                                /u/ctessum/reseact-esast-pin/env-sym); see 1b.
#   RESEACT_RXFIX                XLA:CPU race workaround (1 = default, on)
#   RESEACT_RXJAC                chem block Jacobian: fd (default) or ad (exact,
#                                coloured forward-mode JVPs) -- see below
#
# Helper code it pulls in (see HELPERS.md for the migration plan):
#   prototypes/reseact_3d_chem/{split_common,blockdiag_local,block_jac}.jl
#   tools/reactant_handoff/rx_native_patch.jl        trace enablers (monkey-patch)
#   tools/reactant_handoff/rx_traced_integrator.jl   the traced ROS23/SSPRK43 stepper
# ===========================================================================
import Pkg
const REPO = @__DIR__
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using SciMLBase, DiffEqCallbacks
using LinearAlgebra, Printf, Statistics, Logging
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using Reactant
using Blosc                      # activates EarthSciIOBloscExt for the Zarr sink
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
# block_jac.jl needs `BlockDiagonal` in scope even though this driver only wants
# `cellmajor_perm` from it (the block-diagonal Jacobian is the HOST arm's tool;
# the traced arm builds its own inside ros23_step).
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL      = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL      = get(ENV, "RESEACT_LABEL", "reseact-rx")
const T0         = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const SOLVE_SECS = parse(Float64, get(ENV, "RESEACT_SOLVE_SECS", "604800"))   # 7 days
const MACRO_DT   = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const CSV        = get(ENV, "RESEACT_CSV", "")
const ZARR       = get(ENV, "RESEACT_ZARR", "")
const OUT_EVERY  = parse(Int, get(ENV, "RESEACT_OUT_EVERY", "12"))     # hourly at macro_dt=300
const FLUSH_EVERY = parse(Int, get(ENV, "RESEACT_FLUSH_EVERY", "12"))
const DT0T       = parse(Float64, get(ENV, "RESEACT_DT0T", "15.0"))
const DT0C       = parse(Float64, get(ENV, "RESEACT_DT0C", "0.5"))
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 13), nlat = _env("NLAT", 7),
                           nlev = _env("NLEV", 72))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const T_END    = T0 + SOLVE_SECS
const NDAYS    = forcing_days_for(T0, T_END)
const RTOL, ATOL_T, ATOL_C = 1e-4, 1e-6, 1e-9

# Row-label sun angle at the domain centre (see run_reseact.jl).
const K_GAMMA, D2R = 0.01721420632103996, 0.017453292519943295
const LON_MID = sum(SLICE.lon_deg) / 2
const LAT_MID = sum(SLICE.lat_deg) / 2
function cos_sza(t, lat, lon)
    g = K_GAMMA * (-12 / 24 + t / 86400)
    dec = 0.006918 - 0.399912cos(g) + 0.070257sin(g) - 0.006758cos(2g) +
          0.000907sin(2g) - 0.002697cos(3g) + 0.00148sin(3g)
    eqt = 229.18 * (0.000075 + 0.001868cos(g) - 0.032077sin(g) -
                    0.014615cos(2g) - 0.040849sin(2g))
    w = D2R * ((t / 60 + eqt + 4lon) / 4 - 180)
    return clamp(sin(D2R * lat) * sin(dec) + cos(D2R * lat) * cos(dec) * cos(w), -1, 1)
end

say("=== $LABEL : traced macro-step run ($(basename(MODEL))) " *
    "grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF T0=$(round(Int,T0)) " *
    "window=$(round(SOLVE_SECS/3600, digits=2)) h macro_dt=$(round(Int,MACRO_DT)) ===")
say(@sprintf("    slice: lon %.1f..%.1f, lat %.1f..%.1f (native origin %d,%d); forcing spans %d daily files",
    SLICE.lon_deg[1], SLICE.lon_deg[2], SLICE.lat_deg[1], SLICE.lat_deg[2],
    SLICE.lon0, SLICE.lat0, NDAYS))

# --------------------------------------------------------------------------- #
# 1. Split + build both halves :oop, over shared live forcing buffers, each with
#    its OWN DiscreteMaterializer so a refresh re-derives that half's caches.
#    :oop is required for tracing: an in-place f! captures host scratch per node
#    and cannot trace.
# --------------------------------------------------------------------------- #
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = docs1 = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing
# escaped so RESEACT_RXJAC=sym can prepare the symbolic Jacobian from the SAME
# split part, with the SAME overrides, that part 2's RHS is built from.
splitparts = nothing; merged_const = nothing; ov = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff, docs1
    global splitparts, merged_const, ov
    file = EA.load(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in splitparts]
    docs1 = docs[1]
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

# Seed m(0) from the real GEOS-FP surface pressure (species-major state here).
let dp0 = hydrostatic_dp(merged_param, ff.const_arrays, T0; slice = SLICE)
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dp0(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
                      parse(Int, mm.captures[3]))
    end
end

include(joinpath(RXDIR, "rx_native_patch.jl"))       # AFTER using Reactant/EarthSciAST
include(joinpath(RXDIR, "rx_traced_integrator.jl"))
include(joinpath(RXDIR, "rx_sym_block_jac.jl")); using .RxSymBlockJac

# The kernel-class merge runs INSIDE EarthSciAST's build (src/tree_walk/oop_merge.jl),
# so `build_evaluator(form=:oop)` already hands back lane-batched kernels and the
# RHS here needs nothing further. There used to be a driver-side merge with a
# bit-identity gate in front of it (rx_merge_lib.jl); its reconstruction predated
# materialized array-observed levels and had been failing its own gate -- i.e.
# falling back to this same stock RHS -- on every run since. Deleted rather than
# repaired: the in-package merge is the one that is tested.
const PARTNAME = ("transport", "pointwise")
host_bufs = [EA.forcing_buffers(fo[i]) for i in 1:2]
g4 = [EA.rhs_with_buffers(fo[i]) for i in 1:2]

P = cellmajor_perm(var_map)
const NS = P.NS; const NC = P.NC; const N = P.N
masks = RxTracedIntegrator.species_masks(var_map, NS, NC)

# --------------------------------------------------------------------------- #
# 2. The traced advance: ONE Lie-Trotter step over [t0, t1].
# --------------------------------------------------------------------------- #
const CTRL_T = RxTracedIntegrator.pictrl_ssprk43()
const CTRL_C = RxTracedIntegrator.pictrl_ros23()
const JACMODE = Symbol(get(ENV, "RESEACT_RXJAC", "fd"))
JACMODE in (:fd, :ad, :sym) || error("RESEACT_RXJAC must be fd, ad or sym, got $JACMODE")
const SYMJAC = JACMODE === :sym

# --------------------------------------------------------------------------- #
# 1b. RESEACT_RXJAC=sym -- EarthSciASTDiff's ANALYTIC block Jacobian.
#
# WHY IT MATTERS TO THIS RUNNER, which has no adjoint in it. The longest CONUS
# run on record before 2026-08-20 died at 36.2 h of simulated time, and the
# mechanism recorded at the time was: surface O3 dry-deposits away (correct
# physics), leaving species at ~1e-30, and the FD chemistry Jacobian then fails
# on them -- FD's step is sqrt(eps)*|u|, which at 1e-30 is not a step at all.
# The analytic Jacobian has no step size and evaluates the derivative AT that
# concentration, so it is finite where FD is garbage. That is the specific thing
# this option exists to test.
#
# It also traces SMALLER than either alternative (109k lines of MLIR against
# 323k for both :fd and :ad on the adjoint driver's single step), which matters
# rather more inside a `@trace while` body than it does standalone.
#
# Needs EarthSciASTDiff in the active project -- run-model-jl cannot resolve it;
# use /u/ctessum/reseact-esast-pin/env-sym.
PLAN = gjbJ = host_bufsJ = dev_bufsJ = jacE = nothing
if SYMJAC
    @eval using EarthSciASTDiff
    tj = time()
    jacE = Logging.with_logger(Logging.NullLogger()) do
        EarthSciASTDiff.prepare_jacobian(splitparts[2]; wrt = :states,
            build_kwargs = (; form = :oop, parameter_overrides = ov,
                              const_arrays = merged_const, param_arrays = merged_param))
    end
    say(@sprintf("  prepare_jacobian %.1f s   structure=%s  oop=%s",
                 time() - tj, jacE.structure, jacE.oop))
    jacE.oop || error("RXJAC=sym: the band model came back IN-PLACE; it captures " *
                      "host scratch per node and cannot be traced")
    # indexes by POSITION, and the two var maps come from independent builds
    PLAN = block_jac_plan(jacE; runner_names = first.(sort(collect(var_map), by = last)))
    gjbJ = EA.rhs_with_buffers(jacE.fJ!)
    host_bufsJ = EA.forcing_buffers(jacE.fJ!)
    say("  $PLAN   band buffers=$(length(host_bufsJ))")
    let w = validate_plan(PLAN, jacE, u0, p, T0; gjb = gjbJ, bufs = host_bufsJ)
        say(@sprintf("  plan vs the host JacobianEvaluator: worst relative %.3e  %s",
                     w, w <= 1e-12 ? "PASS" : "FAIL"))
        w <= 1e-12 || error("RXJAC=sym: the gather plan does not reproduce the host Jacobian")
    end
end
adv = let gT = g4[1], gC = g4[2], NS = NS, NC = NC, masks = masks,
         PLAN = PLAN, gjbJ = gjbJ, SYMJAC = SYMJAC
    (uR, pR, t0R, t1R, dtTR, dtCR, bufsT, bufsC, bufsJ) -> begin
        fT = (u, t, dtc, aux) -> RxTracedIntegrator.ssprk43_step(
            (uu, tt) -> gT(uu, aux.p, tt, aux.bufs), u, t, dtc, ATOL_T, RTOL)
        uT, _, dtTend, naT, nrT = RxTracedIntegrator.adaptive_solve(
            fT, uR, t0R, t1R, dtTR, CTRL_T, (p = pR, bufs = bufsT); clamp_nonneg = true)
        fC = (u, t, dtc, aux) -> RxTracedIntegrator.ros23_step(
            (uu, tt) -> gC(uu, aux.p, tt, aux.bufs), u, t, dtc, NS, NC, masks, ATOL_C, RTOL;
            # The host-unrolled Jacobian emits 16 traced copies of the chem RHS per
            # step against the traced loop's 4. That used to make it compile-
            # intractable at CONUS -- XLA's `cse_slice` pattern is pairwise over
            # slice ops, so 4x the copies was ~12x the compile work, and this
            # configuration once ran 3h45m without finishing. EarthSciAST's read
            # interning (one emitted read per (SSA value, window)) removed the
            # duplicate slices feeding that quadratic and the cost went away.
            # Measured head to head at CONUS, same slice and window: unrolled
            # compiles 2% FASTER (547 s vs 560 s) and solves 1.67x faster, because
            # the better Jacobian also cuts rejected chem steps from 23 to 5.
            # There is no tradeoff left, so there is no knob.
            unrolled = true,
            # RESEACT_RXJAC=ad swaps the FD block Jacobian for the EXACT one
            # (coloured forward-mode JVPs, `ad_block_jac`). Default stays :fd
            # because the exact Jacobian has not been shown to pay for itself
            # HERE: measured at 6x6x8 over one 300 s macro step it is compile-
            # neutral (89.3 s vs 87.3 s) and step-count-neutral (120 accepts /
            # 4 rejects vs 116 / 3), agreeing with the FD arm to 8.9e-4 max
            # relative on the smallest species. The native arm's switch to an
            # AD Jacobian cut rejected steps at CONUS, so this may well win at
            # CONUS too -- but nobody has measured that, and the default should
            # not change on a hope. See tools/rx_adjoint_check.jl.
            jac = JACMODE,
            # theta reaches the band model through `aux`, not through a closure:
            # `aux.bufsJ` is the band model's OWN forcing buffer set, so the
            # Jacobian's meteorology advances with the RHS's rather than being
            # baked in at the trace.
            symjac = SYMJAC ?
                ((uu, tt) -> block_jac(PLAN, gjbJ(gather_uj(PLAN, uu), aux.p, tt, aux.bufsJ))) :
                nothing)
        uC, _, dtCend, naC, nrC = RxTracedIntegrator.adaptive_solve(
            fC, uT, t0R, t1R, dtCR, CTRL_C,
            SYMJAC ? (p = pR, bufs = bufsC, bufsJ = bufsJ) : (p = pR, bufs = bufsC);
            clamp_nonneg = true)
        return uC, naT, nrT, naC, nrC, dtTend, dtCend
    end
end
_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
dev_bufs = [map(RX.ConcreteRArray, host_bufs[i]) for i in 1:2]
SYMJAC && (dev_bufsJ = map(RX.ConcreteRArray, host_bufsJ))
function push_forcing!()
    for i in 1:2; EA.sync_forcing!(dev_bufs[i], EA.forcing_buffers(fo[i])); end
    # miss this and the Jacobian freezes at the T0 epoch while the RHS advances:
    # a wrong Jacobian that still converges, which is the worst kind.
    SYMJAC && EA.sync_forcing!(dev_bufsJ, EA.forcing_buffers(jacE.fJ!))
    return nothing
end
push_forcing!()
# a placeholder operand when there is no band model, so the compiled program has
# one arity regardless of RXJAC.
const BUFSJ = SYMJAC ? dev_bufsJ : (; )

PRd = _dev(p)
UR = RX.ConcreteRArray(copy(u0))
say("  @compile of ONE macro step (t0/t1/dt are runtime args, so this compile " *
    "serves every step of the run)...")
# XLA:CPU RACE WORKAROUND -- on by default, because the race is a CORRECTNESS
# problem here, not just a speed one. XLA:CPU has a data race in its intra-op
# parallel execution that intermittently returns non-finite values from the
# chemistry program (~1% of calls at 4 threads; 0 of 400,000 at one thread).
# `xla_cpu_prefer_vector_width=128` is the measured fix (0 of 20,000 against a
# baseline of 4 and 4); see tools/diag/README-nondet.md to reproduce the fault.
#
# Why it matters to a runner that has no adjoint in it: a faulting call NaNs
# `EEst`, the PI controller reads that as a rejection, and each spurious rejection
# ALSO shrinks dt -- so the solve grinds out many more, smaller accepted steps and
# integrates a measurably DIFFERENT trajectory. Measured on the adjoint driver at
# CONUS (slurm 10015169): rejections 24.5% -> 2.9%, accepted steps 428 -> 170,
# forward pass 247.12 s -> 78.61 s (3.14x), objective shifted 4.0e-5 relative.
#
# THAT 3.14x DOES NOT CARRY OVER TO THIS RUNNER -- MEASURED, NOT ASSUMED.
# The hypothesis was that the 5,923 s / 24 h figure in the header above, measured
# with the race active, was itself ~3x too high. It is not. Re-running exactly
# that window with the workaround ON (slurm 10017939, 2026-08-19) gave:
#
#   build 700.6 s (was 718) | @compile 635.3 s (was 557, +14%) | solve 6,301.7 s
#   (was 5,923, +6.4% SLOWER) | nT=939/2  nC=12913/1422
#
# So the workaround COSTS ~6% here rather than saving anything, and the 24 h / week
# baselines stand (a week forward is ~12.2 h, marginally worse than 11.5 h).
#
# WHY THE DIFFERENCE. The 3.14x was measured at adjoint_gradient.jl's JITTERED base
# point (u .*= 1 + 0.1*randn, needed for FD validity), where the trajectory is
# pathological: ~189 chemistry attempts per macro step against ~50 here, median dt
# 1.31 s, and a controller already at its limits, so one NaN cascades. On a real
# trajectory the solve is robust and the residual 9.9% chemistry rejection rate is
# GENUINE STIFFNESS, not the race. Do not project race cost from the jittered point.
#
# The workaround stays ON BY DEFAULT anyway: ~6% of wall clock is a cheap premium
# for not silently integrating a different trajectory. RESEACT_RXFIX=0 to disable.
#
# `compile_options` REPLACES every other compile option (Reactant Macros.jl:7),
# so `sync = true` has to be set inside it rather than alongside it.
const RXFIX = get(ENV, "RESEACT_RXFIX", "1") == "1"
const COPTS = RXFIX ?
    RX.CompileOptions(; sync = true,
                      xla_debug_options = (; xla_cpu_prefer_vector_width = 128)) :
    RX.CompileOptions(; sync = true)
say("  XLA:CPU race workaround: " *
    (RXFIX ? "ON  xla_cpu_prefer_vector_width=128" : "OFF -- expect intermittent NaN"))
tc = time()
xadv = RX.@compile compile_options=COPTS adv(UR, PRd, RX.ConcreteRNumber(T0),
                                 RX.ConcreteRNumber(T0 + MACRO_DT),
                                 RX.ConcreteRNumber(DT0T), RX.ConcreteRNumber(DT0C),
                                 dev_bufs[1], dev_bufs[2], BUFSJ)
say(@sprintf("TRACED @compile: %.1f s", time() - tc))
try; report_patch_stats(); catch; end

# --------------------------------------------------------------------------- #
# 3. Forcing refresh -- the whole point of macro-stepping. EarthSciAST documents
#    that copyto! into the device buffers between calls is observed with NO
#    retrace (rhs_with_buffers), so this costs one host->device copy, not a
#    recompile.
# --------------------------------------------------------------------------- #
function refresh_forcing(t)
    for (k, prov) in discrete
        merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), dms)
    push_forcing!()
end

# Stop grid: the macro lattice UNION the GEOS-FP cadence boundaries. Half-open on
# the right, exactly as lie_trotter_solve builds it -- a boundary landing on a
# macro edge must belong to the step that ENDS there, or it is skipped by both
# neighbours and never refreshes (op_split.jl documents the measurement).
tstops = sort!(unique!(Float64[t for prov in values(discrete)
                               for t in EarthSciIO.refresh_times(prov)]))
fstops = Set(round(t; digits = 6) for t in tstops if T0 + 1e-6 < t <= T_END + 1e-6)
stops  = sort!(unique!(vcat(collect((T0 + MACRO_DT):MACRO_DT:(T_END - 1e-9)),
                            collect(fstops), Float64[T_END])))
say(@sprintf("  %d macro stops, %d of them GEOS-FP cadence boundaries", length(stops),
             length(fstops)))

# --------------------------------------------------------------------------- #
# 4. Digest + the continuity regression check (same quantities as run_reseact.jl,
#    computed on the SPECIES-MAJOR state the traced arm carries).
# --------------------------------------------------------------------------- #
idxof(pre) = sort!([idx for (nm, idx) in var_map if startswith(nm, pre)])
const O3I = idxof("SuperFast.O3["); const OHI = idxof("SuperFast.OH[")
const NO2I = idxof("SuperFast.NO2["); const MI = idxof("Transport3D.m[")
const MCELL = let d = Dict{NTuple{3,Int},Int}()
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        d[(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
           parse(Int, mm.captures[3]))] = idx
    end
    d
end
const NLON_ = maximum(c[1] for c in keys(MCELL))
const NLAT_ = maximum(c[2] for c in keys(MCELL))
const _dA = Float64.(ff.const_arrays["Transport3D.dA"])
const _dB = Float64.(ff.const_arrays["Transport3D.dB"])
w_I3(t) = (t - 10800.0 * floor(t / 10800.0)) / 10800.0
function continuity_drift(u, t)
    F = merged_param["GEOSFP.GEOSFP_I3.PS"]; w = w_I3(t)
    isw(c) = c[1] == 1 || c[1] == NLON_ || c[2] == 1 || c[2] == NLAT_
    s2i = 0.0; ni = 0; s2w = 0.0; nw = 0; worsti = 0.0; wci = (0, 0, 0)
    for (c, idx) in MCELL
        i, j, k = c
        ps = 100.0 * ((1 - w) * F[1, SLICE.lat0 + j, SLICE.lon0 + i] +
                            w * F[2, SLICE.lat0 + j, SLICE.lon0 + i])
        dp = _dA[k] + _dB[k] * ps
        r = (u[idx] - dp) / dp
        if isw(c)
            s2w += r * r; nw += 1
        else
            s2i += r * r; ni += 1
            abs(r) > abs(worsti) && (worsti = r; wci = c)
        end
    end
    return (rms_int = sqrt(s2i / max(ni, 1)), rms_wall = sqrt(s2w / max(nw, 1)),
            worst_int = worsti, cell_int = wci)
end

sink = nothing
if !isempty(ZARR)
    out_times = collect(T0:(MACRO_DT * OUT_EVERY):T_END)
    sink = EA.build_zarr_sink(var_map, ZARR; output_times = out_times,
                              meta = EA.derive_output_meta(docs1))
    EA.sink_open!(sink)
    say("  gridded output -> $ZARR  ($(length(out_times)) records)")
end
sink_put!(t, u) = sink === nothing ? nothing :
    EA.sink_write!(sink, EA.StateSnapshot(Float64(t), [(u, (1:N,))], Dict{String,Array}()))

rows = NamedTuple[]
push_row!(t, u, wall) = push!(rows, (t = t, hours = t / 3600,
    cos_sza = cos_sza(t, LAT_MID, LON_MID),
    o3_min = minimum(u[O3I]), o3_mean = mean(u[O3I]), o3_max = maximum(u[O3I]),
    oh_max = maximum(u[OHI]), no2_mean = mean(u[NO2I]),
    m_min = minimum(u[MI]), wall = wall))
report(t, u, wall) = begin
    push_row!(t, u, wall); r = rows[end]; d = continuity_drift(u, t)
    say(@sprintf("   %6.2f  %7.4f  %9.5f  %9.5f  %9.5f  %10.3e  %9.3e  | %8.2e %9.2e %9.2e %-12s %6.1f",
        r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max, r.oh_max, r.no2_mean,
        d.rms_int, d.rms_wall, d.worst_int, string(d.cell_int), r.wall))
end

# --------------------------------------------------------------------------- #
# 5. The macro-step loop. State and step sizes stay on device between calls.
# --------------------------------------------------------------------------- #
say("  t_hours  cos_sza     O3_min     O3_mean     O3_max      OH_max     NO2_mean  | rms_int  rms_wall  worst_int @cell_int    wall_s")
report(T0, copy(u0), 0.0); sink_put!(T0, copy(u0))

ts = time(); tcur = T0; nT = nC = nrTt = nrCt = 0; nstep = 0; nrefresh = 0; ok = true
dtT = RX.ConcreteRNumber(DT0T); dtC = RX.ConcreteRNumber(DT0C)
for tnext in stops
    tnext <= tcur + 1e-9 && continue
    res = xadv(UR, PRd, RX.ConcreteRNumber(tcur), RX.ConcreteRNumber(tnext),
               dtT, dtC, dev_bufs[1], dev_bufs[2], BUFSJ)
    global UR = res[1]
    global dtT = res[6]; global dtC = res[7]
    global nT += Int(round(Float64(res[2]))); global nrTt += Int(round(Float64(res[3])))
    global nC += Int(round(Float64(res[4]))); global nrCt += Int(round(Float64(res[5])))
    global tcur = tnext; global nstep += 1
    uh = Array(UR)
    if !all(isfinite, uh)
        global ok = false
        say("  !! non-finite state after macro step ending at t=$tnext -- stopping")
        break
    end
    report(tnext, uh, time() - ts)
    if nstep % OUT_EVERY == 0
        sink_put!(tnext, uh)
        sink === nothing || (nstep % (OUT_EVERY * FLUSH_EVERY) == 0 && EA.sink_flush!(sink))
    end
    if round(tnext; digits = 6) in fstops
        refresh_forcing(tnext); global nrefresh += 1
    end
end
solve_s = time() - ts
sink === nothing || (EA.sink_flush!(sink); EA.sink_close!(sink);
                     say("  gridded output committed: $ZARR"))

o3m = [r.o3_mean for r in rows]
say(@sprintf("RESULT label=%s cells=%d nstates=%d window_h=%.2f macro_dt=%.0f nmacro=%d nrefresh=%d nT=%d/%d nC=%d/%d solve_s=%.1f",
    LABEL, NC, N, SOLVE_SECS / 3600, MACRO_DT, nstep, nrefresh, nT, nrTt, nC, nrCt, solve_s))
say(@sprintf("DIURNAL O3_mean: start=%.5f min=%.5f max=%.5f end=%.5f  peak-to-trough=%.5f ppb  ok=%s",
    o3m[1], minimum(o3m), maximum(o3m), o3m[end], maximum(o3m) - minimum(o3m), ok))
if !isempty(CSV)
    open(CSV, "w") do io
        println(io, "t,hours,cos_sza,o3_min,o3_mean,o3_max,oh_max,no2_mean,m_min,wall")
        for r in rows
            println(io, join((r.t, r.hours, r.cos_sza, r.o3_min, r.o3_mean, r.o3_max,
                              r.oh_max, r.no2_mean, r.m_min, r.wall), ","))
        end
    end
    say("wrote $CSV")
end
say("DONE $LABEL")
