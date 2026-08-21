#!/usr/bin/env julia
# ===========================================================================
# run_reseact_adjoint.jl -- the SIMULATION GRADIENT, as far as it goes today.
# ===========================================================================
# The third runner. `run_reseact.jl` integrates the model natively,
# `run_reseact_reactant.jl` integrates it through Reactant/XLA, and this one
# DIFFERENTIATES it: d(objective)/d(parameter vector) for a simulation window,
# where the objective is a functional of the end state (default: mean surface
# O3) and the parameters are the model's runtime scalars.
#
# It is a demonstration, not a production driver. It runs a SHORT window on a
# SMALL grid because that is what is validated end to end. What it cannot yet
# do -- a week of CONUS -- is the subject of the blockers section below, which
# is the more useful half of this file.
#
#   julia --project=run-model-jl run_reseact_adjoint.jl
#
# ---------------------------------------------------------------------------
# WHAT WORKS TODAY (all measured; see DIFFERENTIABILITY_PLAN.md for provenance)
# ---------------------------------------------------------------------------
# * THE 48-HOUR CONUS GRADIENT (slurm 10044327, 2026-08-21, 8 h 18 m, MaxRSS
#   55.2 GB). This is what the blockers section below used to say was out of
#   reach, and it is now the headline result rather than a projection.
#   13x7x72, 85,176 states, 576 macro steps, `clamp_nonneg` ON, un-jittered,
#   jac=:sym, all 49 parameters in ONE backward sweep:
#     J        = 30.1943301698541 ppb mean surface O3 over 48 h
#     forward    7,774.93 s  (576 macro steps, 27,973 accepted inner steps)
#     backward  19,273.16 s  (27,973 VJPs at 0.4612 s; replay 6,372.10 s = 33%)
#     dJ/d(NEIRegrid.scale) = -2.114   d/d(DryDepositionGas.kappa) = -1.617
#     dJ/d(Transport3D.g_acc) = +0.669  -- 19 of 49 components nonzero
#   EVERY REFERENCE-FREE CHECK PASSED: the gather plan reproduces the host
#   Jacobian at 0.000e+00, the fixed-sequence replay lands 0.000e+00 from all
#   576 checkpoints, ZERO flaky-reverse retries over 27,973 VJP calls, and the
#   structural identity scale*dJ/dscale == g0*dJ/dg0 holds to 8.611e-15.
#   A week projects to ~27 h on these numbers.
# * A DISCRETE ADJOINT over the Lie-Trotter macro-step loop. One backward sweep
#   returns the gradient w.r.t. ALL 49 runtime scalars at once, at a cost that
#   does NOT scale with the parameter count -- measured 2.48x one forward solve
#   at CONUS over 48 simulated hours, or 1.66x with the replay excluded, against
#   forward mode's cost-proportional-to-n_params. The 5.1x quoted here until
#   2026-08-21 was measured at DEMONSTRATION SCALE and does not hold at CONUS.
# * FORWARD SENSITIVITY as the independent cross-check, for a handful of named
#   parameters. On CONUS this gave d(surface O3)/d(NEIRegrid.scale) =
#   -7.387e-2 ppb, agreeing with central differences to 5.5e-9. A +1% emissions
#   perturbation moves surface O3 by -0.00186%.
# * `p` IS A REAL XLA INPUT. The same compiled program accepts new parameter
#   values and returns new answers, so a gradient does not force a recompile per
#   parameter value -- decisive, given a ~550 s compile.
# * The parameter-vector ABI: `p` may be a `Vector` or a `ComponentVector`, not
#   only a `NamedTuple`, and the Float64 path stays bit-identical.
#
# HOW GOOD IS THE GRADIENT? Against finite differences OF THE SAME COMPILED MAP
# the adjoint agrees to 4.3e-10. Against Phase 2's forward sensitivity it agrees
# only to 3.1e-6, and that gap is structural rather than a bug: see "the host
# loop" below.
#
# ---------------------------------------------------------------------------
# WHAT THIS RUN ACTUALLY PRINTS -- read this before reading the output
# ---------------------------------------------------------------------------
# A recorded run of exactly this script (6x6x8, 3 macro steps, build 599 s,
# 3,744 states, 49 parameters, exit 0) produced:
#
#   * J = 38.844190627265334, and the frozen replay reproduced it BIT-IDENTICALLY
#     (diff 0.000e+00) -- so the tape and the replay are faithful.
#   * The STRUCTURAL IDENTITY scale*dJ/dscale == g0*dJ/dg0 held to 7.9e-13. This
#     is the check to trust: it is exact algebra between two components of ONE
#     gradient vector and needs no external reference at all.
#   * A full 49-component gradient, e.g. d(surface O3)/d(NEIRegrid.scale) =
#     -8.008e-2, d/d(DryDepositionGas.kappa) = -1.144e-1.
#   * AND TWO LINES SAYING "FAIL". Do not read those as a wrong gradient. They
#     are the FD REFERENCE failing, not the adjoint:
#       - Transport3D.tau_pblmix PASSES at 4.8e-8 -- but only at 3 of its 5 step
#         sizes; at h=9e-3 and h=9e-5 the central difference came back NaN.
#         NaN at interior step sizes with finite neighbours on BOTH sides is not
#         a step-size effect. It is blocker 4 landing on the reference, and this
#         run is a clean instance of it. NOW DIAGNOSED as an XLA:CPU threading
#         race (see blocker 4): at ~251 compiled calls per replay and ~2e-4
#         state-visible fault rate each, ~5% of replays and ~10% of (parameter, h)
#         pairs should be hit -- which is what this table shows. The step size is
#         telling you nothing. Re-running under
#         `xla_cpu_prefer_vector_width=128`, or on one XLA thread, should clear it.
#       - NEIRegrid.scale gets ONE finite step out of five, at 2.4e-5.
#       - NEIRegrid.g0 gets NONE, and used to print `fd=0.0 rel=Inf`, which is
#         the uninitialised `best` sentinel rendering as though FD had measured a
#         zero derivative. That is a far more alarming claim than "no FD was
#         obtained", and it is the exact shape of the const-folded-parameter trap
#         where a wrong zero and a wrong check agree silently. Fixed in
#         tools/adjoint_gradient.jl: it now says NO FINITE FD ... UNCHECKED, not
#         refuted. The g0 component is independently corroborated anyway, by the
#         structural identity above.
#   * The `ref` (forward-mode) stage SELF-DISABLED: its zero-seed guard found
#     r[1] norm = 2.33e4 where a zero tangent seed must give exactly zero, and
#     concluded Enzyme's forward mode is returning the primal rather than the
#     tangent on this RHS. NOTE THAT THIS CONTRADICTS an earlier investigation
#     which tested that exact question and cleared it (zero seed gave exactly
#     zero; the per-step dot-product identity <lam,Jv> == <J'lam,v> passed at
#     1e-15..1e-16 on the real model). Both cannot be right. It is UNRESOLVED at
#     time of writing, and it is deliberately not papered over here: either the
#     guard has a false positive, or the earlier clearing was measured on a path
#     this stage does not take. Until it is settled, prefer `fdtape` and the
#     structural identity, which is what the driver already tells you to do.
#
# The honest summary: the ADJOINT is in good shape and its reference-free check
# passes at 7.9e-13. The things that fail here are the CHECKS -- an intermittent
# non-finite bug in the compiled programs, and one unresolved contradiction about
# forward mode. That is also why blocker 4 is not a footnote.
#
# ---------------------------------------------------------------------------
# WHY THE LOOP RUNS ON THE HOST -- and the correction that reopened it
# ---------------------------------------------------------------------------
# The adaptive time loop is lifted OUT of the compiled program: stage attempts,
# the PI controller, accept/reject and the positivity clamp all run in ordinary
# Julia, and XLA is only ever asked to differentiate ONE step at a FIXED dt.
#
# The stated reason used to be "reverse mode cannot cross a `stablehlo.while`".
# THAT WAS WRONG, and it was retracted on 2026-08-12. Reverse mode crosses a
# `stablehlo.while` exactly, provided the TRIP COUNT IS A COMPILE-TIME CONSTANT;
# Enzyme-MLIR's `AutoDiffWhileRev` tapes the trajectory as a dense `[N, state]`
# tensor, and a non-static `N` makes that `tensor<?x...>`, which XLA cannot
# translate. The old evidence came from a probe labelled "FIXED trip count" that
# actually passed its bound as a runtime argument.
#
# The practical situation is nevertheless UNCHANGED for the loops as written:
# their trip count is data-dependent (the controller decides), which upstream
# does not yet support (Enzyme-JAX #2565, open). So the host lift stays for now.
# What changed is that it is no longer believed to be FORCED -- see next steps.
#
# THE COST OF THE HOST LIFT, since it shows up in every number here:
#   * XLA compiles the same step body differently inside a `while` region and as
#     a standalone program. Measured divergence 2.8e-10 per macro step in the
#     chemistry state, compounding to ~3.8e-6 in the objective. So the adjoint is
#     the exact derivative of the HOST map, and the forward sensitivity is a
#     derivative of the DEVICE map. They are two slightly different floating-
#     point functions, which is the whole of the 3.1e-6 disagreement above.
#   * Re-running the controller from a checkpoint does NOT reproduce the forward
#     pass (95/2 accepts vs 92/1, same inputs, same process) because the PI
#     controller turns that into a different accept/reject decision. The forward
#     pass therefore RECORDS the accepted (t, dt) sequence -- 16 B/step -- and the
#     backward replay replays it with the controller switched off. (The CAUSE is
#     not "an amplified ulp", as recorded until 2026-08-12: it is the XLA:CPU race
#     of blocker 4 NaN-ing `EEst`, which the controller reads as a rejection.
#     The recording is still needed; only the explanation changed.)
#
# ---------------------------------------------------------------------------
# BLOCKERS TO A WEEK OF CONUS -- in the order they will actually bite
# ---------------------------------------------------------------------------
# 1. THE CLAMP MAKES THE SWEEP GO NON-FINITE. **CLOSED 2026-08-21.** It used to
#    read: with the production `clamp_nonneg` ON, lambda acquires non-finite
#    entries partway back through the third macro step (312 entries, 24 cells,
#    all 13 state groups at once, at 6x6x8); not reproducible from the recorded
#    inputs; unexplained; the single biggest obstacle to trusting a long run.
#
#    IT WAS A CONSEQUENCE OF BLOCKERS 2 AND 4, not a fact about the clamp, and
#    the two were never independent suspects. `clamp_nonneg` writes EXACT ZEROS
#    into the state. An FD Jacobian differences the chemistry RHS with step
#    sqrt(eps)*|u|, which at u = 0 is not a step at all, so the clamp is
#    precisely what MAKES the FD Jacobian degenerate; and the race (blocker 4)
#    supplied the nondeterminism that made it irreproducible from the recorded
#    inputs. The observation predates both fixes and had never been repeated with
#    either.
#
#    Retested with both (slurm 10042578, tools/diag/adjoint_clamp_retest.sbatch):
#    the identical configuration run twice, clamp off then clamp on, jac=:sym and
#    the race workaround on in both. The CLAMPED sweep is finite throughout --
#    0 non-finite entries, 0 flaky-reverse retries over 251 VJP calls, replay
#    landing 0.000e+00 from every checkpoint -- with the clamp bit actually
#    firing on 18 of 939,744 (state, accepted-step) pairs, so it is being
#    exercised rather than sitting idle. The clamped and unclamped gradients
#    agree to ~10 significant figures on every component (e.g. NEIRegrid.scale
#    -8.008304991e-2 vs -8.008305251e-2) and the structural identity
#    scale*dJ/dscale == g0*dJ/dg0 passes at 6.9e-16 and 3.5e-16 respectively.
#
#
#    CONFIRMED AT SCALE, which is the part that matters: the 48 h CONUS run
#    above had the clamp ON for all 576 macro steps, it fired 50,973 times over
#    2.38e9 (state, accepted-step) pairs, and the sweep stayed finite with zero
#    retries. The configuration that used to die at macro step 3 now survives
#    576 of them.
#
#    So the adjoint now differentiates the map the production runner actually
#    uses. RESEACT_ADJ_CLAMP still defaults to 0 in the preset below only to keep
#    this demonstration comparable with the recorded output quoted above; the
#    long runs (tools/diag/adjoint_conus_48h.sbatch) set it to 1.
# 2. THE BLOCK JACOBIAN IS FINITE DIFFERENCE. **CLOSED 2026-08-20.** It used to
#    read: `ros23_step` builds its linearization by FD, so the adjoint carries FD
#    contamination on top of a step that `clamp_nonneg` and the controller
#    already make piecewise; and the exact alternative (`jac=:ad`) SEGFAULTS
#    under reverse mode, upstream, in `AutoDiffCallRev::createReverseModeAdjoint`
#    (reproducer reduced to ONE `enzyme.fwddiff` inside one reverse pass; both
#    routes around the nesting also broken upstream -- `BatchDuplicated` leaves an
#    `enzyme.extract` XLA rejects, `Ops.batch` emits a rank-mismatched transpose).
#    All of that is still true of `:ad`. It is now MOOT, because there is a third
#    option that is neither.
#
#    `RESEACT_ADJ_JAC=sym` (the default since 2026-08-20) uses EarthSciASTDiff's
#    ANALYTIC Jacobian: a separately generated "band model" emitting the nonzero
#    entries, gathered into cell blocks by tools/reactant_handoff/rx_sym_block_jac.jl.
#    It contains no nested AD at all, so reverse mode crosses it -- which is the
#    whole point, and it is why this closes the blocker rather than working around
#    it. Measured, at 6x6x8 with the race workaround on:
#      * vs `:ad`, max entry-wise relative error 5.0e-16, and ZERO entries that
#        `:ad` calls nonzero and the gather plan leaves structurally zero;
#      * the per-step dot-product identity <lam,Jv> == <J'lam,v> goes to 5.3e-16
#        in the state direction, 2.3e-16 in p. The SAME check on the SAME jittered
#        base point with `:fd` reads 1.8e-6 (slurm 10042219 vs 10041660) -- i.e.
#        the historical 1e-6 was the FD Jacobian, not the race, and the two were
#        confounded until both were measured with the workaround on;
#      * it is CHEAPER, not a trade: the Jacobian alone compiles in 12.0 s against
#        49.9 s (`:fd`) and 309.2 s (`:ad`), and the ROS23 primal module is
#        109,035 lines of MLIR against 323,179. The VJP module handed to Enzyme is
#        while-free and call-free.
#    The gather plan is validated on the host against EarthSciASTDiff's own sparse
#    Jacobian at every startup (`validate_plan`, worst relative 0.0 at 6x6x8 and
#    at CONUS) and it THROWS rather than falling back.
#    `:fd` and `:ad` remain reachable via RESEACT_ADJ_JAC for comparison.
#    NB `:sym` needs EarthSciASTDiff in the active project -- see the env note
#    at the foot of this header.
# 3. COMPUTE. The tape is cheap -- 681 kB per macro step, so ~1.4 GB for a week,
#    no Griewank scheme warranted. The TIME is the problem, and it is now
#    measured at CONUS rather than projected from demonstration scale (slurm
#    10015169: 3 macro steps, forward 78.61 s / 170 accepted inner steps,
#    backward 225.51 s, of which replay 73.30 s = 32.5% and VJPs 152.21 s =
#    67.5% at 0.8954 s/VJP). The backward/forward ratio is 2.87, or 1.94 with the
#    replay removed -- NOT the 5.1x measured at demonstration scale.
#    Scaled off the traced 24 h CONUS solve (6,301.7 s with the race workaround
#    on, slurm 10017939): a week forward is ~12.2 h and a week adjoint ~35 h,
#    ~24 h if a device-side fixed-step loop removes the replay. Against the
#    stated budget that is the open question, which is why the current target is
#    48 h of simulation rather than a week: ~1/3.5 of those figures.
#    jac=:sym MOVED THESE NUMBERS, measured at CONUS on the identical 3-macro-step
#    configuration (slurm 10042579 against 10015169): forward 78.61 -> 46.16 s,
#    backward 225.51 -> 124.91 s, 0.8954 -> 0.4955 s/VJP. That is 1.70x and 1.81x,
#    for an EXACT Jacobian rather than at the cost of one -- and the objective is
#    unchanged at 39.6131238549519 vs ...521, i.e. the same 170 accepted inner
#    steps, so the speedup is per-VJP work and not a different trajectory.
#    On those figures 48 h is ~2.5 h forward + ~6.7 h backward, and a week
#    ~19 h all in, against the ~35 h projected from the FD numbers.
#
#    THE 48 h CONUS FORWARD RUN NOW COMPLETES (slurm 10042577): 576 macro steps,
#    64 forcing refreshes, solve 10,358.8 s, nT=1859/4 nC=26113/2941, O3_min
#    17.85 ppb at the end with mass-continuity residuals at 1e-13. So the 36.2 h
#    death is gone -- note that run still used the FD Jacobian, so PBL mixing was
#    the fix for it, not jac=:sym. 24 h -> 48 h costs 1.64x, sublinear, because
#    the second day is less stiff than the spin-up.
# 4. INTERMITTENT NON-FINITE RESULTS -- DIAGNOSED 2026-08-12: it is a DATA RACE
#    IN XLA:CPU's intra-op parallel execution. With ONE XLA thread it does not
#    happen at all (0 of 400,000 calls across ten compile variants); at 4 threads
#    it is ~1% per call. The emitted module is a pure StableHLO dataflow graph --
#    no `while`, `rng`, `custom_call` or `sort` -- so this is not numerics: XLA
#    is executing a pure function wrongly, and needs concurrency to do it.
#    Narrowed to the fusion emitters at 256-bit vector width. The corruption is
#    always the same shape: NaN in exactly the six dry-deposition species at one
#    cell (essentially always the first surface cell), everything else
#    bit-identical. Chemistry RHS only; transport is 0 of 70,000.
#    WORKAROUND, per-compile and no rebuild:
#      Reactant.CompileOptions(; sync = true,
#                              xla_debug_options = (; xla_cpu_prefer_vector_width = 128))
#    0 of 20,000 against a baseline of 4 and 4, at no wall-time cost. NOT
#    numerically free -- it moves the answer by up to 1.6e-6 relative on one
#    step, so it is a new floor, not a null change. Pinning XLA:CPU to one thread
#    also works and changes no numbers at all, at the cost of the thread pool.
#    Reproduce in ~5 minutes: tools/diag/README-nondet.md.
#
#    THIS ALSO EXPLAINS THE REPLAY SIGNATURE. A faulting call NaNs `EEst` while
#    usually leaving the state bit-identical (the second RHS evaluation feeds
#    only the error estimate), and `host_adaptive!` maps `isnan(EEst)` to a
#    rejection -- so the 95/2-vs-92/1 replay discrepancy is SPURIOUS REJECTED
#    STEPS, not the "PI controller amplifies an ulp" mechanism recorded until
#    today. There was never a ULP-level difference in over 400,000 calls.
#    Whether it also explains blocker 1 is untested but is now the obvious first
#    hypothesis -- run the clamped sweep under the workaround before assuming a
#    second, independent bug.
#
# ---------------------------------------------------------------------------
# NEXT STEPS, highest value first
# ---------------------------------------------------------------------------
# A. PUT THE LOOP BACK ON THE DEVICE by BOUNDING it rather than removing it. An
#    adaptive loop needs a BOUND, not a `while` region: recast as a `@trace for`
#    over a compile-time attempt cap with termination as an `ifelse` on the
#    carry. A full accept/reject PI-controller stepper in that form
#    differentiates exactly, and `adaptive_solve`'s body is ALREADY written that
#    way -- every update is already `ifelse(accept, ...)`. Only the loop
#    CONDITION has to move. Toy-priced at 0.83x the while's wall time when the
#    cap equals the actual trip count, 1.28x with two-phase bucketing (a cheap
#    forward `while` to learn the count, then differentiate at the next power of
#    two). This retires the 3.1e-6 host/device gap AND the entire (t, dt)
#    recording and replay apparatus. Untried at ReSEACT state sizes -- the tape
#    is dense `[cap, state]`, so large state x large cap is the risk.
# B. Explain blocker 1. It is the only one that is ours rather than upstream's,
#    and it has never been retested with EITHER of the two things that have
#    changed underneath it (the race workaround, and now `jac=:sym`). Note the
#    two are not independent of the clamp: the clamp writes EXACT ZEROS, and an
#    FD Jacobian differencing a stiff RHS about a zero concentration has step
#    size sqrt(eps)*|u| = 0. So the clamped, FD-Jacobian configuration in which
#    blocker 1 was recorded is degenerate by construction.
#    tools/diag/adjoint_clamp_retest.sbatch runs exactly that comparison.
# C. `jac=:sym` inside `adaptive_solve`'s traced `@trace while`. DONE and
#    measured (slurm 10042632, RESEACT_RXJAC=sym): the band model traces inside
#    the loop region, compiles 4% faster (442.8 s vs 460.2 s), solves 1.49x
#    faster (5.3 s vs 7.9 s) and takes the identical accept/reject path
#    (nT=5/1, nC=208/5) to 1e-13 on O3. The forward runner's DEFAULT is still
#    :fd, only because `run-model-jl` cannot resolve EarthSciASTDiff at all;
#    that packaging problem is what stands between the measurement and the
#    default.
# D. Only then attempt a long window, and stage it: 48 h first, against a forward
#    sensitivity for two or three parameters as an independent check.
# E. File the `:ad` reverse-over-forward segfault upstream (write-up ready in
#    tools/diag/UPSTREAM_reverse_over_forward.md). Lower priority now that
#    `jac=:sym` means nothing here depends on it.
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE CHANGING THE BASE POINT
# ---------------------------------------------------------------------------
# Do NOT validate a gradient at the default initial condition. Every SuperFast
# field in it is EXACTLY spatially uniform, which puts u0 exactly ON the
# switching surface of essentially every PPM monotonicity limiter. Those guards
# are products of neighbour differences compared against zero, so they are
# QUADRATIC in the perturbation: forward, backward and central differences all
# flip to the same branch and AGREE WITH EACH OTHER while none of them is the
# derivative. The usual `fwd != bwd` kink test is blind to it. Hence the default
# jitter below. See the long note in tools/rx_adjoint_check.jl.
#
# ---------------------------------------------------------------------------
# Env: everything the two underlying drivers take (RESEACT_ADJ_* and
# RESEACT_SENS_*, documented in tools/adjoint_gradient.jl and
# tools/sensitivity_forward.jl), plus:
#
#   RESEACT_DEMO   adjoint | forward | both   (default "adjoint")
#                  adjoint = full 49-parameter gradient, one backward sweep
#                  forward = per-parameter forward sensitivity + FD check
#                  both    = forward first, then adjoint (independent check)
#
# Anything already set in the environment WINS over the defaults chosen here, so
# this script can be used as a thin preset over the real drivers.
#
# ---------------------------------------------------------------------------
# WHICH JULIA ENVIRONMENT
# ---------------------------------------------------------------------------
# The default `jac=:sym` needs EarthSciASTDiff, and `run-model-jl` CANNOT resolve
# it: run-model-jl develops the live EarthSciAST checkout (0.1.x line) while
# EarthSciASTDiff requires 0.9.x, and the two also disagree about EarthSciIO.
# That is a packaging fact, not something this script can paper over. So either
#
#   julia --project=/u/ctessum/reseact-esast-pin/env-sym run_reseact_adjoint.jl
#     with RESEACT_MODEL=/u/ctessum/reseact-esast-pin/corpus-pin/reseact.esm/reseact.esm
#
# (what tools/diag/adjoint_conus.sbatch and adjoint_clamp_retest.sbatch do), or
# run with RESEACT_ADJ_JAC=fd, which needs nothing beyond run-model-jl and is
# what every result before 2026-08-20 was produced with.
# ===========================================================================

const REPO = @__DIR__
const DEMO = lowercase(get(ENV, "RESEACT_DEMO", "adjoint"))
DEMO in ("adjoint", "forward", "both") ||
    error("RESEACT_DEMO must be adjoint|forward|both, got \"$DEMO\"")

# ---- Demonstration defaults ------------------------------------------------
# Small grid, short window: this is the configuration the acceptance checks in
# tools/adjoint_gradient.jl are green on, and it finishes. The production grid
# is 13x7x72 over a week (see run_reseact.jl); getting THERE is the blockers
# section above, not a matter of changing these numbers.
get!(ENV, "RESEACT_NLON", "6")
get!(ENV, "RESEACT_NLAT", "6")
get!(ENV, "RESEACT_NLEV", "8")

# `clamp_nonneg` OFF -- blocker 1. Not a preference; with it on the backward
# sweep goes non-finite at the third macro step. UNDER RETEST: the configuration
# that observation was made in had both the XLA:CPU race active and an FD block
# Jacobian, and the clamp's exact zeros are what make an FD Jacobian degenerate.
# See tools/diag/adjoint_clamp_retest.sbatch.
get!(ENV, "RESEACT_ADJ_CLAMP", "0")

# The exact, reverse-safe block Jacobian -- blocker 2, closed. See the header for
# what it replaced and what it measured. `fd` reproduces every pre-2026-08-20
# result; `ad` is exact but segfaults under a reverse sweep.
get!(ENV, "RESEACT_ADJ_JAC", "sym")

# Displace the base point off the PPM limiter switching surface (see above).
get!(ENV, "RESEACT_ADJ_UJITTER", "1e-1")
get!(ENV, "RESEACT_SENS_UJITTER", "1e-1")

# Three macro steps = 900 s of simulation. Enough for the Lie-Trotter chaining
# and the backward replay to be exercised more than once, which is what the
# ordering checks actually test.
get!(ENV, "RESEACT_ADJ_NMACRO", "3")
get!(ENV, "RESEACT_SENS_NMACRO", "3")

# Phase 2's objective, kept as the default so the two drivers are comparable.
get!(ENV, "RESEACT_ADJ_OBJ", "SuperFast.O3:surf")
get!(ENV, "RESEACT_SENS_OBJ", "SuperFast.O3:surf")

# For the forward arm, differentiate w.r.t. parameters whose sensitivity is
# known and non-zero -- `scale` and `g0` also satisfy the structural identity
# scale*dJ/dscale == g0*dJ/dg0, which the adjoint arm checks independently.
get!(ENV, "RESEACT_SENS_PARAM", "NEIRegrid.scale,Transport3D.tau_pblmix")

say(s) = (println(s); flush(stdout))

# Run one driver in its OWN module. Both drivers are standalone scripts that
# define `const LABEL` (and `REPO`, `MODEL`, `say`, ...) at top level with
# DIFFERENT values, so including both into `Main` is an invalid constant
# redefinition -- `both` mode would die on the second one. Separate modules also
# stop them clobbering this file's bindings. `Base.include(mod, path)` is used
# rather than a `module ... end` block because `module` is only legal at top
# level, and rather than `@eval` because `@__DIR__` does not resolve under it.
function run_arm(name::Symbol, path::AbstractString)
    isfile(path) || error("driver not found: $path")
    Base.include(Core.eval(Main, :(module $name end)), path)
end

say("="^75)
say("run_reseact_adjoint.jl -- simulation gradient demonstration")
say("="^75)
say("  mode          : $DEMO")
say("  grid          : $(ENV["RESEACT_NLON"])x$(ENV["RESEACT_NLAT"])x$(ENV["RESEACT_NLEV"])" *
    "   (production is 13x7x72 -- see the blockers in this file's header)")
say("  window        : $(ENV["RESEACT_ADJ_NMACRO"]) macro steps")
say("  objective     : $(ENV["RESEACT_ADJ_OBJ"])")
say("  clamp_nonneg  : $(ENV["RESEACT_ADJ_CLAMP"] == "0" ? "OFF (blocker 1)" : "ON")")
say("  block Jacobian: $(ENV["RESEACT_ADJ_JAC"])" *
    (ENV["RESEACT_ADJ_JAC"] == "sym" ? "  (analytic, exact, reverse-safe)" : ""))
say("  base point    : jittered $(ENV["RESEACT_ADJ_UJITTER"]) relative (PPM limiter; see header)")
say("")
say("  NOTE This demonstrates d(objective)/d(parameters) over a SHORT window on")
say("       a SMALL grid. It is not a week of CONUS, and the header of this file")
say("       says precisely what stands between the two.")
say("")

if DEMO in ("forward", "both")
    say("-"^75)
    say("FORWARD SENSITIVITY (Phase 2) -- cost proportional to n_params.")
    say("Differentiates the DEVICE while-loop trajectory, and validates against")
    say("central differences of the same solve. This is the independent check on")
    say("the adjoint; the two agree to ~3e-6, which is the host-vs-device float")
    say("divergence documented above, not an error in either.")
    say("-"^75)
    run_arm(:_ForwardArm, joinpath(REPO, "tools", "sensitivity_forward.jl"))
end

if DEMO in ("adjoint", "both")
    say("-"^75)
    say("DISCRETE ADJOINT (Phase 4) -- ALL 49 parameters in ONE backward sweep,")
    say("at a cost independent of the parameter count. Stages:")
    say("  fwd    forward pass, checkpointing (u, t, dt, forcing epoch) per macro")
    say("         step and recording the accepted (t, dt) sequence")
    say("  adj    backward sweep: chemistry VJP then transport VJP per macro step")
    say("         (the adjoint of a composition reverses it)")
    say("  ref    forward-mode reference over the SAME frozen step sequence, plus")
    say("         the per-step dot-product identity <lam,Jv> == <J'lam,v>")
    say("  fdtape central differences of the frozen-dt composed map, through the")
    say("         SAME compiled programs -- the confounder-free acceptance test")
    say("-"^75)
    run_arm(:_AdjointArm, joinpath(REPO, "tools", "adjoint_gradient.jl"))
end

say("")
say("="^75)
say("Done. For the full phased plan, the measured capability matrix and the")
say("retractions behind both, see DIFFERENTIABILITY_PLAN.md; HELPERS.md section 4")
say("has the same findings in context.")
say("="^75)
