#!/usr/bin/env julia
# ===========================================================================
# adjoint_gradient.jl -- Phase 4 of DIFFERENTIABILITY_PLAN.md:
# the TIME-LOOP ADJOINT DRIVER.
# ===========================================================================
# Turns Phase 3's PER-STEP VJP into dJ/dtheta for a WHOLE simulation window,
# for the FULL runtime parameter vector, in ONE backward sweep whose cost does
# not depend on how many parameters there are.
#
# THE ONE CONSTRAINT THAT SHAPES EVERYTHING. Reverse mode cannot cross a
# `stablehlo.while`, and the production macro step is two of them (the SSPRK43
# transport loop and the ROS23 chemistry loop in rx_traced_integrator.jl). So
# the adaptive time loop CANNOT stay on the device inside the differentiated
# program. Phase 3's answer is a per-step VJP that is straight-line traced code;
# Phase 4's answer is to lift the ENTIRE adaptive loop -- stage attempts, PI
# controller, accept/reject, the `clamp_nonneg` edit -- onto the HOST, so the
# only thing XLA is ever asked to differentiate is one step map at a fixed dt.
#
# `host_adaptive!` below is therefore a line-by-line host transcription of
# `RxTracedIntegrator.adaptive_solve`. Stage `ctl` measures how close the two
# are over one macro step, and the answer is NOT bit-identical: at 6x6x8 the
# accept/reject decisions agree exactly (5/0 transport, 117/3 chemistry) while
# the states differ by 8.8e-16 relative (transport) and 2.8e-10 (chemistry).
# The chemistry half runs the SAME Julia source in both arms, so that gap is
# XLA's: the same stage algebra compiled standalone and compiled inside a
# `stablehlo.while` body are not the same floating-point program. That is a
# floor on how well any host-lifted adjoint can match a device-loop forward
# sensitivity, and it is why the acceptance checks here are run against
# references computed with THESE compiled programs.
#
# THE TWO-LEVEL TAPE. `(u_k, t_k, dtT_k, dtC_k, forcing epoch, the accepted
# inner (t, dt) sequences)` is checkpointed at every MACRO step -- 681 kB/step
# at CONUS, i.e. 196 MB for 24 h and 1.4 GB for a week, which is cheap enough
# that no Griewank scheme is warranted (plan, Phase 4). The inner STATES are not
# stored; on the way back, macro step k is replayed from its checkpoint to
# rebuild the inner tape, which costs one extra forward pass over the window and
# bounds inner-tape memory by ONE macro step. RESEACT_ADJ_KEEPTAPE=1 keeps every
# inner tape from the forward pass instead.
#
# THE REPLAY IS SEQUENCE-FORCED, AND IT HAS TO BE. Re-running the adaptive
# controller from a checkpoint does NOT reproduce the forward pass: measured, a
# replay of macro step 2 took 95 accepts / 2 rejects where the forward pass took
# 92 / 1 -- same checkpoint, same theta, same forcing, same process. The compiled
# ROS23 step is not deterministic call to call. NOT, as recorded until 2026-08-12,
# because "the PI controller amplifies an ulp": in >400,000 calls there was NEVER a
# ULP-level difference. It is an XLA:CPU intra-op THREADING RACE -- the compiled
# step returns NaN in exactly the six dry-deposition species at one cell, `EEst`
# goes NaN with it, and the controller maps NaN to a REJECTION. So the extra
# rejects are spurious, not amplified roundoff. With one XLA thread it never
# happens at all. See tools/diag/README-nondet.md. So the forward pass records the accepted (t, dt) of
# every inner step -- 16 B each -- and the replay REPLAYS THAT SEQUENCE with the
# controller switched off. `tapes_for` then checks the replayed end state
# against the next checkpoint, which is the real faithfulness test.
#
# LIE-TROTTER ORDER. A macro step is transport THEN chemistry. The adjoint of a
# composition reverses it, so the backward sweep does CHEMISTRY first and then
# TRANSPORT within each macro step. Getting this backwards produces a gradient
# that is wrong by an amount no single-parameter comparison reliably catches,
# which is why criterion 2 (below) exists.
#
# THE CLAMP IS OUTSIDE THE STEP. `adaptive_solve(clamp_nonneg=true)` applies
# `max.(unew, 0)` to every ACCEPTED state, after the step and after EEst has
# been formed from the unclamped state. The step VJP knows nothing about it, so
# the driver owns its adjoint: a diagonal 0/1 factor `lambda .*= (u_raw .> 0)`
# applied before pushing lambda through that step's VJP. A clamped component
# contributes no gradient (plan section 4).
#
# AND WITH THE CLAMP ON, THE SWEEP GOES NON-FINITE. Measured at 6x6x8: the clamp
# bit on 18 of 939,744 (state, step) pairs, and partway back through the third
# macro step lambda acquired 312 non-finite entries -- 24 cells across all 13
# state groups at once. It is NOT reproducible from the recorded inputs: the
# PROBE stage re-runs that exact VJP (same u, lambda, t, dt, theta) and gets a
# finite answer, for four different seeds including all-ones and all-zeros, and
# the primal of that step is finite too. With `RESEACT_ADJ_CLAMP=0` the same
# sweep is finite throughout and passes every check here, and the clamp changes
# the primal objective by 6.7e-11 relative -- so the unclamped gradient is a
# gradient of, for practical purposes, the same trajectory. Not explained; see
# DIFFERENTIABILITY_PLAN.md's Phase 4 section.
#
# FORCING HAS NO DERIVATIVE. `theta` is `(p = the runtime scalars, bufs = the
# GEOS-FP forcing buffers)`. The VJP returns a gradient for both halves; only
# `p` is accumulated. That is not a shortcut -- meteorology is input data, not a
# function of the parameters, and we are not differentiating w.r.t. GEOS-FP. The
# forcing refresh is a clean boundary: it is replayed from the checkpointed epoch
# on the way back, so every macro step is adjointed against the same forcing
# fields it was integrated with.
#
# WHICH MAKES 111 OF THE 160 PRINTED COMPONENTS STRUCTURAL ZEROS -- READ THE
# TABLE ACCORDINGLY. Under esm 0.8.0 `p` held 49 scalars, every one of them a
# real knob. esm 1.0.0 redeclares each data-loader field as a PARAMETER carrying
# `update: {kind: "data", source: ..., from: {file_variable: ...}}` -- 42 GEOS-FP
# met fields and 69 NEI2016 species fluxes -- so `p` is now 160 long and the
# gradient table prints all of them.
#
# Those 111 report `theta=0  dJ/dtheta=0.000e+00`, and BOTH halves of that are
# placeholders rather than physics. The scalar slot is not how the field enters:
# the values arrive as ARRAYS through `merged_param` / the forcing buffers (the
# coupling carries them with `transform: param_to_var`), which is the `bufs` half
# we deliberately do not accumulate. theta=0 for `GEOSFP.T` is the proof -- the
# chemistry would not survive a 0 K temperature, so the slot is plainly unread.
#
# So `dJ/d(NEI2016Emis.flux_NO) = 0` does NOT mean surface O3 is insensitive to
# NO emissions. It is intensely sensitive to them: the same emissions field
# produces `dJ/d(NEIRegrid.scale) = -2.114`, the largest component in the table,
# because NEIRegrid.scale is a genuine scalar in the differentiated map while
# flux_NO is data. Only 20 of the 111 are even consumed by this model (15 of the
# 42 GEOS-FP fields, 5 of the 69 NEI fluxes -- NO, NO2, CO, ISOP, FORM); the
# other 91 are collection members nothing here reads, and their zeros are doubly
# uninformative. The 49 pre-migration scalars are still the calibratable set.
#
# WHAT IS VALIDATED, AND WHY EACH CHECK EXISTS
#   ctl    host adaptive loop vs traced `adaptive_solve` over one macro step.
#   ref    the SAME frozen step sequence differentiated in FORWARD mode, chained
#          by hand over the same tape with the same clamp masks, plus the
#          per-step dot-product identity <lam,Jv> == <J'lam,v> + <g,dth>.
#   fdtape central finite differences of the FROZEN-dt composed map, through the
#          SAME compiled single-step programs. This is the confounder-free
#          acceptance test: the discrete adjoint IS the derivative of that map.
#          Comparing against Phase 2's `sensitivity_forward.jl` instead carries
#          three separate confounders, each larger than the 1e-8 target -- it
#          also differentiates the controller's dt (the discrete adjoint holds
#          dt fixed), it runs the clamp, and its trajectory is the device
#          while-loop's rather than the host loop's.
#   ident  the structural identity scale*dJ/dscale == g0*dJ/dg0, which holds
#          because every NEI emission rate is proportional to scale*g0/delp and
#          g0 appears nowhere else. It is exact algebra between two components
#          of ONE gradient vector, needs no reference at all, and catches whole
#          classes of chaining error (a dropped step, a mis-ordered Lie-Trotter
#          half, a lambda applied at the wrong end) that a single-parameter
#          comparison cannot.
#
# BASE POINT. Do NOT validate at the default initial condition. Every SuperFast
# field in it is exactly spatially uniform, which puts u0 exactly ON the
# switching surface of essentially every PPM monotonicity limiter -- and the
# guard is QUADRATIC in the perturbation, so the usual fwd-vs-bwd kink test is
# blind to it (see the long note in tools/rx_adjoint_check.jl). Default here is
# RESEACT_ADJ_UJITTER=1e-1, with the same seeded stream that harness uses, so
# the base point is reproducible and shareable across drivers.
#
# Env: RESEACT_MODEL / LABEL / LON0 / LAT0 / NLON / NLAT / NLEV / T0 / MACRO_DT
#      / DT0T / DT0C / RXENV as in the runners, plus
#   RESEACT_ADJ_NMACRO    macro steps in the window (default 3)
#   RESEACT_ADJ_OBJ       "<state prefix>:<scope>", scope = surf|all|lev<K>
#                         (default "SuperFast.O3:surf" -- Phase 2's objective)
#   RESEACT_ADJ_UJITTER   relative jitter on the base point (default 1e-1)
#   RESEACT_ADJ_KEEPTAPE  1 = keep inner tapes from the forward pass (no replay)
#   RESEACT_ADJ_CLAMP     0 = no clamp_nonneg (REQUIRED today; see above)
#   RESEACT_ADJ_JAC       block Jacobian for Rosenbrock23: sym (default, exact,
#                         reverse-safe), fd (sqrt(eps), the old default), or
#                         ad (exact but SEGFAULTS under a reverse sweep)
#   RESEACT_ADJ_STAGES    subset of ctl,fwd,adj,ref,fdtape (default all but ctl)
#   RESEACT_ADJ_REFPARAM  parameters for the forward-mode reference
#                         (default NEIRegrid.scale,Transport3D.tau_pblmix,NEIRegrid.g0)
#   RESEACT_ADJ_CSV       write the full 49-component gradient here
# ===========================================================================
import Pkg
const REPO = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Printf, Statistics, Logging, Random
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using Reactant
const EA = EarthSciAST
const RX = Reactant
const EZ = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
const RXDIR   = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(CHEMDIR, "blockdiag_local.jl")); using .BlockDiag
include(joinpath(CHEMDIR, "block_jac.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

const MODEL    = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const LABEL    = get(ENV, "RESEACT_LABEL", "reseact-adj")
const T0       = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const MACRO_DT = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))
const NMACRO   = parse(Int,     get(ENV, "RESEACT_ADJ_NMACRO", "3"))
const DT0T     = parse(Float64, get(ENV, "RESEACT_DT0T", "15.0"))
const DT0C     = parse(Float64, get(ENV, "RESEACT_DT0C", "0.5"))
const OBJSPEC  = get(ENV, "RESEACT_ADJ_OBJ", "SuperFast.O3:surf")
const UJIT     = parse(Float64, get(ENV, "RESEACT_ADJ_UJITTER", "1e-1"))
const KEEPTAPE = get(ENV, "RESEACT_ADJ_KEEPTAPE", "0") == "1"
# `clamp_nonneg` is what the production runner does and what Phase 2 differentiated,
# so it is the default. RESEACT_ADJ_CLAMP=0 turns it off, which is the plan's
# "offer a gradient mode without it" (section 4) and is also the experiment that
# says whether the exact zeros it writes are what an infinite derivative is
# coming from.
const CLAMP = Ref(get(ENV, "RESEACT_ADJ_CLAMP", "1") == "1")
const ADJCSV   = get(ENV, "RESEACT_ADJ_CSV", "")
const REFPARAM = String.(split(get(ENV, "RESEACT_ADJ_REFPARAM",
                   "NEIRegrid.scale,Transport3D.tau_pblmix,NEIRegrid.g0"), ','))
const STAGES   = Set(String.(split(get(ENV, "RESEACT_ADJ_STAGES", "fwd,adj,ref,fdtape"), ',')))
want(s) = s in STAGES
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 13), nlat = _env("NLAT", 7),
                           nlev = _env("NLEV", 72))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const T_END    = T0 + NMACRO * MACRO_DT
const NDAYS    = forcing_days_for(T0, T_END)
const RTOL, ATOL_T, ATOL_C = 1e-4, 1e-6, 1e-9

say("=== $LABEL : TIME-LOOP ADJOINT ($(basename(MODEL))) " *
    "grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF T0=$(round(Int,T0)) " *
    "nmacro=$NMACRO macro_dt=$(round(Int,MACRO_DT)) ujitter=$UJIT ===")
say("    stages: " * join(sort(collect(STAGES)), ", "))

# --------------------------------------------------------------------------- #
# 1. Build -- identical to run_reseact_reactant.jl / sensitivity_forward.jl.
# --------------------------------------------------------------------------- #
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing
# escaped from the build block below so the SYMBOLIC JACOBIAN can be prepared
# from the same split part, with the same overrides, that part 2's RHS was built
# from. Any drift between the two would be a silently wrong Jacobian.
splitparts = nothing; merged_const = nothing; ov = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff
    global splitparts, merged_const, ov
    file = EA.load_path(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in splitparts]
    f0 = reseact_forcing(CHEMDIR; ndays = NDAYS)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld; discrete[k] = prov
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
say(@sprintf("BUILD %.2f s   nstates=%d  nparams=%d  discrete_providers=%d",
             time() - tb, length(u0), length(p), length(discrete)))

# Seed m(0) from the real GEOS-FP surface pressure (species-major state).
let dp0 = hydrostatic_dp(merged_param, ff.const_arrays, T0; slice = SLICE)
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dp0(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
                      parse(Int, mm.captures[3]))
    end
end

include(joinpath(RXDIR, "rx_native_patch.jl"))
include(joinpath(RXDIR, "rx_traced_integrator.jl"))
const RTI = RxTracedIntegrator
# unconditional: the gather plan needs only Reactant, and a `using` inside a
# conditional block is one more thing that can only fail at runtime.
include(joinpath(RXDIR, "rx_sym_block_jac.jl"))
using .RxSymBlockJac

host_bufs = [EA.forcing_buffers(fo[i]) for i in 1:2]
g4 = [EA.rhs_with_buffers(fo[i]) for i in 1:2]
PERM = cellmajor_perm(var_map)
const NS = PERM.NS; const NC = PERM.NC; const N = PERM.N
const MASKS = RTI.species_masks(var_map, NS, NC)
say("  NS=$NS species, NC=$NC cells, N=$N states")

# --------------------------------------------------------------------------- #
# 1b. BLOCKER 2 -- the block Jacobian Rosenbrock23 needs.
#
# `jac = :fd` (the historical default) finite-differences the chemistry RHS once
# per species to build each cell block. It costs NS extra RHS evaluations per
# attempt and, worse, it is only accurate to sqrt(eps): it degraded the adjoint's
# dot-product identity <lam,Jv> == <J'lam,v> to 1.1e-6 -- exactly the accuracy
# asked of the whole gradient, with no margin left over. `jac = :ad` is exact but
# it is a nested forward-over-reverse, and Enzyme-MLIR segfaults in
# `AutoDiffCallRev::createReverseModeAdjoint` when a reverse sweep crosses it.
#
# `jac = :sym` is EarthSciASTDiff's ANALYTIC Jacobian: a separate generated
# "band model" that emits the nonzero entries directly, gathered into cell blocks
# by RxSymBlockJac. It contains no nested AD, so reverse mode crosses it, and it
# is exact -- measured 5.0e-16 max entry-wise against :ad, and it restores the
# dot-product identity to 5.3e-16. It also compiles 4x cheaper than :fd
# (12.0 s vs 49.9 s standalone; 109k lines of MLIR vs 323k).
#
# The band model carries ITS OWN forcing buffers (15 of them), so meteorology
# stays a runtime operand there too rather than being baked into the program.
# They are built from the same `merged_param` dict the RHS uses, which is what
# makes `refresh_forcing` reach them.
const JACMODE = Symbol(get(ENV, "RESEACT_ADJ_JAC", "sym"))
JACMODE in (:sym, :fd, :ad) || error("RESEACT_ADJ_JAC must be sym, fd or ad")
const SYMJAC = JACMODE === :sym
PLAN = gjbJ = host_bufsJ = dev_bufsJ = nothing
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
    jacE.oop || error("jac=:sym: the band model came back IN-PLACE; it captures " *
                      "host scratch per node and cannot be traced")
    # `runner_names` costs nothing and would otherwise be a silent catastrophe:
    # two independent builds, two var maps, and a plan that indexes by POSITION.
    PLAN = block_jac_plan(jacE;
                          runner_names = first.(sort(collect(var_map), by = last)))
    gjbJ = EA.rhs_with_buffers(jacE.fJ!)
    host_bufsJ = EA.forcing_buffers(jacE.fJ!)
    say("  $PLAN   band buffers=$(length(host_bufsJ))")
    # the plan is pure index algebra, so it is checkable on the host against the
    # sparse Jacobian EarthSciASTDiff assembles itself -- one evaluation, and it
    # is the only thing standing between a transposed gather and a plausible
    # gradient that is wrong everywhere.
    let w = validate_plan(PLAN, jacE, u0, p, T0; gjb = gjbJ, bufs = host_bufsJ)
        say(@sprintf("  plan vs the host JacobianEvaluator: worst relative %.3e  %s",
                     w, w <= 1e-12 ? "PASS" : "FAIL"))
        w <= 1e-12 || error("jac=:sym: the gather plan does not reproduce the host Jacobian")
    end
end

# The RHS with its differentiable payload an EXPLICIT argument: the production
# call site hides p and the forcing buffers in a closure, and a closure is
# opaque to Enzyme.
gT(u, th, t) = g4[1](u, th.p, t, th.bufs)
gC(u, th, t) = g4[2](u, th.p, t, th.bufs)

dev_bufs = [map(RX.ConcreteRArray, host_bufs[i]) for i in 1:2]
SYMJAC && (dev_bufsJ = map(RX.ConcreteRArray, host_bufsJ))
function push_forcing!()
    for i in 1:2; EA.sync_forcing!(dev_bufs[i], EA.forcing_buffers(fo[i])); end
    # the band model has its own buffer set; it is fed from the same
    # `merged_param` arrays, so `refresh_forcing` moves it too -- but only if
    # this line is here. Miss it and the Jacobian freezes at the T0 epoch while
    # the RHS moves, which is a wrong Jacobian that still converges.
    SYMJAC && EA.sync_forcing!(dev_bufsJ, EA.forcing_buffers(jacE.fJ!))
    return nothing
end
push_forcing!()
_devp(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
const PRd = _devp(p)
const THT = (p = PRd, bufs = dev_bufs[1])
# The chemistry theta grows a THIRD field under jac=:sym. Every construction of
# one goes through `thC` so the shape cannot drift between primal, VJP and JVP --
# a mismatch there is an Enzyme activity error at best and a dropped dJ/dp at
# worst. `theta` is passed to the band model APART from the closure for the same
# reason: a closure over `p` would make dJ/dp through the JACOBIAN invisible.
thC(pp, bb, bbJ) = SYMJAC ? (p = pp, bufs = bb, bufsJ = bbJ) : (p = pp, bufs = bb)
const THC = thC(PRd, dev_bufs[2], dev_bufsJ)
# u -> the NS x NS matrix of length-NC cell-block diagonals, all on device.
gJ = SYMJAC ? ((u, th, t) -> block_jac(PLAN, gjbJ(gather_uj(PLAN, u), th.p, t, th.bufsJ))) :
              nothing

# every entry of `p` must be a runtime scalar; the gradient vector IS `keys(p)`.
const PNAMES = Tuple(sort(collect(keys(p))))
for k in PNAMES
    getfield(p, k) isa Number ||
        error("p.$k is a $(typeof(getfield(p, k))), not a scalar -- the gradient " *
              "vector this driver reports is the scalar runtime parameter vector")
end
say("  parameter vector: $(length(PNAMES)) runtime scalars")

# --------------------------------------------------------------------------- #
# 2. Base point. See the header: the default IC is degenerate for validation.
# --------------------------------------------------------------------------- #
const UBASE = let u = copy(u0)
    if UJIT > 0
        # the SAME stream tools/rx_adjoint_check.jl uses, so "jittered base
        # point" means the same point in both harnesses
        u .*= (1 .+ UJIT .* randn(Random.MersenneTwister(31337), N))
        say(@sprintf("  base point jittered by %.0e relative (seed 31337)", UJIT))
    else
        say("  base point = the DEFAULT initial condition (degenerate; see header)")
    end
    u
end

# --------------------------------------------------------------------------- #
# 3. The objective, as a weight vector: J(u) = dot(w, u), grad J = w.
#    Identical to sensitivity_forward.jl's, so the two drivers score the same
#    scalar. A nonlinear objective needs only its gradient swapped in here.
# --------------------------------------------------------------------------- #
function objective_weights(var_map, spec::AbstractString)
    parts = split(spec, ':')
    length(parts) == 2 || error("objective must be \"<prefix>:<scope>\", got \"$spec\"")
    pre, scope = String(parts[1]), String(parts[2])
    rx = Regex("^" * replace(pre, "." => "\\.") * "\\[(\\d+),(\\d+),(\\d+)\\]\$")
    keep = if scope == "all"; _ -> true
    elseif scope == "surf"; k -> k == 1
    elseif startswith(scope, "lev"); kk = parse(Int, scope[4:end]); k -> k == kk
    else error("unknown objective scope \"$scope\"") end
    idxs = Int[]
    for (nm, idx) in var_map
        mm = match(rx, nm)
        mm === nothing && continue
        keep(parse(Int, mm.captures[3])) && push!(idxs, idx)
    end
    isempty(idxs) && error("objective \"$spec\" selected no states")
    w = zeros(Float64, length(var_map)); w[idxs] .= 1 / length(idxs)
    return w, length(idxs)
end
const _OBJ = objective_weights(var_map, OBJSPEC)
const WOBJ = _OBJ[1]
say("  objective: mean over $(_OBJ[2]) states matching $OBJSPEC")

# --------------------------------------------------------------------------- #
# 4. The compiled pieces. Every one of them is a SINGLE STEP at a fixed dt --
#    there is no time loop on the device anywhere in this driver.
# --------------------------------------------------------------------------- #
ssp_step(u, th, t, dt) = RTI.ssprk43_step_unrolled((uu, tt) -> gT(uu, th, tt),
                                                   u, t, dt, ATOL_T, RTOL)
ros_step(u, th, t, dt) = RTI.ros23_step((uu, tt) -> gC(uu, th, tt), u, t, dt,
                                        NS, NC, MASKS, ATOL_C, RTOL;
                                        unrolled = true, jac = JACMODE,
                                        symjac = SYMJAC ? ((uu, tt) -> gJ(uu, th, tt)) : nothing)
ssp_vjp(u, th, lam, t, dt) = RTI.ssprk43_step_vjp(gT, u, th, t, dt, lam, ATOL_T, RTOL)
ros_vjp(u, th, lam, t, dt) = RTI.ros23_step_vjp(gC, u, th, t, dt, lam,
                                                NS, NC, MASKS, ATOL_C, RTOL;
                                                jac = JACMODE, gj = gJ)
# The JVPs return the WHOLE `Enzyme.autodiff` result, not `r[1]`. MEASURED, and
# it is a trap: `RxTracedIntegrator.{ros23,ssprk43}_step_jvp` hand back `r[1]`,
# and on the REAL model that slot is the PRIMAL, not the tangent -- a chained
# forward reference built on it silently returns the state instead of its
# derivative (||du|| came back as 2.3e4 for a parameter that does not appear in
# the transport RHS at all, and the dot-product identity read rel=1.6). The slot
# is resolved below by the only reliable test there is: a ZERO seed must give an
# exactly zero tangent.
ssp_jvp(u, du, th, dth, t, dt) = EZ.autodiff(EZ.Forward, RTI._ssprk43_out_c, EZ.Duplicated,
    EZ.Const(gT), EZ.Duplicated(u, du), EZ.Duplicated(th, dth),
    EZ.Const(t), EZ.Const(dt), EZ.Const(ATOL_T), EZ.Const(RTOL))
ros_jvp(u, du, th, dth, t, dt) = EZ.autodiff(EZ.Forward, RTI._ros23_out_c, EZ.Duplicated,
    EZ.Const(gC), EZ.Duplicated(u, du), EZ.Duplicated(th, dth),
    EZ.Const(t), EZ.Const(dt), EZ.Const(NS), EZ.Const(NC), EZ.Const(MASKS),
    EZ.Const(ATOL_C), EZ.Const(RTOL), EZ.Const(JACMODE), EZ.Const(gJ))

const U_R  = RX.ConcreteRArray(copy(UBASE))
const T_R  = RX.ConcreteRNumber(T0)
const DTT_R = RX.ConcreteRNumber(DT0T)
const DTC_R = RX.ConcreteRNumber(DT0C)
const LAM_R = RX.ConcreteRArray(copy(WOBJ))

# `@compile` needs a literal call expression, so the timing wrapper takes a thunk.
function timed_compile(nm::AbstractString, thunk)
    t0 = time(); c = thunk(); say(@sprintf("  @compile %-12s %8.1f s", nm, time() - t0))
    return c
end
# BLOCKER 4 WORKAROUND -- wired in 2026-08-19, and it is not optional at CONUS.
# XLA:CPU has a data race in its intra-op parallel execution that intermittently
# returns non-finite values from the chemistry program (~1% of calls at 4 threads;
# 0 of 400,000 at one thread). It killed the FIRST CONUS backward sweep outright:
# the fixed-sequence replay of macro step 3 came back NaN and the driver aborted.
# `xla_cpu_prefer_vector_width=128` is the measured fix (0 of 20,000 against a
# baseline of 4 and 4).
#
# TWO CLAIMS THAT WERE IN THIS COMMENT ARE REFUTED by the first completed CONUS
# adjoint (slurm 10015169, 2026-08-19); do not restore them:
#
#  * "at no wall-time cost" -- WRONG, it is worth **3.14x on the forward pass**
#    (247.12 s -> 78.61 s on the identical configuration). Per-attempt cost is
#    unchanged (0.436 -> 0.449 s); what vanishes is SPURIOUS WORK. A faulting call
#    NaNs `EEst`, `host_adaptive!` maps that to a rejection, and each spurious
#    rejection ALSO shrinks dt -- so the controller then grinds out many more,
#    smaller accepted steps. Rejections 24.5% -> 2.9%, accepted 428 -> 170.
#    Corollary CHASED AND REFUTED (slurm 10017939): this does NOT carry over to
#    the traced runner's real trajectory. Re-running its 24 h CONUS window with
#    the workaround on gave solve 6,301.7 s against 5,923 s -- 6.4% SLOWER, not
#    3x faster. The 3.14x is a property of the JITTERED base point below (~189
#    chemistry attempts/macro step vs ~50 on a real trajectory, median dt 1.31 s,
#    controller at its limits, so one NaN cascades). The week baselines stand.
#
#  * "up to 1.6e-6 relative ... a new floor" -- UNDERSTATED, and the wrong model.
#    J moved 4.0e-5 relative (39.6115434688274 -> 39.6131238549521), 25x that
#    figure. It is not a rounding shift: the racy run integrated a DIFFERENT
#    TRAJECTORY, via those spurious rejections. The race was BIASING results, not
#    merely NaN-ing them occasionally.
#
# Reproduce the fault: tools/diag/README-nondet.md.
#
# `compile_options` REPLACES every other compile option (Reactant Macros.jl:7),
# so `sync = true` has to be set inside it rather than alongside it.
const XLAFIX = get(ENV, "RESEACT_ADJ_XLAFIX", "1") == "1"
const COPTS = XLAFIX ?
    RX.CompileOptions(; sync = true,
                      xla_debug_options = (; xla_cpu_prefer_vector_width = 128)) :
    RX.CompileOptions(; sync = true)

say("\n---- compiling single-step programs (no while region in any of them) ----")
say("     XLA:CPU race workaround (blocker 4): " *
    (XLAFIX ? "ON  xla_cpu_prefer_vector_width=128" : "OFF -- expect intermittent NaN"))
CSSP = timed_compile("ssp_step", () -> RX.@compile compile_options=COPTS ssp_step(U_R, THT, T_R, DTT_R))
CROS = timed_compile("ros_step", () -> RX.@compile compile_options=COPTS ros_step(U_R, THC, T_R, DTC_R))
CSSPV = want("adj") ?
    timed_compile("ssp_vjp", () -> RX.@compile compile_options=COPTS ssp_vjp(U_R, THT, LAM_R, T_R, DTT_R)) : nothing
CROSV = want("adj") ?
    timed_compile("ros_vjp", () -> RX.@compile compile_options=COPTS ros_vjp(U_R, THC, LAM_R, T_R, DTC_R)) : nothing

# --------------------------------------------------------------------------- #
# 5. The host adaptive loop -- a line-by-line transcription of
#    RxTracedIntegrator.adaptive_solve. Stage `ctl` proves it is exact.
#    `tape` collects, for every ACCEPTED step, the state that ENTERED it, its
#    (t, dt), and the 0/1 clamp mask applied to its OUTPUT. Rejected attempts
#    do not move the state and so contribute nothing to the adjoint.
# --------------------------------------------------------------------------- #
struct StepRec
    u::Vector{Float64}
    t::Float64
    dt::Float64
    mask::BitVector
end

const StepSeq = Vector{Tuple{Float64,Float64}}      # the accepted (t, dt) sequence

function host_adaptive!(cstep, uh::Vector{Float64}, t0::Float64, t1::Float64,
                        dt0::Float64, ctrl::RTI.PICtrl, TH;
                        clamp_nonneg::Bool = true, tape::Union{Nothing,Vector{StepRec}} = nothing,
                        seq::Union{Nothing,StepSeq} = nothing,
                        maxiters::Int = 20000)
    beta1 = ctrl.beta1; beta2 = ctrl.beta2
    invqmax = 1.0 / ctrl.qmax; invqmin = 1.0 / ctrl.qmin
    gamma = ctrl.gamma
    qsmin = ctrl.qsteady_min; qsmax = ctrl.qsteady_max
    qoldinit = ctrl.qoldinit
    u = copy(uh); t = t0; dt = dt0; qold = qoldinit
    nacc = 0; nrej = 0; iters = 0
    tlim = t1 - 1.0e-9
    while (t < tlim) && (iters < maxiters)
        dtc = min(dt, t1 - t)
        r = cstep(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t), RX.ConcreteRNumber(dtc))
        raw = Array(r[1]); ee = Float64(r[2])
        EEst = isnan(ee) ? 1.0e10 : ee
        q11 = max(EEst, 1.0e-35)^beta1
        q = q11 / qold^beta2
        q = max(invqmax, min(invqmin, q / gamma))
        accept = EEst <= 1.0
        insteady = (qsmin <= q) & (q <= qsmax)
        qa = insteady ? one(q) : q
        if accept
            unew = clamp_nonneg ? max.(raw, 0.0) : raw
            if tape !== nothing
                push!(tape, StepRec(copy(u), t, dtc,
                                    clamp_nonneg ? BitVector(raw .> 0.0) : trues(length(raw))))
            end
            seq !== nothing && push!(seq, (t, dtc))
            u = unew
            t = t + dtc
            dt = dtc / qa
            qold = max(EEst, qoldinit)
            nacc += 1
        else
            dt = dtc / min(invqmin, q11 / gamma)
            nrej += 1
        end
        iters += 1
    end
    iters >= maxiters && error("host_adaptive! hit maxiters at t=$t (t1=$t1)")
    return u, t, dt, nacc, nrej
end

# --------------------------------------------------------------------------- #
# 6. The macro-step lattice and the forcing epochs -- run_reseact_reactant.jl's
#    fencepost, kept verbatim so this differentiates the window that runner
#    would simulate.
# --------------------------------------------------------------------------- #
const TSTOPS = sort!(unique!(Float64[t for prov in values(discrete)
                                     for t in EarthSciIO.refresh_times(prov)]))
const FSTOPS = Set(round(t; digits = 6) for t in TSTOPS if T0 + 1e-6 < t <= T_END + 1e-6)
const STOPS  = sort!(unique!(vcat(collect((T0 + MACRO_DT):MACRO_DT:(T_END - 1e-9)),
                                  collect(FSTOPS), Float64[T_END])))
say(@sprintf("  %d macro stops over [%.0f, %.0f] s, %d of them GEOS-FP boundaries",
             length(STOPS), T0, T_END, length(FSTOPS)))

function refresh_forcing(t)
    for (k, prov) in discrete
        merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), dms)
    push_forcing!()
end

# One macro step, Lie-Trotter: transport over [t0,t1] THEN chemistry over the
# same interval. Records the ACCEPTED (t, dt) sequence of each half -- 16 B per
# inner step, which is nothing next to the 681 kB state, and it is what makes
# the backward replay reproducible (see `replay_fixed`).
function macro_step(u::Vector{Float64}, t0::Float64, t1::Float64,
                    dtT::Float64, dtC::Float64; record::Bool = false)
    tpT = record ? StepRec[] : nothing
    tpC = record ? StepRec[] : nothing
    sT = StepSeq(); sC = StepSeq()
    uT, _, dtTe, naT, nrT = host_adaptive!(CSSP, u,  t0, t1, dtT, RTI.pictrl_ssprk43(), THT;
                                           tape = tpT, seq = sT, clamp_nonneg = CLAMP[])
    uC, _, dtCe, naC, nrC = host_adaptive!(CROS, uT, t0, t1, dtC, RTI.pictrl_ros23(), THC;
                                           tape = tpC, seq = sC, clamp_nonneg = CLAMP[])
    return uC, dtTe, dtCe, (naT, nrT, naC, nrC), tpT, tpC, sT, sC
end

struct Ckpt
    u::Vector{Float64}
    t::Float64
    dtT::Float64
    dtC::Float64
    epoch::Float64
    seqT::StepSeq
    seqC::StepSeq
end

# ---- the FIXED-SEQUENCE replay ------------------------------------------------
# THE REASON THIS EXISTS, MEASURED. Re-running the adaptive controller from a
# checkpoint does NOT reproduce the forward pass. In one run, macro step 2
# replayed as 95 accepts / 2 rejects where the forward pass took 92 / 1 -- same
# checkpoint, same theta, same forcing, same process. The compiled ROS23 step is
# not deterministic call to call -- an XLA:CPU threading race, NOT reassociation
# and NOT ulp amplification (both were recorded here until 2026-08-12 and both are
# wrong; there was never a ULP-level difference in >400,000 calls). The race NaNs
# `EEst`, and the controller maps NaN to a rejection. Phase 3's unstable counts
# ("116/3 vs 120/4") are the same fault, not a resolution limit.
#
# So the backward sweep must NOT re-derive the step sequence. The forward pass
# records the accepted (t, dt) of every inner step and the replay REPLAYS THAT,
# with the controller switched off entirely. This is also the honest statement of
# what a discrete adjoint differentiates: a FIXED composition of fixed-dt step
# maps, with the controller's decisions treated as the discrete choices they are.
function replay_fixed(u0::Vector{Float64}, seqT::StepSeq, seqC::StepSeq;
                      TH_T = THT, TH_C = THC, record::Bool = true)
    u = copy(u0)
    tpT = StepRec[]; tpC = StepRec[]
    for (cstep, TH, sq, tp) in ((CSSP, TH_T, seqT, tpT), (CROS, TH_C, seqC, tpC))
        for (tt, dd) in sq
            r = cstep(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(tt), RX.ConcreteRNumber(dd))
            raw = Array(r[1])
            record && push!(tp, StepRec(copy(u), tt, dd,
                                        CLAMP[] ? BitVector(raw .> 0.0) : trues(length(raw))))
            u = CLAMP[] ? max.(raw, 0.0) : raw
        end
    end
    return u, tpT, tpC
end

"""
    forward_pass(; record) -> (u_end, ckpts, tapes, counts, seconds)

The macro-step loop, checkpointing (u_k, t_k, dtT_k, dtC_k, forcing epoch, and
the accepted inner (t, dt) sequences) at EVERY macro step. `record=true` also
keeps the inner state tapes (no replay needed on the way back); `record=false`
is the memory-lean default.
"""
function forward_pass(; record::Bool = false)
    refresh_forcing(T0); epoch = T0
    u = copy(UBASE); tcur = T0; dtT = DT0T; dtC = DT0C
    ckpts = Ckpt[]; tapes = Any[]; counts = NTuple{4,Int}[]
    tstart = time()
    for tnext in STOPS
        tnext <= tcur + 1e-9 && continue
        ustart = copy(u)
        u, dtT, dtC, cnt, tpT, tpC, sT, sC = macro_step(u, tcur, tnext, dtT, dtC; record = record)
        push!(ckpts, Ckpt(ustart, tcur, dtT, dtC, epoch, sT, sC))
        push!(counts, cnt); record && push!(tapes, (tpT, tpC))
        tcur = tnext
        if round(tnext; digits = 6) in FSTOPS
            refresh_forcing(tnext); epoch = tnext
        end
    end
    return u, ckpts, tapes, counts, time() - tstart
end

steps_sig(counts) = join((@sprintf("%d/%d,%d/%d", c[1], c[2], c[3], c[4]) for c in counts), " ")

# --------------------------------------------------------------------------- #
# 8. Stage `fwd` -- the forward pass and its checkpoints.
# --------------------------------------------------------------------------- #
UEND = Float64[]; CKPTS = Ckpt[]; TAPES = Any[]; COUNTS = NTuple{4,Int}[]
T_FWD = 0.0
if want("fwd")
    say("\n---- FWD: the macro-step loop, checkpointed ----")
    UEND, CKPTS, TAPES, COUNTS, T_FWD = forward_pass(; record = KEEPTAPE)
    JVAL = dot(WOBJ, UEND)
    ninner = sum(c[1] + c[3] for c in COUNTS)
    say(@sprintf("  forward pass %.2f s over %d macro steps (%.3f s/macro step)",
                 T_FWD, length(CKPTS), T_FWD / length(CKPTS)))
    say("  accept/reject per macro step (T then C): $(steps_sig(COUNTS))")
    say(@sprintf("  %d accepted inner steps total (%d transport, %d chemistry)",
                 ninner, sum(c[1] for c in COUNTS), sum(c[3] for c in COUNTS)))
    say(@sprintf("  checkpoint cost: %d states x 8 B = %.1f kB/macro step, %.1f MB total",
                 N, N * 8 / 1024, length(CKPTS) * N * 8 / 1024^2))
    say(@sprintf("  J = %.15g", JVAL))
    all(isfinite, UEND) || error("non-finite state at the end of the window")
end

# --------------------------------------------------------------------------- #
# 9. Stage `adj` -- the backward sweep.
#    lambda starts at grad J(u_end) = w. For k = N..1: replay macro step k from
#    its checkpoint to rebuild the inner tapes, then walk CHEMISTRY backwards
#    and TRANSPORT backwards (the reverse of the Lie-Trotter composition).
# --------------------------------------------------------------------------- #
gacc = Dict{Symbol,Float64}(k => 0.0 for k in PNAMES)
NCLAMPED = 0                                     # how often the clamp actually bites
BADREC = Ref{Any}(nothing)                       # the FIRST step that goes non-finite
NVJP_RETRIES = 0                                 # how often the flaky reverse pass was re-issued
const VJP_MAXRETRY = parse(Int, get(ENV, "RESEACT_ADJ_VJP_RETRY", "4"))

# state index -> species/group name, for attributing a non-finite lambda
const GROUP_OF = let v = fill("?", N)
    for (nm, idx) in var_map; v[idx] = replace(nm, r"\[.*$" => ""); end
    v
end

function backward_stage!(cvjp, tape::Vector{StepRec}, lam::Vector{Float64}, TH,
                         label::AbstractString)
    for j in length(tape):-1:1
        e = tape[j]
        global NCLAMPED += count(!, e.mask)
        lam = lam .* e.mask                      # adjoint of max.(u,0) after the step
        nin = count(!isfinite, lam)
        UD = RX.ConcreteRArray(e.u); LD = RX.ConcreteRArray(lam)
        TD = RX.ConcreteRNumber(e.t); DD = RX.ConcreteRNumber(e.dt)
        r = cvjp(UD, TH, LD, TD, DD)
        lout = Array(r[1])
        gp = r[2].p                              # r[2].bufs is dJ/d(GEOS-FP): discarded
        gbad = count(k -> !isfinite(Float64(getfield(gp, k))), PNAMES)
        # RE-ISSUING THE IDENTICAL CALL IS A LEGITIMATE FIX HERE, and only
        # because the fault is measured to be nondeterministic: the compiled
        # reverse program intermittently returns non-finite entries from a
        # finite lambda at a state whose primal is finite, and the PROBE stage
        # re-runs that exact call -- same u, lambda, t, dt, theta -- and gets a
        # finite answer. So this retries the SAME inputs rather than perturbing
        # anything, and every retry is counted and reported so the fault cannot
        # be hidden by the workaround.
        nretry = 0
        while (count(!isfinite, lout) > nin || gbad > 0) && nretry < VJP_MAXRETRY
            nretry += 1
            r = cvjp(UD, TH, LD, TD, DD)
            lout = Array(r[1]); gp = r[2].p
            gbad = count(k -> !isfinite(Float64(getfield(gp, k))), PNAMES)
        end
        global NVJP_RETRIES += nretry
        # A gradient that has gone non-finite is worthless from here on, and the
        # step it FIRST happened on is the only useful piece of evidence there
        # is -- record everything needed to reproduce that one VJP.
        if BADREC[] === nothing && (nout = count(!isfinite, lout); nout > nin || gbad > 0)
            BADREC[] = (label = String(label), j = j, n = length(tape), t = e.t,
                        dt = e.dt, u = copy(e.u), lam = copy(lam))
            say(@sprintf("  FIRST NON-FINITE: %s step %d of %d (t=%.6f dt=%.6g): lambda_in bad %d -> lambda_out bad %d, grad_theta bad %d of %d",
                         label, j, length(tape), e.t, e.dt, nin, nout, gbad, length(PNAMES)))
            byg = Dict{String,Int}()
            for i in 1:N
                isfinite(lout[i]) && continue
                byg[GROUP_OF[i]] = get(byg, GROUP_OF[i], 0) + 1
            end
            say("    non-finite lambda by state group: " *
                join(["$(q.first) $(q.second)" for q in sort(collect(byg); by = last, rev = true)], ", "))
            nzu = e.u[e.u .!= 0.0]
            say(@sprintf("    u entering that step: %d exactly 0, %d negative, min|u|(nonzero)=%.3e, max|u|=%.3e",
                         count(==(0.0), e.u), count(<(0.0), e.u),
                         isempty(nzu) ? 0.0 : minimum(abs.(nzu)), maximum(abs.(e.u))))
        end
        lam = lout
        for k in PNAMES; gacc[k] += Float64(getfield(gp, k)); end
    end
    return lam
end

"""
    tapes_for(k) -> (transport tape, chemistry tape, replay seconds)

The inner tape of macro step `k`, either kept from the forward pass or rebuilt by
a FIXED-SEQUENCE replay from its checkpoint (see `replay_fixed` -- re-running the
controller does not reproduce the forward pass). The replayed end state is
checked against the next checkpoint, which is the real faithfulness test: same
states, in the same order, as the forward pass produced.
"""
REPLAY_MAXREL = 0.0
function tapes_for(k::Int)
    KEEPTAPE && return (TAPES[k][1], TAPES[k][2], 0.0)
    ck = CKPTS[k]
    trep = time()
    uend, a, b = replay_fixed(ck.u, ck.seqT, ck.seqC)
    el = time() - trep
    ref = k < length(CKPTS) ? CKPTS[k + 1].u : UEND
    if !isempty(ref)
        d = maximum(abs.(uend .- ref) ./ max.(abs.(ref), 1e-30))
        global REPLAY_MAXREL = max(REPLAY_MAXREL, d)
        d < 1e-6 || error("fixed-sequence replay of macro step $k lands $(d) relative " *
                          "away from the checkpointed end state -- the tape is not the " *
                          "trajectory the forward pass took")
    end
    return (a, b, el)
end

function backward_sweep(lam0::Vector{Float64})
    lam = copy(lam0)
    t_replay = 0.0; nvjp = 0
    for k in length(CKPTS):-1:1
        ck = CKPTS[k]
        refresh_forcing(ck.epoch)                # forcing replayed from the checkpoint
        tpT, tpC, el = tapes_for(k)
        t_replay += el
        lam = backward_stage!(CROSV, tpC, lam, THC, "chem[macro $k]")   # chemistry LAST forward => FIRST back
        lam = backward_stage!(CSSPV, tpT, lam, THT, "transport[macro $k]")
        nvjp += length(tpC) + length(tpT)
        # FLUSH, not decoration. Julia block-buffers a file stdout at 64 kB and
        # these lines are ~80 B, so a 576-macro-step sweep emits ~46 kB and
        # NOTHING reaches the log until the sweep ends and the next `say` flushes.
        # Observed on slurm 10044327: four hours of a live, 3-core-busy backward
        # sweep looking exactly like a hang.
        @printf("  macro step %2d  t=%.0f  vjps: %d chem + %d transport   ||lambda||=%.6e\n",
                k, ck.t, length(tpC), length(tpT), norm(lam))
        flush(stdout)
    end
    return lam, t_replay, nvjp
end

T_BWD = 0.0; T_REPLAY = 0.0; NVJP = 0
if want("adj")
    say("\n---- ADJ: the backward sweep (ONE sweep, all $(length(PNAMES)) parameters) ----")
    tstart = time()
    LAM_END, T_REPLAY, NVJP = backward_sweep(WOBJ)
    T_BWD = time() - tstart
    nm_ = length(CKPTS)
    say(@sprintf("  backward sweep %.2f s over %d macro steps (%.3f s/macro step), %d VJP calls",
                 T_BWD, nm_, T_BWD / nm_, NVJP))
    say(@sprintf("  of which replay-of-the-primal %.2f s (%.3f s/macro step) and VJPs %.2f s (%.3f s/macro step, %.4f s/VJP)",
                 T_REPLAY, T_REPLAY / nm_, T_BWD - T_REPLAY, (T_BWD - T_REPLAY) / nm_,
                 NVJP > 0 ? (T_BWD - T_REPLAY) / NVJP : 0.0))
    say(@sprintf("  clamp bit on %d of %d (state, accepted-step) pairs on the tape (%.4f%%)",
                 NCLAMPED, NVJP * N, 100 * NCLAMPED / max(NVJP * N, 1)))
    KEEPTAPE || say(@sprintf("  fixed-sequence replay lands within %.3e relative of every checkpoint",
                             REPLAY_MAXREL))
    say(@sprintf("  flaky-reverse retries: %d over %d VJP calls (%.3f%%)%s",
                 NVJP_RETRIES, NVJP, 100 * NVJP_RETRIES / max(NVJP, 1),
                 BADREC[] === nothing ? "" : " -- see PROBE below"))
    T_FWD > 0 && say(@sprintf("  COST RATIO backward/forward = %.2f  (VJP-only %.2f)  [per macro step %.3f s vs %.3f s]",
                              T_BWD / T_FWD, (T_BWD - T_REPLAY) / T_FWD,
                              T_BWD / nm_, T_FWD / nm_))
end

# --------------------------------------------------------------------------- #
# 9b. PROBE -- reproduce the FIRST non-finite VJP in isolation. A non-finite
#     lambda poisons everything after it, so the only informative object is the
#     single step where it started. Four questions, in order of what they would
#     mean:
#       * is the PRIMAL of that step finite?  (if not, the trajectory is bad,
#         not the adjoint)
#       * does it depend on lambda?           (a lambda-independent blow-up is a
#         Jacobian with an Inf in it, not a seeding problem)
#       * do the exact ZEROS in u cause it?   (`clamp_nonneg` writes exact 0s,
#         and a rate law with a negative power or a reciprocal in it has an
#         infinite derivative there -- plan section 4's clamp caveat, in its
#         sharpest form)
# --------------------------------------------------------------------------- #
if want("adj") && BADREC[] !== nothing
    b = BADREC[]
    ischem = startswith(b.label, "chem")
    cvjp = ischem ? CROSV : CSSPV
    cstep = ischem ? CROS : CSSP
    TH = ischem ? THC : THT
    say("\n---- PROBE: the first non-finite VJP ($(b.label), step $(b.j) of $(b.n)) ----")
    UB = RX.ConcreteRArray(b.u); TB = RX.ConcreteRNumber(b.t); DB = RX.ConcreteRNumber(b.dt)
    let rp = cstep(UB, TH, TB, DB)
        pr = Array(rp[1])
        say(@sprintf("  primal of that step: %d/%d non-finite, EEst=%.6g, max|unew|=%.3e",
                     count(!isfinite, pr), N, Float64(rp[2]), maximum(abs.(pr))))
    end
    scu = max.(abs.(b.u), 1e-12)
    for (nm, lv) in (("the objective lambda", b.lam),
                     ("dense random",         randn(Random.MersenneTwister(4242), N) ./ scu),
                     ("all ones",             ones(N)),
                     ("all zeros",            zeros(N)))
        local rr = cvjp(UB, TH, RX.ConcreteRArray(lv), TB, DB)
        local nb = count(!isfinite, Array(rr[1]))
        local gb = count(k -> !isfinite(Float64(getfield(rr[2].p, k))), PNAMES)
        say(@sprintf("  VJP seeded with %-20s -> lambda_in non-finite %5d/%d, grad_theta non-finite %2d/%d",
                     nm, nb, N, gb, length(PNAMES)))
    end
    nz = count(==(0.0), b.u)
    if nz > 0
        u2 = copy(b.u); u2[u2 .== 0.0] .= 1e-30
        local rr = cvjp(RX.ConcreteRArray(u2), TH, RX.ConcreteRArray(b.lam), TB, DB)
        say(@sprintf("  VJP with the %d exact zeros in u replaced by 1e-30 -> lambda_in non-finite %d/%d, grad_theta non-finite %d/%d",
                     nz, count(!isfinite, Array(rr[1])), N,
                     count(k -> !isfinite(Float64(getfield(rr[2].p, k))), PNAMES), length(PNAMES)))
    else
        say("  (u on that step has no exact zeros, so the clamp is not the trigger)")
    end
    # is it this step, or this STATE? re-run the same u at a much smaller dt.
    for f in (0.1, 0.01)
        local rr = cvjp(UB, TH, RX.ConcreteRArray(b.lam), TB, RX.ConcreteRNumber(b.dt * f))
        say(@sprintf("  VJP at the same state with dt*%.2g -> lambda_in non-finite %d/%d",
                     f, count(!isfinite, Array(rr[1])), N))
    end
end

# --------------------------------------------------------------------------- #
# 9c. If the clamped sweep went non-finite, redo the WHOLE thing with the clamp
#     off. That is the plan's "offer a gradient mode without it" (section 4) and
#     it is also the decisive experiment: `clamp_nonneg` is the only thing in
#     this pipeline that writes EXACT zeros into the state, and a rate law with
#     a reciprocal or a fractional power has an infinite derivative there. The
#     retry replaces the trajectory, so everything downstream (the identity, the
#     forward-mode reference) is then self-consistently the unclamped one.
# --------------------------------------------------------------------------- #
RETRIED_NOCLAMP = false
if want("adj") && want("fwd") && BADREC[] !== nothing && CLAMP[]
    say("\n---- RETRY: the same window and sweep with clamp_nonneg = false ----")
    CLAMP[] = false; BADREC[] = nothing; NCLAMPED = 0
    for k in PNAMES; gacc[k] = 0.0; end
    global RETRIED_NOCLAMP = true
    UEND, CKPTS, TAPES, COUNTS, T_FWD = forward_pass(; record = KEEPTAPE)
    say(@sprintf("  forward pass (no clamp) %.2f s, accept/reject: %s, J = %.15g",
                 T_FWD, steps_sig(COUNTS), dot(WOBJ, UEND)))
    all(isfinite, UEND) || error("non-finite state with the clamp off -- the " *
                                 "unclamped trajectory is not usable")
    tstart = time()
    _, T_REPLAY, NVJP = backward_sweep(WOBJ)
    T_BWD = time() - tstart
    nm_ = length(CKPTS)
    say(@sprintf("  backward sweep (no clamp) %.2f s over %d macro steps, %d VJP calls, %.4f s/VJP",
                 T_BWD, nm_, NVJP, (T_BWD - T_REPLAY) / max(NVJP, 1)))
    say(@sprintf("  COST RATIO backward/forward = %.2f  (VJP-only %.2f)",
                 T_BWD / T_FWD, (T_BWD - T_REPLAY) / T_FWD))
    say(BADREC[] === nothing ?
        "  the unclamped sweep is FINITE throughout -- the clamp's exact zeros were the trigger" :
        "  the unclamped sweep ALSO goes non-finite -- the clamp is not the trigger")
end

if want("adj")

    ord = sort(collect(PNAMES); by = k -> -abs(gacc[k]))
    say("\n  dJ/dtheta, full parameter vector (sorted by magnitude):")
    for k in ord
        th = Float64(getfield(p, k))
        @printf("    %-30s theta=%14.7g  dJ/dtheta=% .12e  elasticity=% .6e\n",
                String(k), th, gacc[k], th * gacc[k] / dot(WOBJ, UEND))
    end
    nz = count(k -> gacc[k] != 0.0, PNAMES)
    say(@sprintf("  %d of %d components nonzero", nz, length(PNAMES)))

    # ---- criterion 2: the structural identity, on the ADJOINT gradient alone -
    let ks = :var"NEIRegrid.scale", kg = :var"NEIRegrid.g0"
        if ks in PNAMES && kg in PNAMES
            lhs = Float64(getfield(p, ks)) * gacc[ks]
            rhs = Float64(getfield(p, kg)) * gacc[kg]
            rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            say(@sprintf("\n  STRUCTURAL IDENTITY  scale*dJ/dscale = g0*dJ/dg0 :  %.12e vs %.12e   rel=%.3e  %s",
                         lhs, rhs, rel, rel <= 1e-12 ? "PASS" : "FAIL"))
        end
    end

    if !isempty(ADJCSV)
        open(ADJCSV, "w") do io
            println(io, "param,theta0,dJ_dtheta,objective,nmacro,window_s,grid,nstates,J,ujitter")
            for k in ord
                println(io, join((String(k), Float64(getfield(p, k)), gacc[k], OBJSPEC,
                                  NMACRO, T_END - T0,
                                  "$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF",
                                  N, dot(WOBJ, UEND), UJIT), ","))
            end
        end
        say("  wrote $ADJCSV")
    end
end

# --------------------------------------------------------------------------- #
# 7. Stage `ctl` -- the host loop vs the traced adaptive_solve, on one macro
#    step. The entire design rests on the host loop reproducing the device loop
#    EXACTLY; the traced pipeline is bit-reproducible run to run, so a nonzero
#    difference here is signal, not noise.
# --------------------------------------------------------------------------- #
if want("ctl")
    say("\n---- CTL: host adaptive loop vs traced adaptive_solve (one macro step) ----")
    adapT(u, th, t0_, t1_, d0) = RTI.adaptive_solve(
        (uu, tt, dd, ax) -> RTI.ssprk43_step((x, s) -> gT(x, ax, s), uu, tt, dd, ATOL_T, RTOL),
        u, t0_, t1_, d0, RTI.pictrl_ssprk43(), th; clamp_nonneg = true)
    adapC(u, th, t0_, t1_, d0) = RTI.adaptive_solve(
        (uu, tt, dd, ax) -> RTI.ros23_step((x, s) -> gC(x, ax, s), uu, tt, dd,
                                           NS, NC, MASKS, ATOL_C, RTOL; unrolled = true,
                                           jac = JACMODE,
                                           symjac = SYMJAC ? ((x, s) -> gJ(x, ax, s)) : nothing),
        u, t0_, t1_, d0, RTI.pictrl_ros23(), th; clamp_nonneg = true)
    T1R = RX.ConcreteRNumber(T0 + MACRO_DT)
    refresh_forcing(T0)
    cT = timed_compile("adaptiveT", () -> RX.@compile compile_options=COPTS adapT(U_R, THT, T_R, T1R, DTT_R))
    cC = timed_compile("adaptiveC", () -> RX.@compile compile_options=COPTS adapC(U_R, THC, T_R, T1R, DTC_R))
    for (nm, c, TH, dt0, ctrl, cstep) in ((:transport, cT, THT, DT0T, RTI.pictrl_ssprk43(), CSSP),
                                          (:chemistry, cC, THC, DT0C, RTI.pictrl_ros23(), CROS))
        r = c(U_R, TH, T_R, T1R, RX.ConcreteRNumber(dt0))
        ud = Array(r[1]); ndev = (Float64(r[4]), Float64(r[5]))
        uh, th_, dth, na, nr = host_adaptive!(cstep, copy(UBASE), T0, T0 + MACRO_DT, dt0, ctrl, TH)
        d = maximum(abs.(ud .- uh))
        sc = max.(abs.(ud), abs.(uh), 1e-30)
        @printf("  %-10s device naccept/nreject %d/%d   host %d/%d   max|du| %.3e  max rel %.3e  dt_end dev %.17g host %.17g  %s\n",
                String(nm), Int(ndev[1]), Int(ndev[2]), na, nr, d,
                maximum(abs.(ud .- uh) ./ sc), Float64(r[3]), dth,
                (d == 0.0 && Int(ndev[1]) == na && Int(ndev[2]) == nr) ?
                    "BIT-IDENTICAL" : "DIFFERS")
    end
end

# --------------------------------------------------------------------------- #
# 10. Stage `ref` -- FORWARD mode over the SAME frozen step sequence.
#     This is the apples-to-apples reference for criterion 1: it differentiates
#     the identical fixed-dt composed map, with the identical clamp masks, by a
#     completely separate Enzyme program (forward JVP vs reverse VJP). One pass
#     over the window per parameter -- which is exactly the cost model the
#     adjoint exists to escape, and is why the list is short.
# --------------------------------------------------------------------------- #
# NOTE: no `AbstractVector` method. `bufs` is a NamedTuple of dense arrays, and
# an `AbstractVector` method would be ambiguous with the `AbstractArray{<:Real}`
# one for a 1-D buffer -- Julia silently picked the vector method, which mapped
# elementwise and produced a `Vector{ConcretePJRTNumber}` where the compiled
# thunk wanted a `ConcretePJRTArray`. Same shape of container in, same out.
_zero_like(::Real) = 0.0
_zero_like(x::AbstractArray{<:Real}) = zeros(size(x))
_zero_like(x::NamedTuple) = NamedTuple{keys(x)}(map(_zero_like, values(x)))
_zero_like(x::Tuple) = map(_zero_like, x)
_todev(x::Real) = RX.ConcreteRNumber(Float64(x))
_todev(x::AbstractArray{<:Real}) = RX.ConcreteRArray(Array{Float64}(x))
_todev(x::NamedTuple) = NamedTuple{keys(x)}(map(_todev, values(x)))
_todev(x::Tuple) = map(_todev, x)

if want("ref") && want("adj")
    say("\n---- REF: forward-mode JVP chained over the SAME frozen tape ----")
    CSSPJ = timed_compile("ssp_jvp", () -> RX.@compile compile_options=COPTS ssp_jvp(U_R, U_R, THT, THT, T_R, DTT_R))
    CROSJ = timed_compile("ros_jvp", () -> RX.@compile compile_options=COPTS ros_jvp(U_R, U_R, THC, THC, T_R, DTC_R))

    zh_T = _zero_like((p = p, bufs = host_bufs[1]))
    zh_C = _zero_like(thC(p, host_bufs[2], host_bufsJ))
    # `merge`, not a literal (p=..., bufs=...): the chemistry theta has a third
    # field under jac=:sym and a literal would silently drop it.
    onehot_dth(zh, k::Symbol) =
        _todev(merge(zh, (p = NamedTuple{keys(zh.p)}(map(kk -> kk === k ? 1.0 : 0.0,
                                                         keys(zh.p))),)))

    # WHICH SLOT IS THE TANGENT. Settled empirically, not assumed: with a zero
    # state seed AND a zero theta seed the tangent must be EXACTLY zero, and no
    # other slot of the returned tuple is (the primal is the state, which is not).
    ZTHT = _todev(zh_T)
    ZTHC = _todev(zh_C)
    ZU = RX.ConcreteRArray(zeros(Float64, N))
    JSLOT = let e = (refresh_forcing(CKPTS[1].epoch); tapes_for(1)[1][1])
        r = CSSPJ(RX.ConcreteRArray(e.u), ZU, THT, ZTHT,
                  RX.ConcreteRNumber(e.t), RX.ConcreteRNumber(e.dt))
        zs = [i for i in eachindex(r) if all(iszero, Array(r[i]))]
        say("  zero-seed JVP: " * join([@sprintf("r[%d] norm=%.6e", i, norm(Array(r[i])))
                                        for i in eachindex(r)], ", "))
        if length(zs) == 1
            say("  tangent is slot r[$(zs[1])] of $(length(r))")
            zs[1]
        else
            # MEASURED, and it is why this stage is skippable rather than fatal:
            # on the real ReSEACT RHS `Enzyme.autodiff(Forward, ..., Duplicated,
            # ...)` comes back with ONE slot, and under an exactly zero seed that
            # slot has norm ~= ||u|| -- it is the PRIMAL, not the tangent. The
            # same call shape on a toy carrying the same constructs returns the
            # correct tangent in the same slot. So there is no forward-mode
            # reference to be had here today; `fdtape` is the reference instead.
            say("  NO SLOT IS ZERO UNDER A ZERO SEED -- Enzyme's forward mode is " *
                "returning the primal, not the tangent, on this RHS. The forward-mode " *
                "reference is NOT AVAILABLE; use the fdtape stage. Skipping ref.")
            0
        end
    end
    JSLOT == 0 && @goto skip_ref

    function forward_stage(cjvp, tape::Vector{StepRec}, du::Vector{Float64}, TH, DTH)
        for e in tape
            r = cjvp(RX.ConcreteRArray(e.u), RX.ConcreteRArray(du), TH, DTH,
                     RX.ConcreteRNumber(e.t), RX.ConcreteRNumber(e.dt))
            du = Array(r[JSLOT]) .* e.mask
        end
        return du
    end

    # Is the blow-up specific to REVERSE, or is the step map's derivative
    # genuinely infinite at that state? Forward mode over the SAME step at the
    # SAME state answers that, and it is one call.
    if BADREC[] !== nothing
        b = BADREC[]; ischem = startswith(b.label, "chem")
        cj = ischem ? CROSJ : CSSPJ; TH = ischem ? THC : THT
        zh = ischem ? zh_C : zh_T
        dth0 = _todev(zh)                                 # zero theta tangent
        duv = randn(Random.MersenneTwister(4242), N) .* max.(abs.(b.u), 1e-12)
        r = cj(RX.ConcreteRArray(b.u), RX.ConcreteRArray(duv), TH, dth0,
               RX.ConcreteRNumber(b.t), RX.ConcreteRNumber(b.dt))
        jv = Array(r[JSLOT])
        say(@sprintf("  PROBE (forward): JVP of the SAME step at the SAME state -> %d/%d non-finite tangent entries",
                     count(!isfinite, jv), N))
    end

    # ---- the per-step dot-product identity, in THIS driver, on a tape step ---
    # <lam, J v> == <J^T lam, v> + <grad_theta, dtheta>. Exact arithmetic, no
    # step size. If it holds, the per-step VJP and JVP are exact transposes at
    # the states this sweep actually visits, and any disagreement further down
    # is in the CHAINING, not in the step derivative.
    let (tpT1, tpC1, _) = (refresh_forcing(CKPTS[1].epoch); tapes_for(1))
        for (nm, cj, cv, tp, TH, zh) in (("SSPRK43", CSSPJ, CSSPV, tpT1, THT, zh_T),
                                         ("ROS23",   CROSJ, CROSV, tpC1, THC, zh_C))
            e = tp[end]
            rng = Random.MersenneTwister(90210)
            scu = max.(abs.(e.u), 1e-12)
            v = randn(rng, N) .* scu
            lam = randn(rng, N) ./ scu
            dthh = NamedTuple{keys(p)}(map(kk -> randn(rng) * max(abs(Float64(getfield(p, kk))), 1e-12),
                                           keys(p)))
            DTHd = _todev(merge(zh, (p = dthh,)))
            local jv = Array(cj(RX.ConcreteRArray(e.u), RX.ConcreteRArray(v), TH, DTHd,
                                RX.ConcreteRNumber(e.t), RX.ConcreteRNumber(e.dt))[JSLOT])
            local rv = cv(RX.ConcreteRArray(e.u), TH, RX.ConcreteRArray(lam),
                          RX.ConcreteRNumber(e.t), RX.ConcreteRNumber(e.dt))
            local lhs = dot(lam, jv)
            local rhs = dot(Array(rv[1]), v) +
                  sum(Float64(getfield(rv[2].p, kk)) * getfield(dthh, kk) for kk in keys(p))
            rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            @printf("  dot-product identity %-8s <lam,Jv>=% .12e  <J'lam,v>+<g,dth>=% .12e  rel=%.3e  %s\n",
                    nm, lhs, rhs, rel, rel < 1e-6 ? "PASS" : "FAIL")
        end
    end

    say(@sprintf("  %-30s %-22s %-22s %s", "parameter", "adjoint dJ/dtheta",
                 "forward JVP (frozen dt)", "rel"))
    for nm in REFPARAM
        k = Symbol(nm)
        k in PNAMES || (say("  $nm is not a runtime scalar parameter -- skipped"); continue)
        dTHT = onehot_dth(zh_T, k); dTHC = onehot_dth(zh_C, k)
        du = zeros(Float64, N)
        ta = time()
        for kk in 1:length(CKPTS)
            refresh_forcing(CKPTS[kk].epoch)
            tpT, tpC, _ = tapes_for(kk)
            du = forward_stage(CSSPJ, tpT, du, THT, dTHT)   # transport first, forward order
            kk == 1 && @printf("    [%s] ||du|| after macro 1 transport = %.6e\n", nm, norm(du))
            du = forward_stage(CROSJ, tpC, du, THC, dTHC)
            @printf("    [%s] ||du|| after macro %d = %.6e   dot(w,du) = %.12e\n",
                    nm, kk, norm(du), dot(WOBJ, du))
        end
        fw = dot(WOBJ, du); ad = gacc[k]
        rel = abs(fw - ad) / max(abs(fw), abs(ad), 1e-300)
        @printf("  %-30s % .14e  % .14e  %.3e  %s   [%.1f s]\n",
                nm, ad, fw, rel, rel <= 1e-8 ? "PASS" : "CHECK", time() - ta)
    end
    @label skip_ref
end

# --------------------------------------------------------------------------- #
# 11. Stage `fdtape` -- central finite differences of the FROZEN-dt composed
#     map. This is the confounder-free acceptance test for criterion 1, and it
#     needs no extra compile: it replays the SAME (t, dt) sequence the forward
#     pass accepted, through the SAME compiled single-step programs, at theta +/- h.
#
#     It is the right comparison precisely because the discrete adjoint is the
#     derivative of THAT map. Comparing against `sensitivity_forward.jl` instead
#     carries three confounders that are each larger than the 1e-8 target:
#       * it differentiates the adaptive controller's dt as well (the standard
#         discrete adjoint holds dt fixed -- see the header);
#       * it runs the clamp, and the clamped adjoint is not finite here;
#       * its trajectory is the DEVICE while-loop's, which the CTL stage
#         measures as ~3e-10 per macro step away from the host loop's.
#     Fixing the dt sequence and the compiled programs removes all three.
# --------------------------------------------------------------------------- #
if want("fdtape") && want("adj")
    say("\n---- FDTAPE: central fd of the FROZEN-dt map (no new compile) ----")
    say(@sprintf("  frozen sequence: %d transport + %d chemistry steps over %d macro steps",
                 sum(length(c.seqT) for c in CKPTS), sum(length(c.seqC) for c in CKPTS),
                 length(CKPTS)))

    function replay_frozen(THT_, THC_)
        u = copy(UBASE)
        for ck in CKPTS
            refresh_forcing(ck.epoch)
            u, _, _ = replay_fixed(u, ck.seqT, ck.seqC; TH_T = THT_, TH_C = THC_, record = false)
        end
        return dot(WOBJ, u)
    end
    # The frozen replay at the BASE theta must reproduce the adaptive forward
    # pass exactly -- same states, same order, same programs. If it does not,
    # the frozen map is not the map the adjoint was taken along.
    let J0 = replay_frozen(THT, THC), Jf = dot(WOBJ, UEND)
        @printf("  frozen replay at theta0: J=%.17g vs forward pass J=%.17g  (diff %.3e)  %s\n",
                J0, Jf, abs(J0 - Jf), J0 == Jf ? "BIT-IDENTICAL" : "DIFFERS")
    end
    theta_dev(k, h) = let vals = map(kk -> RX.ConcreteRNumber(Float64(getfield(p, kk)) +
                                                              (kk === k ? h : 0.0)), keys(p))
        pr = NamedTuple{keys(p)}(vals)
        ((p = pr, bufs = dev_bufs[1]), thC(pr, dev_bufs[2], dev_bufsJ))
    end
    say("  parameter                        h            fd derivative        adjoint dJ/dtheta     rel")
    for nm in REFPARAM
        k = Symbol(nm)
        k in PNAMES || continue
        sc = max(abs(Float64(getfield(p, k))), 1.0)
        best = (Inf, 0.0, 0.0)
        for rel_h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
            h = sc * rel_h
            tp, tc = theta_dev(k, h);  Jp = replay_frozen(tp, tc)
            tm, tmc = theta_dev(k, -h); Jm = replay_frozen(tm, tmc)
            fd = (Jp - Jm) / (2h)
            r = abs(fd - gacc[k]) / max(abs(gacc[k]), 1e-300)
            @printf("  %-30s %10.3e   % .12e   % .12e   %.3e\n", nm, h, fd, gacc[k], r)
            r < best[1] && (best = (r, h, fd))
        end
        # If EVERY step gave a non-finite fd -- which HAPPENS, because the frozen
        # replay intermittently returns NaN (the non-finite issue in
        # DIFFERENTIABILITY_PLAN.md) -- `best` is still its `(Inf, 0.0, 0.0)`
        # initializer, and printing it renders as `fd=0.000000000000e+00
        # rel=Inf`. That reads as "finite differences measured a ZERO
        # derivative", which is a different and much more alarming claim than
        # "no finite difference was obtained" -- and it is the same shape as the
        # const-folded-parameter trap, where a wrong zero and a wrong check agree
        # silently. Report what actually happened instead of printing sentinels.
        if !isfinite(best[1])
            # NB: ONE string literal. `@printf` validates its format at MACRO EXPANSION
            # time, so a concatenated `"..." * "..."` format is a LoadError that fires
            # when this file is included -- not when this branch runs. It made the whole
            # driver unloadable on Julia 1.12 regardless of which stages were selected.
            @printf("  %-30s NO FINITE FD at any step -- every central difference was non-finite. adjoint=%.12e is UNCHECKED here, not refuted.\n", nm, gacc[k])
        else
            @printf("  %-30s BEST h=%.3e  fd=%.12e  adjoint=%.12e  rel=%.3e  %s\n",
                    nm, best[2], best[3], gacc[k], best[1], best[1] <= 1e-6 ? "PASS" : "FAIL")
        end
    end
end

for k in PNAMES
    want("adj") || break
    gacc[k] == 0.0 && continue
    @printf("RESULT label=%s param=%s theta0=%.10g obj=%s nmacro=%d window_s=%.0f grid=%dx%dx%d nstates=%d ujitter=%g J=%.10g dJ_dtheta=%.10e\n",
            LABEL, String(k), Float64(getfield(p, k)), OBJSPEC, NMACRO, T_END - T0,
            GRID_MP["NLON"], GRID_MP["NLAT"], NLEV_EFF, N, UJIT, dot(WOBJ, UEND), gacc[k])
end
say("DONE $LABEL")
