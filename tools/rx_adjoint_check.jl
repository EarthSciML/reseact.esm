#!/usr/bin/env julia
# ===========================================================================
# rx_adjoint_check.jl -- validation harness for the Phase 3 DISCRETE ADJOINT
# (DIFFERENTIABILITY_PLAN.md section 3, Phase 3).
# ===========================================================================
# Builds ReSEACT through exactly the path run_reseact_reactant.jl uses (same
# split, same :oop emitter, same forcing), then checks the ONE-STEP VJPs added
# to rx_traced_integrator.jl against everything cheap that can disagree with
# them:
#
#   1. AD block Jacobian vs the existing FD one            -- to FD accuracy
#   2. dot-product identity  <lam, J v> == <J^T lam, v>    -- the strongest
#      check: exact arithmetic, no step size, fails loudly on any transpose or
#      index error
#   3. central finite differences of the SAME compiled step, swept over eps
#   4. a `stablehlo.while` census of the compiled VJP modules -- the whole
#      design rests on there being none (reverse mode cannot cross one)
#   5. a short adaptive solve, jac=:fd vs jac=:ad: compile time and the
#      accepted/rejected step counts the exact Jacobian buys
#
# Run it small -- the stage algebra is model-independent, and a CONUS build is
# ~718 s:
#   RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8 julia --project=<rx env> \
#       tools/rx_adjoint_check.jl
#
# Env: RESEACT_MODEL / LON0 / LAT0 / NLON / NLAT / NLEV / T0 as in the runners,
# plus RESEACT_ADJ_STAGES (comma-separated subset of
#   census,jac,ssprk,ros_fd,ros_ad,solve -- see the stage banner below).
# ===========================================================================
import Pkg
const REPO = dirname(@__DIR__)
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Printf, Random, Statistics, Logging
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

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
_env(k, d)  = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 6), nlat = _env("NLAT", 6),
                           nlev = _env("NLEV", 8))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const RTOL, ATOL_T, ATOL_C = 1e-4, 1e-6, 1e-9
const MACRO_DT = parse(Float64, get(ENV, "RESEACT_MACRO_DT", "300"))

say("=== rx_adjoint_check: grid $(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF T0=$(round(Int,T0)) ===")

# --------------------------------------------------------------------------- #
# 1. Build, exactly as run_reseact_reactant.jl does.
# --------------------------------------------------------------------------- #
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff
    file = EA.load(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
    f0 = reseact_forcing(CHEMDIR; ndays = 1)
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
        i == 1 ? (global u0, p, var_map = u0i, pi, vmi) :
                 (vmi == var_map || error("split part 2 var_map != part 1"))
    end
end
foreach(d -> d.materialize!(), dms)
say(@sprintf("BUILD %.2f s   nstates=%d  nparams=%d", time() - tb, length(u0), length(p)))

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

host_bufs = [EA.forcing_buffers(fo[i]) for i in 1:2]
g4 = [EA.rhs_with_buffers(fo[i]) for i in 1:2]
PERM = cellmajor_perm(var_map)
const NS = PERM.NS; const NC = PERM.NC; const N = PERM.N
const MASKS = RTI.species_masks(var_map, NS, NC)
say("  NS=$NS species, NC=$NC cells, N=$N states")

# The RHS with its differentiable payload made an EXPLICIT argument, which is
# what the adjoint entry points need: the production call site hides p and the
# forcing buffers in a closure, and a closure is opaque to Enzyme.
gT(u, th, t) = g4[1](u, th.p, t, th.bufs)
gC(u, th, t) = g4[2](u, th.p, t, th.bufs)

dev_bufs = [map(RX.ConcreteRArray, host_bufs[i]) for i in 1:2]
for i in 1:2; EA.sync_forcing!(dev_bufs[i], EA.forcing_buffers(fo[i])); end
_devp(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
PRd = _devp(p)
THT = (p = PRd, bufs = dev_bufs[1])
THC = (p = PRd, bufs = dev_bufs[2])

# --------------------------------------------------------------------------- #
# 2. Test vectors. State magnitudes here span ~1e-9 (OH) to ~1e3 (m), so every
#    perturbation direction is RELATIVE to the state it perturbs -- an absolute
#    random direction would be pure noise on the small species and swamp the FD
#    difference on the large ones.
# --------------------------------------------------------------------------- #
Random.seed!(20260811)
uh = copy(u0)
scale_u = max.(abs.(uh), 1e-12)
vuh = randn(N) .* scale_u                     # relative direction in state space
lamh = randn(N) ./ scale_u                    # dual to it, so <lam,u> is O(1)

# THE DEFAULT BASE POINT IS DEGENERATE, AND CHECK 3 CANNOT SAY SO.
# `u0` is the model's default IC: every SuperFast field is EXACTLY spatially
# uniform (only Transport3D.m varies -- it is overwritten with hydrostatic dp
# above). The PPM advection in EarthSciDiscretizations is monotonicity-limited,
# and every one of its limiter switches is a product of NEIGHBOUR DIFFERENCES
# compared against zero -- ppm_slope_mono / ppm_lev_slope_mono (CW84 eq. 1.8,
# `(ap-a0)*(a0-am) > 0`) and ppmflux_limit_left / ppm_limit_right (eq. 1.10,
# `(qr-qi)*(qi-ql) <= 0`). On a uniform field every one of those products is
# exactly 0, so u0 sits ON the switching surface of essentially every interior
# cell, and the step map is not differentiable there.
#
# It is non-differentiable IN A WAY THE fwd/bwd TEST BELOW IS BLIND TO. The
# guard is QUADRATIC in the perturbation: at u0 +/- e*v it equals
# e^2*(dv_{i+1}-dv_i)*(dv_i-dv_{i-1}), which has the SAME SIGN for both signs of
# e. Both one-sided quotients therefore land on the same (other) branch, they
# agree with each other, and the residual is flat in eps -- the exact signature
# CHECK 3's banner attributes to "a wrong adjoint". Measured on the transport
# RHS with ForwardDiff on the HOST (no Reactant, no Enzyme, no integrator): AD
# at u0 differs from every FD quotient by 9.8e-3 relative and eps-independently,
# while AD at u0 +/- 1e-14*v -- EITHER SIGN -- agrees with the same FD quotient
# to 5.6e-10. The 9.15e-6 the step-level check reports is that same gap diluted
# by dt/||u||, and it carries the same CO 90.7% / O3 9.3% attribution.
#
# Set RESEACT_ADJ_UJITTER (e.g. 1e-3) to move the base point off the uniform
# field. The residual then behaves like an ordinary FD error and shrinks with
# eps -- provided eps is small enough that the FD stencil does not itself step
# across a switch, which is why the sweep has to reach 1e-8 to see it.
const UJIT = parse(Float64, get(ENV, "RESEACT_ADJ_UJITTER", "0"))
if UJIT > 0
    # a PRIVATE stream, so switching the knob on does not also move `dtheta_host`
    uh .*= (1 .+ UJIT .* randn(Random.MersenneTwister(31337), N))
    say(@sprintf("  base point jittered by %.0e relative (RESEACT_ADJ_UJITTER)", UJIT))
end
UR = RX.ConcreteRArray(uh); VU = RX.ConcreteRArray(vuh); LAM = RX.ConcreteRArray(lamh)
TR = RX.ConcreteRNumber(T0)

# A tangent for theta with the same nesting: relative on the scalars, relative
# on the buffers.
_rnd_like(x::Real) = randn() * max(abs(Float64(x)), 1e-12)
_rnd_like(x::AbstractArray) = randn(size(x)) .* max.(abs.(Float64.(x)), 1e-12)
_rnd_like(nt::NamedTuple) = NamedTuple{keys(nt)}(map(_rnd_like, values(nt)))
_rnd_like(tp::Tuple) = map(_rnd_like, tp)
dtheta_host(th_host) = _rnd_like(th_host)
_todev(x::Real) = RX.ConcreteRNumber(Float64(x))
_todev(x::AbstractArray) = RX.ConcreteRArray(Array{Float64}(x))
_todev(nt::NamedTuple) = NamedTuple{keys(nt)}(map(_todev, values(nt)))
_todev(tp::Tuple) = map(_todev, tp)
th_host_T = (p = p, bufs = host_bufs[1])
th_host_C = (p = p, bufs = host_bufs[2])
dTHT_h = dtheta_host(th_host_T); dTHC_h = dtheta_host(th_host_C)
dTHT = _todev(dTHT_h); dTHC = _todev(dTHC_h)

# host-side <grad_theta, dtheta>, over the same nesting
_ip(a::RX.ConcretePJRTNumber, b) = Float64(a) * Float64(b)
_ip(a::Real, b) = Float64(a) * Float64(b)
_ip(a::AbstractArray, b::AbstractArray) = sum(Float64.(Array(a)) .* Float64.(Array(b)))
_ip(a::NamedTuple, b::NamedTuple) = sum(_ip(getfield(a, k), getfield(b, k)) for k in keys(a); init = 0.0)
_ip(a::Tuple, b::Tuple) = sum(_ip(a[i], b[i]) for i in eachindex(a); init = 0.0)
_ip(::Nothing, _) = 0.0

# theta perturbed by +/- eps along dtheta, on device (structure-preserving)
_axpy(a::Real, b::Real, e) = RX.ConcreteRNumber(Float64(a) + e * Float64(b))
_axpy(a::AbstractArray, b::AbstractArray, e) = RX.ConcreteRArray(Float64.(Array(a)) .+ e .* Float64.(Array(b)))
_axpy(a::NamedTuple, b::NamedTuple, e) = NamedTuple{keys(a)}(map(k -> _axpy(getfield(a, k), getfield(b, k), e), keys(a)))
_axpy(a::Tuple, b::Tuple, e) = map((x, y) -> _axpy(x, y, e), a, b)
_pert(th_host, dth_host, e) = _axpy(th_host, dth_host, e)

# zeroed tangents, so a direction can be switched on one piece at a time. The
# FD check has to be able to say WHERE a disagreement lives: perturbing the
# state, the 49 scalar parameters, and the forcing buffers all at once gives one
# number that cannot be attributed, and the three have very different expected
# behaviour (the buffers reach `interp.searchsorted` and the calendar, which
# contribute NO derivative by contract -- plan section 4).
_zero_like(x::Real) = 0.0
_zero_like(x::AbstractArray) = zeros(size(x))
_zero_like(nt::NamedTuple) = NamedTuple{keys(nt)}(map(_zero_like, values(nt)))
_zero_like(tp::Tuple) = map(_zero_like, tp)

# state index -> species/group name, for attributing a residual
const GROUP_OF = let v = fill("?", N)
    for (nm, idx) in var_map
        v[idx] = replace(nm, r"\[.*$" => "")
    end
    v
end

# --------------------------------------------------------------------------- #
# 3. Stage selection. Enzyme-MLIR's reverse pass can SEGFAULT on some modules
#    (see the ROS23/chemistry note in the report), and a segfault takes the
#    process with it -- a try/catch cannot save the ~560 s build that preceded
#    it. So every check is an independently selectable stage and the harness is
#    meant to be run several times, in parallel, one stage each:
#      RESEACT_ADJ_STAGES=census,jac,solve   (no reverse mode; always survives)
#      RESEACT_ADJ_STAGES=ssprk              (transport VJP)
#      RESEACT_ADJ_STAGES=ros_fd             (chemistry VJP, FD Jacobian)
#      RESEACT_ADJ_STAGES=ros_ad             (chemistry VJP, exact Jacobian)
#      RESEACT_ADJ_STAGES=ros_adfwd          (exact-Jacobian step, FORWARD only)
#      RESEACT_ADJ_STAGES=addump             (write the pre-pipeline modules to
#                                             $RESEACT_ADJ_HLO; no reverse pass,
#                                             so it survives the segfault)
# --------------------------------------------------------------------------- #
const STAGES = Set(split(get(ENV, "RESEACT_ADJ_STAGES",
                             "census,jac,ssprk,ros_fd,ros_ad,solve"), ","))
want(s) = s in STAGES
say("  stages: " * join(sort(collect(STAGES)), ", "))

_flatJ(Jb) = vcat([Jb[r, s] for r in 1:NS for s in 1:NS]...)
jac_ad_f(u, th, t) = _flatJ(RTI.ad_block_jac(uu -> gC(uu, th, t), u, Val(NS), NC))
function jac_fd_f(u, th, t)
    f0 = gC(u, th, t)
    f0b = [RTI._blk(f0, r, NC) for r in 1:NS]
    _flatJ(RTI.fd_block_jac_unrolled(uu -> gC(uu, th, t), u, f0b, NS, NC, MASKS))
end
# Stage `jacrev`: REVERSE-OVER-FORWARD WITH THE STAGE ALGEBRA REMOVED. The
# reverse pass here crosses nothing but the coloured-JVP Jacobian itself -- no
# blocksolve, no ROS23 stages, no `sum(lambda .* unew)`. If `ros_ad` segfaults
# and this does too, the ROS23 step is irrelevant to the crash and the minimal
# statement is "reverse mode over NCOL nested enzyme.fwddiff calls of the
# emitted chemistry RHS". RESEACT_ADJ_NCOL sweeps the colour count (1..NS),
# which is the one axis the model-free bisection in tools/diag/rof_repro.jl
# cannot reach. `jacrev_fd` is the control: same objective, FD Jacobian, so
# reverse-over-nothing.
const NCOL = parse(Int, get(ENV, "RESEACT_ADJ_NCOL", string(NS)))
function _jdot_ad(g, u, th, t, ::Val{K}) where {K}
    Jb = RTI.ad_block_jac(uu -> g(uu, th, t), u, Val(K), NC)
    acc = nothing
    for r in 1:K, s in 1:K
        p = sum(Jb[r, s])
        acc = acc === nothing ? p : acc + p
    end
    return acc
end
function _jdot_fd(g, u, th, t, ::Val{K}) where {K}
    f0 = g(u, th, t)
    f0b = [RTI._blk(f0, r, NC) for r in 1:NS]
    Jb = RTI.fd_block_jac_unrolled(uu -> g(uu, th, t), u, f0b, NS, NC, MASKS)
    acc = nothing
    for r in 1:K, s in 1:K
        p = sum(Jb[r, s])
        acc = acc === nothing ? p : acc + p
    end
    return acc
end
for (stg, fn) in (("jacrev", _jdot_ad), ("jacrev_fd", _jdot_fd))
    want(stg) || continue
    say("\n---- $stg : reverse over the block Jacobian ALONE, NCOL=$NCOL ----")
    # `t` rides as an ARGUMENT, not a capture: a captured ConcreteRNumber inside
    # a traced function is "Cannot trace existing trace type".
    grad(u, th, t) = EZ.gradient(EZ.Reverse, fn, EZ.Const(gC), u, th,
                                 EZ.Const(t), EZ.Const(Val(NCOL)))
    t0 = time(); cg = @compile grad(UR, THC, TR); tc = time() - t0
    r = cg(UR, THC, TR)
    @printf("  compile %.1f s   ||dJ/du||=%.6e\n", tc, norm(Array(r[2])))
end

if want("jac")
    say("\n---- CHECK 1: AD block Jacobian vs FD block Jacobian (chemistry) ----")
    t0 = time(); cja = @compile jac_ad_f(UR, THC, TR); tca = time() - t0
    t0 = time(); cjf = @compile jac_fd_f(UR, THC, TR); tcf = time() - t0
    Ja = Array(cja(UR, THC, TR)); Jf = Array(cjf(UR, THC, TR))
    @printf("  compile: ad %.1f s, fd %.1f s\n", tca, tcf)
    @printf("  ||J_ad||=%.6e  ||J_fd||=%.6e  ||J_ad-J_fd||/||J_ad||=%.3e\n",
            norm(Ja), norm(Jf), norm(Ja .- Jf) / norm(Ja))
    # A per-entry relative error is meaningless on entries whose magnitude is
    # far below the column's: that is exactly where FD cancellation lives, and
    # it is the FD Jacobian that is wrong there, not the AD one. So report the
    # error scaled by the COLUMN norm (the quantity the linear solve actually
    # sees) and, separately, the per-entry relative error restricted to entries
    # that carry weight.
    @printf("  max|dJ| = %.3e   (FD h ~ sqrt(eps) = %.1e)\n",
            maximum(abs.(Ja .- Jf)), sqrt(eps(Float64)))
    for frac in (1e-2, 1e-4, 1e-6)
        m = abs.(Ja) .>= frac * maximum(abs.(Ja))
        @printf("    entries with |J_ad| >= %.0e*max: n=%-7d max rel err = %.3e\n",
                frac, count(m), maximum(abs.(Ja[m] .- Jf[m]) ./ abs.(Ja[m])))
    end
    nz = abs.(Ja) .> 0
    @printf("    ALL nonzero entries:              n=%-7d max rel err = %.3e\n",
            count(nz), maximum(abs.(Ja[nz] .- Jf[nz]) ./ abs.(Ja[nz])))
end

# --------------------------------------------------------------------------- #
# 4. CHECKS 2+3 -- dot-product identity and finite differences, per method.
# --------------------------------------------------------------------------- #
ros_out(u, th, t, dt; jac=:ad) = RTI.ros23_step_out(gC, u, th, t, dt, NS, NC, MASKS, ATOL_C, RTOL; jac=jac)
ros_out_fd(u, th, t, dt) = RTI.ros23_step_out(gC, u, th, t, dt, NS, NC, MASKS, ATOL_C, RTOL; jac=:fd)
# jac=:ad MUST be explicit here. `ros23_step_vjp`'s own default was flipped to
# :fd when the reverse-over-forward segfault was found, and this call site --
# the one the "jac=:ad" stage uses -- silently inherited it, so the AD VJP has
# not actually been exercised since. Measured: with the kwarg omitted the module
# this produces is byte-identical to `ros_vjp_fd`'s.
ros_vjp(u, th, lam, t, dt) = RTI.ros23_step_vjp(gC, u, th, t, dt, lam, NS, NC, MASKS, ATOL_C, RTOL; jac=:ad)
ros_jvp(u, du, th, dth, t, dt) = RTI.ros23_step_jvp(gC, u, du, th, dth, t, dt, NS, NC, MASKS, ATOL_C, RTOL)
ros_vjp_fd(u, th, lam, t, dt) = RTI.ros23_step_vjp(gC, u, th, t, dt, lam, NS, NC, MASKS, ATOL_C, RTOL; jac=:fd)
ros_jvp_fd(u, du, th, dth, t, dt) = RTI.ros23_step_jvp(gC, u, du, th, dth, t, dt, NS, NC, MASKS, ATOL_C, RTOL; jac=:fd)
ssp_out(u, th, t, dt) = RTI.ssprk43_step_out(gT, u, th, t, dt, ATOL_T, RTOL)
ssp_vjp(u, th, lam, t, dt) = RTI.ssprk43_step_vjp(gT, u, th, t, dt, lam, ATOL_T, RTOL)
ssp_jvp(u, du, th, dth, t, dt) = RTI.ssprk43_step_jvp(gT, u, du, th, dth, t, dt, ATOL_T, RTOL)

function run_checks(nm, TH, dTH, th_host, dth_host, dtval, outf, vjpf, jvpf)
    say("\n---- $nm : one step, dt=$dtval ----")
    DT = RX.ConcreteRNumber(dtval)
    t0 = time(); cout = @compile outf(UR, TH, TR, DT); tco = time() - t0
    t0 = time(); cjvp = @compile jvpf(UR, VU, TH, dTH, TR, DT); tcj = time() - t0
    # vjpf === nothing: forward-mode only. That is not a shortcut, it is the
    # only way to exercise `jac=:ad` on the real model at all -- reverse over it
    # segfaults inside Enzyme-MLIR (see ros23_step_vjp's comment). Forward mode
    # still answers the question that matters for the stage algebra: is AD the
    # derivative of the compiled step?
    cvjp = vjpf === nothing ? nothing : (@compile vjpf(UR, TH, LAM, TR, DT))
    tcv = time() - t0 - tcj
    @printf("  compile: primal %.1f s, JVP %.1f s, VJP %s\n", tco, tcj,
            cvjp === nothing ? "(skipped)" : @sprintf("%.1f s", tcv))

    lam_in = zeros(N); gth = nothing
    if cvjp !== nothing
        # CHECK 4 on the module Enzyme is actually HANDED, not just on the
        # primal: optimize=false stops before the pipeline, so what is printed
        # is the `enzyme.autodiff` op wrapping the step body -- exactly the
        # thing that must contain no while region.
        cvjp_hlo = @code_hlo optimize=false vjpf(UR, TH, LAM, TR, DT)
        census("VJP module as handed to Enzyme", sprint(show, cvjp_hlo))
        g = cvjp(UR, TH, LAM, TR, DT)
        lam_in = Array(g[1]); gth = g[2]
    end

    # Direction decomposition. `zth` is a zero theta tangent, so each piece can
    # be exercised alone.
    zu = zeros(N); zth_h = _zero_like(th_host)
    dth_p_h    = (p = dth_host.p,          bufs = zth_h.bufs)
    dth_bufs_h = (p = zth_h.p,             bufs = dth_host.bufs)
    dirs = (("state u only",   vuh, zth_h),
            ("params p only",  zu,  dth_p_h),
            ("forcing bufs",   zu,  dth_bufs_h),
            ("all together",   vuh, dth_host))

    if cvjp !== nothing
        say("  CHECK 2 dot-product identity <lam, J v> == <J' lam, v>, per direction:")
        # scale the "is this direction numerically zero" test against the LARGEST
        # direction, not against the direction itself -- a direction whose two
        # sides are both 1e-17 next to an O(60) sibling carries no information,
        # and scoring its relative difference would be a reporting artifact.
        ref = maximum(abs(dot(lamh, Array(cjvp(UR, RX.ConcreteRArray(d[2]), TH,
                                               _todev(d[3]), TR, DT)))) for d in dirs)
        for (dn, du_h, dth_h) in dirs
            DU = RX.ConcreteRArray(du_h); DTHd = _todev(dth_h)
            jv = Array(cjvp(UR, DU, TH, DTHd, TR, DT))
            lhs = dot(lamh, jv)
            rhs = dot(lam_in, du_h) + _ip(gth, dth_h)
            rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            nul = max(abs(lhs), abs(rhs)) < 1e-12 * ref
            @printf("    %-15s <lam,Jv>=% .12e  <J'lam,v>=% .12e  rel=%.3e %s\n",
                    dn, lhs, rhs, rel, nul ? "(both ~0 vs the largest direction)" :
                    rel < 1e-10 ? "PASS" : "FAIL")
        end
    end

    say("  CHECK 3 FD of the SAME compiled step vs AD, per direction.")
    say("    central = (f(+e)-f(-e))/2e; fwd/bwd are the one-sided quotients.")
    say("    fwd != bwd is SUFFICIENT for a kink but NOT NECESSARY: a switch whose")
    say("    guard is EVEN in the perturbation -- which is what every PPM limiter")
    say("    guard is, being a product of neighbour differences -- flips the same")
    say("    way on both sides, so fwd == bwd == central and the residual is flat")
    say("    in eps while still not being the derivative. A flat, eps-independent")
    say("    residual is therefore evidence of a DEGENERATE BASE POINT, not of a")
    say("    wrong adjoint; re-run with RESEACT_ADJ_UJITTER=1e-3 to tell them apart.")
    for (dn, du_h, dth_h) in dirs
        DU = RX.ConcreteRArray(du_h); DTHd = _todev(dth_h)
        jv = Array(cjvp(UR, DU, TH, DTHd, TR, DT))
        njv = max(norm(jv), 1e-300)
        pred = cvjp === nothing ? dot(lamh, jv) : dot(lam_in, du_h) + _ip(gth, dth_h)
        f0v = Array(cout(UR, TH, TR, DT))
        best = Inf; bestv = (0.0, 0.0, 0.0)
        for epsr in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
            Up = RX.ConcreteRArray(uh .+ epsr .* du_h); Um = RX.ConcreteRArray(uh .- epsr .* du_h)
            fp = Array(cout(Up, _pert(th_host, dth_h, epsr), TR, DT))
            fm = Array(cout(Um, _pert(th_host, dth_h, -epsr), TR, DT))
            cen = (fp .- fm) ./ (2epsr); fwd = (fp .- f0v) ./ epsr; bwd = (f0v .- fm) ./ epsr
            r1 = abs(dot(lamh, cen) - pred) / max(abs(pred), 1e-300)
            r2 = norm(cen .- jv) / njv
            @printf("    %-15s eps=%.0e  scalar=%.3e  central=%.3e  fwd=%.3e  bwd=%.3e\n",
                    dn, epsr, r1, r2, norm(fwd .- jv) / njv, norm(bwd .- jv) / njv)
            if r2 < best; best = r2; bestv = (epsr, r1, r2); end
        end
        @printf("    %-15s BEST eps=%.0e: scalar rel %.3e, vector rel %.3e  %s\n",
                dn, bestv[1], bestv[2], bestv[3], bestv[3] < 1e-6 ? "PASS (<1e-6)" : "FAIL")
        # attribute the residual: which state groups carry ||FD - Jv||?
        epsr = bestv[1]
        Up = RX.ConcreteRArray(uh .+ epsr .* du_h); Um = RX.ConcreteRArray(uh .- epsr .* du_h)
        cen = (Array(cout(Up, _pert(th_host, dth_h, epsr), TR, DT)) .-
               Array(cout(Um, _pert(th_host, dth_h, -epsr), TR, DT))) ./ (2epsr)
        res = abs2.(cen .- jv)
        byg = Dict{String,Float64}()
        for i in 1:N; byg[GROUP_OF[i]] = get(byg, GROUP_OF[i], 0.0) + res[i]; end
        tot = sum(values(byg))
        tot > 0 && println("      residual by state group: ",
            join(["$(p.first) $(round(100p.second/tot; digits=1))%"
                  for p in first(sort(collect(byg); by=last, rev=true), 5)], ", "))
    end
    return cvjp
end

# --------------------------------------------------------------------------- #
# 5. CHECK 4 -- the census, FIRST, because it is cheap and because it is what
#    the whole design rests on: reverse mode cannot cross a stablehlo.while, so
#    there must be none in the module Enzyme is handed. Counted on the RAW
#    (pre-optimization) module too, since an optimizer could remove a
#    trip-count-1 loop and hide the problem. `func.call` is counted alongside,
#    because Enzyme-MLIR's reverse rule for a call is where the chemistry half
#    crashes.
# --------------------------------------------------------------------------- #
DTC = RX.ConcreteRNumber(0.5); DTT = RX.ConcreteRNumber(15.0)
function census(nm, s)
    nw = count(_ -> true, eachmatch(r"stablehlo\.while", s))
    ncall = count(_ -> true, eachmatch(r"func\.call", s))
    callees = Dict{String,Int}()
    for m in eachmatch(r"func\.call\s+@([A-Za-z0-9_.\$]+)", s)
        k = replace(m.captures[1], r"_\d+$" => "")
        callees[k] = get(callees, k, 0) + 1
    end
    @printf("  %-32s lines=%-9d stablehlo.while=%-3d func.call=%-5d %s\n",
            nm, count(==('\n'), s), nw, ncall, nw == 0 ? "while-free" : "HAS WHILE")
    isempty(callees) || println("       callees: ",
        join(["$(p.first)x$(p.second)" for p in first(sort(collect(callees); by=last, rev=true), 8)], ", "))
    return (nw, ncall)
end
if want("census")
    say("\n---- CHECK 4: stablehlo.while / func.call census (raw modules) ----")
    census("chem RHS gC alone",   sprint(show, @code_hlo optimize=false gC(UR, THC, TR)))
    census("transport RHS gT alone", sprint(show, @code_hlo optimize=false gT(UR, THT, TR)))
    census("ROS23 primal jac=:ad", sprint(show, @code_hlo optimize=false ros_out(UR, THC, TR, DTC)))
    census("ROS23 primal jac=:fd", sprint(show, @code_hlo optimize=false ros_out_fd(UR, THC, TR, DTC)))
    census("SSPRK43 primal",       sprint(show, @code_hlo optimize=false ssp_out(UR, THT, TR, DTT)))
    # contrast: the PRODUCTION adaptive solve is exactly what reverse cannot
    # cross -- if this one does not show a while, the census measures nothing.
    adap_T(u, th, t0_, t1_, dt0_) = first(RTI.adaptive_solve(
        (uu, tt, dd, ax) -> RTI.ssprk43_step((x, s) -> gT(x, ax, s), uu, tt, dd, ATOL_T, RTOL),
        u, t0_, t1_, dt0_, RTI.pictrl_ssprk43(), th; clamp_nonneg = true))
    T1R = RX.ConcreteRNumber(T0 + MACRO_DT)
    hlo_ctl = @code_hlo optimize=false adap_T(UR, THT, TR, T1R, DTT)
    census("adaptive_solve (CONTROL)", sprint(show, hlo_ctl))
end

# Stage `addump`: write the jac=:ad chemistry VJP module EXACTLY AS ENZYME IS
# HANDED IT. `optimize=false` stops before the pass pipeline, so this runs no
# DifferentiatePass and SURVIVES the segfault that stage `ros_ad` dies of --
# which is the only way to get the crashing module off this machine. The files
# are the attachment for tools/diag/UPSTREAM_reverse_over_forward.md.
if want("addump")
    dest = get(ENV, "RESEACT_ADJ_HLO",
               joinpath(REPO, "tools", "diag", "reseact_ros_vjp.mlir"))
    mkpath(dirname(dest))
    DTC_ = RX.ConcreteRNumber(0.5)
    say("\n---- addump: pre-pipeline modules ----")
    for (nm, f) in (("ad", () -> @code_hlo optimize=false ros_vjp(UR, THC, LAM, TR, DTC_)),
                    ("fd", () -> @code_hlo optimize=false ros_vjp_fd(UR, THC, LAM, TR, DTC_)),
                    ("rhs", () -> @code_hlo optimize=false gC(UR, THC, TR)))
        p = replace(dest, ".mlir" => "_$nm.mlir")
        t0 = time(); s = sprint(show, f())
        open(p, "w") do io; write(io, s); end
        @printf("  %-4s %8.1f s  %10d bytes  %s\n", nm, time() - t0, filesize(p), p)
        flush(stdout)
    end
end

# Stage `ros_advjp`: ONLY the jac=:ad REVERSE pass, nothing else. `ros_ad`
# compiles the primal and then the JVP before it gets to the VJP, and the
# jac=:ad JVP is the one with the compile-cost problem -- so `ros_ad` can burn
# an hour before it reaches the question this stage exists to answer.
# RESEACT_ADJ_EXCL is a comma-separated `CompileOptions(; excluded_passes=...)`,
# which is how to get past the `concat_broadcast_slice` miscompile if it fires
# here (see tools/diag/UPSTREAM_reverse_over_forward.md, bug B). A verdict taken
# with a pass excluded is a DIFFERENT verdict; the banner prints which.
if want("ros_advjp")
    excl = filter(!isempty, String.(split(get(ENV, "RESEACT_ADJ_EXCL", ""), ",")))
    copts = isempty(excl) ? RX.CompileOptions() : RX.CompileOptions(; excluded_passes = excl)
    say("\n---- ros_advjp : jac=:ad REVERSE only, excluded_passes=$excl ----")
    DTCV = RX.ConcreteRNumber(0.5)
    t0 = time(); cv = @compile compile_options = copts ros_vjp(UR, THC, LAM, TR, DTCV)
    say(@sprintf("  VJP compiled in %.1f s", time() - t0))
    g = cv(UR, THC, LAM, TR, DTCV)
    @printf("  ||lambda_in||=%.6e  ||grad_theta.bufs[1]||=%.6e\n",
            norm(Array(g[1])), norm(Array(first(values(g[2].bufs)))))
end

want("ssprk")  && run_checks("SSPRK43 (transport)", THT, dTHT, th_host_T, dTHT_h, 15.0,
                             ssp_out, ssp_vjp, ssp_jvp)
want("ros_fd") && run_checks("ROS23 (chemistry, jac=:fd)", THC, dTHC, th_host_C, dTHC_h, 0.5,
                             ros_out_fd, ros_vjp_fd, ros_jvp_fd)
want("ros_ad") && run_checks("ROS23 (chemistry, jac=:ad)", THC, dTHC, th_host_C, dTHC_h, 0.5,
                             ros_out, ros_vjp, ros_jvp)
# forward-only twin of ros_ad: the exact-Jacobian step CAN be validated against
# finite differences even though its reverse crashes.
want("ros_adfwd") && run_checks("ROS23 (chemistry, jac=:ad, FORWARD only)", THC, dTHC,
                                th_host_C, dTHC_h, 0.5, ros_out, nothing, ros_jvp)

# --------------------------------------------------------------------------- #
# 6. CHECK 5 -- what the exact Jacobian does to a real adaptive solve.
# --------------------------------------------------------------------------- #
if want("solve")
    say("\n---- CHECK 5: one macro step of chemistry, jac=:fd vs jac=:ad ----")
    mk(jac) = (u, th, t0_, t1_, dt0_) -> RTI.adaptive_solve(
        (uu, tt, dd, ax) -> RTI.ros23_step((x, s) -> gC(x, ax, s), uu, tt, dd,
                                           NS, NC, MASKS, ATOL_C, RTOL;
                                           unrolled = true, jac = jac),
        u, t0_, t1_, dt0_, RTI.pictrl_ros23(), th; clamp_nonneg = true)
    T1 = RX.ConcreteRNumber(T0 + MACRO_DT); DT0 = RX.ConcreteRNumber(0.5)
    res = Dict{Symbol,Any}()
    for jac in (:fd, :ad)
        fn = mk(jac)
        local t0 = time(); c = @compile fn(UR, THC, TR, T1, DT0); tc = time() - t0
        c(UR, THC, TR, T1, DT0)                       # warm: the first call
        t0 = time(); r = c(UR, THC, TR, T1, DT0); ts = time() - t0   # carries setup
        res[jac] = (u = Array(r[1]), nacc = Float64(r[4]), nrej = Float64(r[5]),
                    tc = tc, ts = ts)
        @printf("  jac=%-3s compile %7.1f s  solve %6.2f s  naccept=%6.0f  nreject=%6.0f\n",
                jac, tc, ts, Float64(r[4]), Float64(r[5]))
    end
    du = res[:ad].u .- res[:fd].u
    sc = max.(abs.(res[:ad].u), abs.(res[:fd].u), 1e-30)
    @printf("  state agreement over the macro step: max abs %.3e, max rel %.3e\n",
            maximum(abs.(du)), maximum(abs.(du) ./ sc))
end
say("\nDONE")
