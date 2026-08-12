#!/usr/bin/env julia
# ===========================================================================
# rx_ssprk_basepoint.jl -- rx_adjoint_check.jl's SSPRK43 "state u only" check,
# swept over the BASE POINT instead of over the FD step size.
#
# The ~9.15e-6 AD-vs-FD gap that stage reported is measured at u0, the model's
# DEFAULT initial condition, whose SuperFast fields are spatially UNIFORM.  The
# PPM monotonicity limiters in EarthSciDiscretizations
# (stencils/ppm_slope_mono.esm eq.(1.8), stencils/ppmflux_limit_left.esm and
# ppm_limit_right.esm eq.(1.10)) branch on products of neighbour differences,
# which are EXACTLY ZERO on a uniform field -- so u0 sits precisely on a switch.
# Perturbing by e*v makes those products O(e^2) > 0 for BOTH signs of e, so every
# difference quotient lands on the other branch, at every step size, and the
# usual fwd-vs-bwd kink test cannot see it.
#
# If that is the whole story, jittering the base point off the uniform field
# makes the gap collapse to FD accuracy.  One build, one pair of compiles, the
# base point swept as an ARGUMENT.
#
# MEASURED, 6x6x8, T0=5400, dt=15, "state u only" (||FD - J v|| / ||J v||):
#
#   base point       eps=1e-2   eps=1e-4   eps=1e-6   eps=1e-8
#   u0 (uniform)     9.151e-06  9.151e-06  9.151e-06  9.151e-06   <- FLAT, and
#                      the residual is CO 90.7% / O3 9.3%: the number and the
#                      split Phase 3b reported, now held over four MORE decades
#                      of eps than its sweep reached
#   *(1+1e-6*randn)  1.987e-05  1.986e-05  1.698e-05  1.610e-06
#   *(1+1e-3*randn)  1.967e-05  9.834e-06  9.400e-08  5.875e-09
#   *(1+1e-1*randn)  1.015e-05  9.448e-08  6.082e-11  5.972e-09
#
# The bottom row is an ordinary finite-difference error curve: it falls like
# eps^2 and turns around at 1e-8 on roundoff.  6.082e-11 is the answer to "is
# the Enzyme JVP the derivative of the compiled step" -- yes, to 6e-11, once it
# is asked away from the switching surface.  Every entry still stuck at ~1e-5 is
# one where the FD stencil is WIDER than the base point's distance to a switch,
# so the quotient straddles a branch: those failures are a property of the
# (base point, eps) pair, not of the derivative.
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
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
const RTOL, ATOL_T = 1e-4, 1e-6

say("=== rx_ssprk_basepoint: grid $(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF T0=$(round(Int,T0)) ===")

validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); ff = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, ff
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
        EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi
        i == 1 ? (global u0, p, var_map = u0i, pi, vmi) : nothing
    end
end
foreach(d -> d.materialize!(), dms)
const N = length(u0)
say(@sprintf("BUILD %.2f s   nstates=%d", time() - tb, N))

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

host_bufs = EA.forcing_buffers(fo[1])
g4 = EA.rhs_with_buffers(fo[1])
gT(u, th, t) = g4(u, th.p, t, th.bufs)
dev_bufs = map(RX.ConcreteRArray, host_bufs)
EA.sync_forcing!(dev_bufs, EA.forcing_buffers(fo[1]))
TH = (p = NamedTuple{keys(p)}(map(RX.ConcreteRNumber, values(p))), bufs = dev_bufs)

const NAME_OF = let v = fill("?", N); for (nm, i) in var_map; v[i] = nm; end; v end
const GROUP_OF = [replace(nm, r"\[.*$" => "") for nm in NAME_OF]
const GROUPS = sort(unique(GROUP_OF))
const IDX_OF = Dict(gp => findall(==(gp), GROUP_OF) for gp in GROUPS)

say("\n-- spatial spread of u0 per state group --")
for gp in GROUPS
    vals = u0[IDX_OF[gp]]
    @printf("  %-22s n=%-5d min=% .6e max=% .6e  spread=%.3e\n", gp, length(vals),
            minimum(vals), maximum(vals),
            (maximum(vals) - minimum(vals)) / max(abs(mean(vals)), 1e-300))
end

Random.seed!(20260811)
scale_u = max.(abs.(u0), 1e-12)
vuh = randn(N) .* scale_u             # the harness's direction, same RNG stream
_ = randn(N) ./ scale_u
Random.seed!(777); jitter = randn(N)

ssp_out(u, th, t, dt) = RTI.ssprk43_step_out(gT, u, th, t, dt, ATOL_T, RTOL)
ssp_jvp(u, du, th, dth, t, dt) = RTI.ssprk43_step_jvp(gT, u, du, th, dth, t, dt, ATOL_T, RTOL)
_zero_like(x::Real) = 0.0
_zero_like(x::AbstractArray) = zeros(size(x))
_zero_like(nt::NamedTuple) = NamedTuple{keys(nt)}(map(_zero_like, values(nt)))
_zero_like(tp::Tuple) = map(_zero_like, tp)
_todev(x::Real) = RX.ConcreteRNumber(Float64(x))
_todev(x::AbstractArray) = RX.ConcreteRArray(Array{Float64}(x))
_todev(nt::NamedTuple) = NamedTuple{keys(nt)}(map(_todev, values(nt)))
_todev(tp::Tuple) = map(_todev, tp)
ZTH = _todev(_zero_like((p = p, bufs = host_bufs)))

UR = RX.ConcreteRArray(copy(u0)); VU = RX.ConcreteRArray(vuh)
TR = RX.ConcreteRNumber(T0); DT = RX.ConcreteRNumber(15.0)
t0 = time(); cout = @compile ssp_out(UR, TH, TR, DT); tc1 = time() - t0
t0 = time(); cjvp = @compile ssp_jvp(UR, VU, TH, ZTH, TR, DT); tc2 = time() - t0
say(@sprintf("compile: primal %.1f s, JVP %.1f s", tc1, tc2))

rel(a, b) = norm(a .- b) / max(norm(b), 1e-300)
prim(u) = Array(cout(RX.ConcreteRArray(u), TH, TR, DT))
function sweep(label, ub)
    UB = RX.ConcreteRArray(ub)
    jv = Array(cjvp(UB, VU, TH, ZTH, TR, DT))
    f0 = Array(cout(UB, TH, TR, DT))
    say("  $label")
    for e in (1e-2, 1e-4, 1e-6, 1e-8)
        fp = prim(ub .+ e .* vuh); fm = prim(ub .- e .* vuh)
        cen = (fp .- fm) ./ (2e); fwd = (fp .- f0) ./ e; bwd = (f0 .- fm) ./ e
        @printf("    eps=%.0e  cen=%.3e  fwd=%.3e  bwd=%.3e\n",
                e, rel(cen, jv), rel(fwd, jv), rel(bwd, jv))
    end
    let e = 1e-4
        cen = (prim(ub .+ e .* vuh) .- prim(ub .- e .* vuh)) ./ (2e)
        res = abs2.(cen .- jv); tot = sum(res)
        tot > 0 && println("    residual by group: ", join(
            ["$gp $(round(100*sum(res[IDX_OF[gp]])/tot; digits=1))%" for gp in
             sort(GROUPS; by = gp -> -sum(res[IDX_OF[gp]]))][1:min(4, end)], ", "))
    end
end

say("\n-- SSPRK43 one step, 'state u only', swept over the BASE POINT --")
sweep("u0                        (model default IC, uniform tracer fields)", copy(u0))
for r in (1e-6, 1e-3, 1e-1)
    sweep(@sprintf("u0 .* (1 + %.0e*randn)     (degeneracy broken)", r),
          u0 .* (1 .+ r .* jitter))
end
say("\nDONE")
