#!/usr/bin/env julia
# ===========================================================================
# rhs_kink_probe.jl -- HOST-ONLY probe of the AD-vs-FD gap in the transport RHS.
#
# No Reactant, no Enzyme, no integrator: just the emitted :oop Julia function,
# ForwardDiff through it, and central differences of it.  If the ~9e-6 gap
# reproduces HERE, nothing about the traced lowering is implicated and the
# question becomes "is the map differentiable at u0 along v".
#
# The sweep is over BASE POINTS.  u0 is the model's default initial condition,
# which is SPATIALLY UNIFORM per species; the PPM CW84 monotonicity limiter's
# local-extremum test (qr-qi)*(qi-ql) <= 0 is then satisfied with EXACT zero in
# every interior cell.  Any finite perturbation e*v makes that product
# O(e^2) > 0 in about half the cells and flips the branch -- identically for
# +e and -e, which is exactly why the one-sided quotients agree with each other
# and why the gap does not shrink with eps.  Jittering the base point off the
# uniform field should therefore make the gap vanish.
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Printf, Random, Statistics, Logging
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using ForwardDiff
const EA = EarthSciAST

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
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
const PART = parse(Int, get(ENV, "RESEACT_PART", "1"))

say("=== rhs_kink_probe: grid $(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF T0=$(round(Int,T0)) part=$PART ===")

validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); ff = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, ff
    file = EA.load_path(MODEL; metaparameters = GRID_MP)
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

host_bufs = EA.forcing_buffers(fo[PART])
g = EA.rhs_with_buffers(fo[PART])
ghost(u) = g(u, p, T0, host_bufs)

const NAME_OF = let v = fill("?", N); for (nm, i) in var_map; v[i] = nm; end; v end
const GROUP_OF = [replace(nm, r"\[.*$" => "") for nm in NAME_OF]
const GROUPS = sort(unique(GROUP_OF))
const IDX_OF = Dict(gp => findall(==(gp), GROUP_OF) for gp in GROUPS)

say("\n-- is u0 spatially uniform per state group? --")
for gp in GROUPS
    ii = IDX_OF[gp]; vals = u0[ii]
    @printf("  %-22s n=%-5d min=% .6e max=% .6e  spread=%.3e\n",
            gp, length(ii), minimum(vals), maximum(vals),
            (maximum(vals) - minimum(vals)) / max(abs(mean(vals)), 1e-300))
end

Random.seed!(20260811)
scale_u = max.(abs.(u0), 1e-12)
vuh = randn(N) .* scale_u              # SAME direction rx_adjoint_check.jl uses
_ = randn(N) ./ scale_u                # keep the RNG stream aligned
Random.seed!(777)
jitter = randn(N)

rel(a, b) = norm(a .- b) / max(norm(b), 1e-300)
function groupattr(a, b)
    res = abs2.(a .- b); tot = sum(res); tot == 0 && return "identical"
    join(["$gp $(round(100*sum(res[IDX_OF[gp]])/tot; digits=1))%" for gp in
          sort(GROUPS; by = gp -> -sum(res[IDX_OF[gp]]))][1:min(4, end)], ", ")
end

function probe(label, ub)
    ad = ForwardDiff.derivative(e -> ghost(ub .+ e .* vuh), 0.0)
    f0 = ghost(ub)
    say("  $label   ||J v|| = $(@sprintf("%.6e", norm(ad)))")
    for e in (1e-2, 1e-4, 1e-6, 1e-8)
        fp = ghost(ub .+ e .* vuh); fm = ghost(ub .- e .* vuh)
        cen = (fp .- fm) ./ (2e); fwd = (fp .- f0) ./ e; bwd = (f0 .- fm) ./ e
        @printf("    eps=%.0e  cen=%.3e fwd=%.3e bwd=%.3e   | fwd-bwd |=%.3e\n",
                e, rel(cen, ad), rel(fwd, ad), rel(bwd, ad), rel(fwd, bwd))
    end
    let e = 1e-4
        cen = (ghost(ub .+ e .* vuh) .- ghost(ub .- e .* vuh)) ./ (2e)
        println("    residual (cen vs AD) by group: ", groupattr(cen, ad))
    end
    return ad
end

say("\n-- BASE POINT SWEEP: AD vs FD of the RHS along the harness direction --")
ad_u0 = probe("u0            (model default IC)", copy(u0))
for r in (1e-12, 1e-9, 1e-6, 1e-3)
    probe(@sprintf("u0*(1+%.0e*randn)", r), u0 .* (1 .+ r .* jitter))
end

say("\n-- the derivative's own discontinuity at u0: AD at u0 + delta*v --")
say("   (if u0 sits ON a limiter switch, an INFINITESIMAL step along v already")
say("    puts AD on the branch the finite differences see)")
cen_ref = let e = 1e-4
    (ghost(u0 .+ e .* vuh) .- ghost(u0 .- e .* vuh)) ./ (2e)
end
for d in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6)
    ad = ForwardDiff.derivative(e -> ghost(u0 .+ (d + e) .* vuh), 0.0)
    @printf("    delta=%.0e   ||AD(u0+delta v) - AD(u0)||/||AD(u0)|| = %.3e   vs central FD: %.3e\n",
            d, rel(ad, ad_u0), rel(ad, cen_ref))
end
say("\n-- and along -v --")
for d in (1e-12, 1e-8)
    ad = ForwardDiff.derivative(e -> ghost(u0 .+ (-d + e) .* vuh), 0.0)
    @printf("    delta=-%.0e  vs AD(u0): %.3e   vs central FD: %.3e\n",
            d, rel(ad, ad_u0), rel(ad, cen_ref))
end
say("\nDONE")
