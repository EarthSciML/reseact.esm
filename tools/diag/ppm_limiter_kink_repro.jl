# Minimal reproducer for the AD-vs-FD gap seen in ReSEACT's SSPRK43/transport step.
#
# The construct is the CW84 (Colella & Woodward 1984 eq. 1.8) monotonized-slope
# limiter that EarthSciDiscretizations' PPM stencils use
# (grids/latlon3d/stencils/ppm_slope_mono.esm, ppm_lev_slope_mono.esm) and the
# eq.(1.10) parabola limiter (ppmflux_limit_left.esm, ppm_limit_right.esm):
#
#     ifelse((a_{i+1} - a_i) * (a_i - a_{i-1}) > 0,  <limited slope>,  0)
#
# On a SPATIALLY UNIFORM field the guard product is EXACTLY 0, so `> 0` is false
# and the branch taken is the constant 0.  Its derivative is 0.  But a finite
# perturbation e*v makes the product e^2 * (dv_{i+1}-dv_i)*(dv_i-dv_{i-1}),
# which is POSITIVE in about half the cells FOR BOTH SIGNS OF e -- so every
# finite-difference quotient, one-sided or central, at every step size, lands on
# the OTHER branch.  AD and FD disagree by an amount that does not shrink with
# eps and that forward/backward/central quotients agree on, which is exactly the
# signature that was mistaken for "AD is not the derivative of the program".
#
#     julia --project=<env with EarthSciAST> tools/diag/ppm_limiter_kink_repro.jl
using EarthSciAST, ForwardDiff, Printf, Random, LinearAlgebra
const ESM = EarthSciAST

_Dt(v) = Dict{String,Any}("op" => "D", "args" => Any[v], "wrt" => "t")
_o(o, a...) = Dict{String,Any}("op" => o, "args" => Any[a...])
_state(v) = Dict{String,Any}("type" => "state", "default" => v)

# CW84 eq.(1.8) monotonized slope of cell `b`, neighbours `a` and `c`.
mono_slope(a, b, c) = _o("ifelse",
    _o(">", _o("*", _o("-", c, b), _o("-", b, a)), 0),
    _o("min", _o("*", 2, _o("min", _o("abs", _o("-", b, a)), _o("abs", _o("-", c, b)))),
              _o("*", 0.5, _o("-", c, a))),
    0)

const NCELL = 12
vars = Dict{String,Any}("c$i" => _state(1.0) for i in 1:NCELL)
eqs = Any[]
for i in 1:NCELL
    rhs = (i == 1 || i == NCELL) ? _o("*", 0.0, "c$i") :
          mono_slope("c$(i-1)", "c$i", "c$(i+1)")
    push!(eqs, Dict{String,Any}("lhs" => _Dt("c$i"), "rhs" => rhs))
end
doc = Dict{String,Any}("esm" => "0.5.0",
    "metadata" => Dict{String,Any}("name" => "PPMKink"),
    "models" => Dict{String,Any}("M" => Dict{String,Any}(
        "variables" => vars, "equations" => eqs)))

f, u0, p, _... = ESM.build_evaluator(doc; form = :oop)
g(u) = f(u, p, 0.0)
N = length(u0)
@printf("states = %d, u0 = %s  (spatially UNIFORM)\n", N, string(u0))

Random.seed!(4)
v = randn(N)
# symmetric denominator: at the degenerate base point AD is IDENTICALLY ZERO,
# and dividing by it would print 1e300 instead of "these are different vectors".
rel(a, b) = norm(a .- b) / max(norm(a), norm(b), 1e-300)

function report(label, ub)
    ad = ForwardDiff.derivative(e -> g(ub .+ e .* v), 0.0)
    f0 = g(ub)
    @printf("\n%s\n  AD  J*v = %s\n", label, string(round.(ad; sigdigits = 6)))
    for e in (1e-2, 1e-4, 1e-6, 1e-8)
        fp = g(ub .+ e .* v); fm = g(ub .- e .* v)
        cen = (fp .- fm) ./ (2e); fwd = (fp .- f0) ./ e; bwd = (f0 .- fm) ./ e
        @printf("  eps=%.0e  ||cen-AD||/||AD||=%.3e  fwd=%.3e  bwd=%.3e  ||fwd-bwd||/||cen||=%.3e\n",
                e, rel(cen, ad), rel(fwd, ad), rel(bwd, ad),
                norm(fwd .- bwd) / max(norm(cen), 1e-300))
    end
    let e = 1e-4
        cen = (g(ub .+ e .* v) .- g(ub .- e .* v)) ./ (2e)
        @printf("  FD  J*v = %s\n", string(round.(cen; sigdigits = 6)))
    end
    ad
end

ad0 = report("BASE POINT = u0 (uniform: every guard product is exactly 0)", copy(u0))
Random.seed!(9)
for r in (1e-12, 1e-6, 1e-2)
    report(@sprintf("BASE POINT = u0 .* (1 + %.0e*randn)  (degeneracy broken)", r),
           u0 .* (1 .+ r .* randn(N)))
end

println("\n-- the derivative jumps for an INFINITESIMAL displacement along v --")
cen_ref = let e = 1e-4; (g(u0 .+ e .* v) .- g(u0 .- e .* v)) ./ (2e); end
for d in (0.0, 1e-14, 1e-10, -1e-14, -1e-10)
    ad = ForwardDiff.derivative(e -> g(u0 .+ (d + e) .* v), 0.0)
    @printf("  delta=%+.0e  ||AD-AD(u0)||/||AD(u0)||=%.3e   ||AD-FD||/||FD||=%.3e\n",
            d, rel(ad, ad0), rel(ad, cen_ref))
end
