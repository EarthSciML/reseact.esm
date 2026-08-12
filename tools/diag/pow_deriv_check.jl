# Construct-level bisect of the caller's first hypothesis: does EarthSciAST's
# `^` (and `exp`, `log10`) differentiate consistently on the HOST (ForwardDiff
# through the emitted Julia) and under REACTANT (Enzyme forward over the traced
# module), and does each agree with finite differences of its own primal?
#
# Exponents are the ones SuperFast actually uses: 2, 3, 1.3, 6.1, and the
# `c^x` form with a positive literal base (0.6^x).  Also the one form that
# appears in the TRANSPORT half: (a-b)^2 with a NEGATIVE base, which
# src/tree_walk/compile.jl warns can poison a gradient with NaN.
#
#   julia --project=<env with EarthSciAST + Reactant + ForwardDiff> \
#       tools/diag/pow_deriv_check.jl
using EarthSciAST, ForwardDiff, Printf, Reactant
const ESM = EarthSciAST
const RX = Reactant
const EZ = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end

_Dt(v) = Dict{String,Any}("op" => "D", "args" => Any[v], "wrt" => "t")
_o(o, a...) = Dict{String,Any}("op" => o, "args" => Any[a...])
_state(v) = Dict{String,Any}("type" => "state", "default" => v)

# each case: (name, rhs-for-D(x), rhs-for-D(y), x0, y0)
cases = [
    ("x^2",            _o("^", "x", 2),                 0.0, 1.7,  -0.3),
    ("x^3",            _o("^", "x", 3),                 0.0, 1.7,  -0.3),
    ("x^1.3",          _o("^", "x", 1.3),               0.0, 1.7,  -0.3),
    ("x^6.1",          _o("^", "x", 6.1),               0.0, 1.7,  -0.3),
    ("0.6^x",          _o("^", 0.6, "x"),               0.0, 1.7,  -0.3),
    ("(300/x)^3",      _o("^", _o("/", 300, "x"), 3),   0.0, 290.0, -0.3),
    ("(y-x)^2  y-x<0", _o("^", _o("-", "y", "x"), 2),   0.0, 1.7,  -0.3),
    ("exp(-x)",        _o("exp", _o("neg", "x")),       0.0, 1.7,  -0.3),
    ("log10(x)",       _o("log10", "x"),                0.0, 1.7,  -0.3),
]

@printf("%-18s %-14s %-12s %-12s %-12s\n", "case", "host AD", "host FD", "traced AD", "trx FD")
for (nm, rx, ry, x0, y0) in cases
    doc = Dict{String,Any}("esm" => "0.5.0",
        "metadata" => Dict{String,Any}("name" => "POW"),
        "models" => Dict{String,Any}("M" => Dict{String,Any}(
            "variables" => Dict{String,Any}("x" => _state(x0), "y" => _state(y0)),
            "equations" => Any[
                Dict{String,Any}("lhs" => _Dt("x"), "rhs" => rx),
                Dict{String,Any}("lhs" => _Dt("y"), "rhs" => _o("*", ry, "y"))])))
    f, u0, p, _... = ESM.build_evaluator(doc; form = :oop)
    g(u) = f(u, p, 0.0)
    v = [1.0, 0.37]
    had = ForwardDiff.derivative(e -> g(u0 .+ e .* v), 0.0)
    hfd = (g(u0 .+ 1e-6 .* v) .- g(u0 .- 1e-6 .* v)) ./ 2e-6

    U = RX.ConcreteRArray(collect(u0)); V = RX.ConcreteRArray(v)
    gout(u) = g(u)
    gjvp(u, du) = only(EZ.autodiff(EZ.Forward, gout, EZ.Duplicated, EZ.Duplicated(u, du)))
    cout = @compile gout(U)
    cjvp = @compile gjvp(U, V)
    tad = Array(cjvp(U, V))
    tfd = (Array(cout(RX.ConcreteRArray(u0 .+ 1e-6 .* v))) .-
           Array(cout(RX.ConcreteRArray(u0 .- 1e-6 .* v)))) ./ 2e-6
    @printf("%-18s % -14.8g % -12.8g % -12.8g % -12.8g   |hostAD-trxAD|/|hostAD|=%.2e\n",
            nm, had[1], hfd[1], tad[1], tfd[1],
            abs(had[1] - tad[1]) / max(abs(had[1]), 1e-300))
end
