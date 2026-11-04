# Load, validate and flatten reseact.esm, and check that the deposition sink
# lands on the chemistry's own species.
#
#   julia tools/diag/flatten_check.jl            # uses RESEACT_RXENV (see _env.jl)
#   RESEACT_RXENV=/path/to/env julia tools/diag/flatten_check.jl
#
# Needs only EarthSciAST in the environment (no Reactant, no data): the check
# stops at the flattened system. Exits non-zero on the first failure.
#
# What it asserts, beyond "loads, validates, flattens":
#   * no state survives under the SuperFastDepositionSink namespace -- the
#     sink's seven species unknowns are consumed by the `translate` of the
#     SuperFast + SuperFastDepositionSink operator_compose, so the flattened
#     system integrates SuperFast.X and nothing else;
#   * each of those seven SuperFast.X tendencies carries a contribution authored
#     by SuperFastDepositionSink, and that contribution reads SuperFast.X (the
#     sink's own spelling was retargeted, not left dangling).
#
# Result (2026-11-02, EarthSciAST 0.8.0 from the General registry, Julia 1.12.6,
# EarthSciModels main 80f303c, EarthSciDiscretizations main 755a289):
#   valid; 794 equations, 13 states (Transport3D.m + the 12 SuperFast species);
#   sink term present on O3 NO2 NO HNO3 H2O2 CH2O CH3OOH. About 2 min, mostly
#   compilation. Before the translate entries, flatten raised
#   operator_compose_ambiguous_bare_name on SuperFast.O3; before atol took
#   units of area, validate reported unit_inconsistency on NEIRegrid.E_*.
include(joinpath(@__DIR__, "_env.jl"))

const MODEL = normpath(joinpath(@__DIR__, "..", "..", "reseact.esm"))
const SINK_SPECIES = ("O3", "NO2", "NO", "HNO3", "H2O2", "CH2O", "CH3OOH")

fail(msg) = (println(stderr, "FAIL: ", msg); exit(1))

file = EarthSciAST.load_path(MODEL)
println("loaded ", MODEL)

vr = EarthSciAST.validate(file)
if !vr.is_valid
    foreach(e -> println(stderr, "  schema: ", e), vr.schema_errors)
    foreach(e -> println(stderr, "  ", e.path, " [", e.error_type, "] ", e.message),
            vr.structural_errors)
    fail("validate: $(length(vr.schema_errors)) schema, $(length(vr.structural_errors)) structural errors")
end
println("valid")

fl = EarthSciAST.flatten(file)
states = collect(keys(fl.state_variables))
println("flattened: $(length(fl.equations)) equations, $(length(states)) states")

stray = filter(s -> startswith(s, "SuperFastDepositionSink."), states)
isempty(stray) || fail("sink species survived as their own states: $stray")

# The dependent variable an equation's LHS differentiates, lifted or not.
function dependent(eq)
    m = match(r"OpExpr\(\"D\", ASTExpr\[(?:OpExpr\(\"index\", ASTExpr\[)?VarExpr\(\"([^\"]+)\"\)",
              sprint(show, eq.lhs))
    m === nothing ? nothing : m.captures[1]
end

for x in SINK_SPECIES
    name = "SuperFast.$x"
    name in states || fail("$name is not a state of the flattened system")
    i = findfirst(eq -> dependent(eq) == name, fl.equations)
    i === nothing && fail("no D($name) equation")
    sink = filter(c -> any(o -> o.component == "SuperFastDepositionSink", c.origins),
                  fl.equation_contributions[i])
    isempty(sink) && fail("D($name) has no SuperFastDepositionSink contribution")
    for c in sink
        occursin("VarExpr(\"$name\")", sprint(show, c.expr)) ||
            fail("the sink term on D($name) does not read $name")
    end
end
println("sink term present on ", join(SINK_SPECIES, " "))
println("OK")
