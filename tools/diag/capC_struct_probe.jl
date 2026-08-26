#!/usr/bin/env julia
# capC_struct_probe.jl -- STRUCTURAL probe (no build_evaluator, no forcing fetch).
# Can the CHEMISTRY half (split part 2) be built at an arbitrary lane-capacity
# grid shape and fed gathered per-lane inputs? That requires that nothing the
# chemistry RHS TRANSITIVELY needs derive a per-cell quantity from the GRID
# INDEX ITSELF (level index k into Ap/Bp; lon/lat index into solar geometry).
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "_env.jl"))
using EarthSciAST, JSON3
const EA = EarthSciAST
include(joinpath(REPO, "prototypes", "reseact_3d_chem", "split_common.jl"))
using EarthSciASTSplitter: split_system
using Printf

NLON = parse(Int, get(ENV, "RESEACT_NLON", "6"))
NLAT = parse(Int, get(ENV, "RESEACT_NLAT", "6"))
NLEV = parse(Int, get(ENV, "RESEACT_NLEV", "8"))
SLICE = native_slice(nlon = NLON, nlat = NLAT, nlev = NLEV)
MODEL = joinpath(REPO, "reseact.esm")
say(s) = (println(s); flush(stdout))

t0 = time()
file = EA.load_path(MODEL; metaparameters = SLICE.metaparameters)
flat = EA.flatten(file)
pre  = EA.algebraic_states_to_observeds(flat)
flat = EA.promote_downstream_shapes(pre)
promoted = EA.promoted_array_names(pre, flat)
parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
say(@sprintf("load+flatten+split %.1f s", time() - t0))
p2 = parts[2]

names_in(e, acc = Set{String}()) = begin
    if e isa EA.VarExpr; push!(acc, String(e.name))
    elseif e isa EA.OpExpr
        for a in e.args; names_in(a, acc); end
        e.expr_body !== nothing && names_in(e.expr_body, acc)
    end
    acc
end
expr_str(e) = try; EA.to_ascii(e); catch err; "<<to_ascii failed: $err>>" end

# observed definitions inside part 2
obsdef = Dict{String,Any}()
deriv  = Any[]
for eq in p2.equations
    if eq.lhs isa EA.VarExpr && haskey(p2.observed_variables, String(eq.lhs.name))
        get!(obsdef, String(eq.lhs.name), eq.rhs)
    else
        push!(deriv, eq)
    end
end
say(@sprintf("part 2: %d equations, %d observed defs, %d non-observed (derivative) eqs",
             length(p2.equations), length(obsdef), length(deriv)))

# transitive closure of what the derivative equations need
need = Set{String}()
stack = String[]
for eq in deriv; for n in names_in(eq.rhs); push!(stack, n); end; end
while !isempty(stack)
    n = pop!(stack)
    n in need && continue
    push!(need, n)
    haskey(obsdef, n) && for m in names_in(obsdef[n]); m in need || push!(stack, m); end
end
say(@sprintf("  transitive closure of the chemistry RHS: %d names", length(need)))

CONSTNAMES = ["Transport3D.dA","Transport3D.dB","Transport3D.Ap","Transport3D.Bp"]

CONSTNAMES = ["Transport3D.dA","Transport3D.dB","Transport3D.Ap","Transport3D.Bp"]
say("\n---- hybrid-coef const arrays in the chemistry closure ----")
for c in CONSTNAMES; say("   $c  " * (c in need ? "PRESENT" : "absent")); end

# every closure member whose DEFINITION reads a raw grid index
GRIDIDX = Set(["gi","gj","gk","i","j","k"])
say("\n---- closure members whose def reads a RAW GRID INDEX (gi/gj/gk/i/j/k) ----")
let m = 0
    for (nm, rhs) in sort(collect(obsdef), by = first)
        nm in need || continue
        isempty(intersect(names_in(rhs), GRIDIDX)) && continue
        m += 1
        s = expr_str(rhs)
        say(@sprintf("  [%2d] %s\n        = %s", m, nm, length(s) > 1200 ? s[1:1200]*" ..." : s))
    end
    say("  ($m of $(length(need)) closure members)")
end

say("\n---- LEAVES of the chemistry closure (no def in part 2) ----")
leaves = sort([x for x in need if !haskey(obsdef, x)])
say("  $(length(leaves)) leaves: " * join(leaves, ", "))

say("\n---- FULL closure member list ----")
say(join(sort(collect(need)), "\n"))
