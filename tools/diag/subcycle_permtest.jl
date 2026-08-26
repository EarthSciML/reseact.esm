#!/usr/bin/env julia
# ===========================================================================
# subcycle_permtest.jl -- the LANE INDEX ALGEBRA, with no build at all.
# ===========================================================================
# tools/subcycle_chem.jl needs a ~10-minute build and a compile before it can be
# run, so a mistake in its permutation arithmetic is a mistake you find ten
# minutes into a job. Everything the driver supplies is STUBBED here (EA, RX,
# RTI, the var maps, the parameter vector), the file is loaded against those
# stubs, and only its pure host functions are called. Seconds, no build, no
# device -- the same bargain `LevelSubcycle.selftest()` makes for the scheduler.
#
# Loading against the stubs is itself a check: it MACRO-EXPANDS the whole file,
# which is what catches an `@sprintf` whose format string is a concatenation
# ("a" * "b" fails at macro expansion, so it survives `Meta.parseall` and dies
# at load -- it cost slurm 10127221).
#
# WHAT IT PROVES
#   * `lanes_in!` writes EVERY lane (an unwritten lane is a NaN that reaches the
#     error norm and takes the whole batch with it).
#   * the in/out round trip is BIT-IDENTICAL over a random cell subset in random
#     order, against a DELIBERATELY SCRAMBLED `capsel` -- so an implementation
#     that quietly assumed capacity index == (s-1)*C + l is caught rather than
#     accidentally right.
#   * padding lanes really do duplicate real cells.
#   * a whole window's batch list writes every cell EXACTLY ONCE, on the rung its
#     level was packed onto, and reconstructs the state bit-identically.
#   * and the round-trip check FIRES when one lane is written back to the wrong
#     cell -- a check that cannot fail has not passed.
#
#   julia --project=run-model-jl tools/diag/subcycle_permtest.jl
# ===========================================================================
# Exercise tools/subcycle_chem.jl's LANE INDEX ARITHMETIC with no build at all.
# Everything the driver supplies is stubbed; only pure host functions are called.
module Stub
using Printf, Logging, Random, Statistics
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const NS = 13
const NC = 288
const DT0C = 0.5
const T0 = 5400.0
const SYMJAC = false
const JACMODE = :sym
const RTOL = 1e-4
const ATOL_C = 1e-9
const CLAMP = Ref(true)
const MODEL = joinpath(REPO, "reseact.esm")
const GRID_MP = Dict("NLON" => 6, "NLAT" => 6, "NLEV" => 8, "LON0" => 11, "LAT0" => 29)
say(s) = println(s)
# names the file references but this test never calls:
module EA end
module RX
    macro compile(args...); :(nothing); end
end
module RTI end
module RxSymBlockJac end
const merged_param = Dict{String,Any}()
const merged_const = Dict{String,Any}()
const ov = Dict{String,Float64}()
const var_map = Dict{String,Int}()
const p = NamedTuple()
thC(a, b, c) = (p = a, bufs = b)
_devp(x) = x
validate_plan(args...; kw...) = 0.0
split_system(args...; kw...) = nothing
stencil_following_rule(x) = nothing
index_promoted_refs_by_loop!(a, b) = a
include(joinpath(REPO, "tools", "subcycle_chem.jl"))
end

using Printf, Random
const S = Stub
NS, NC = S.NS, S.NC

# A fake rung: capsel is a genuine bijection but a DELIBERATELY SCRAMBLED one,
# so a lanes_in!/lanes_out! that quietly assumed capacity index == (s-1)*C + l
# would be caught rather than accidentally right.
function fake_rung(C::Int, rng)
    perm = shuffle(rng, collect(1:(NS * C)))
    return S.SubRung(C, nothing, zeros(NS * C), nothing, Dict{String,Int}(),
                     Dict{String,Any}(), nothing, nothing, nothing, nothing, nothing,
                     nothing, nothing, nothing, perm, 0.0, 0.0, 0.0)
end

rng = MersenneTwister(7)
u = randn(rng, NS * NC)
fails = 0
chk(c, msg) = (c || (global fails += 1; println("FAIL: ", msg)); c)

# 1. round trip through every ladder rung, over a random subset in random order
for C in S.SUB_LADDER
    r = fake_rung(C, rng)
    n = min(C, NC)
    cellpos = shuffle(rng, collect(1:NC))[1:n]
    lanepos = [cellpos[l <= n ? l : mod1(l - n, n)] for l in 1:C]
    uc = S.lanes_in!(fill(NaN, NS * C), u, r, lanepos)
    chk(!any(isnan, uc), "C=$C: lanes_in! left $(count(isnan, uc)) lanes unwritten")
    uo = fill(NaN, NS * NC)
    S.lanes_out!(uo, uc, r, lanepos, n)
    bad = count(c -> any(s -> !isequal(uo[(s - 1) * NC + c], u[(s - 1) * NC + c]), 1:NS), cellpos)
    chk(bad == 0, "C=$C: $bad of $n cells did not round trip bit-identically")
    # padding lanes must be duplicates of real lanes, never garbage
    for l in (n + 1):C, s in 1:NS
        chk(isequal(uc[r.capsel[(s - 1) * C + l]], u[(s - 1) * NC + lanepos[l]]),
            "C=$C lane $l: padding does not duplicate a real cell")
    end
end

# 2. a WHOLE WINDOW's batch list: every cell written exactly once, by the rung
#    its level was packed onto.
for trial in 1:20
    need = exp.(9 .* rand(rng, NC))          # spans j = 0 .. 9
    levels = S.LevelSubcycle.assign_levels(need, S.SUB_JMAX)
    batches = S.LevelSubcycle.pack(levels, S.SUB_LADDER; jmax = S.SUB_JMAX)
    rungs = Dict(C => fake_rung(C, rng) for C in S.SUB_LADDER)
    uo = fill(NaN, NS * NC)
    seen = zeros(Int, NC)
    for b in batches
        r = rungs[b.cap]
        nreal = length(b.cells)
        lanepos = [b.cells[l <= nreal ? l : mod1(l - nreal, nreal)] for l in 1:b.cap]
        uc = S.lanes_in!(fill(NaN, NS * b.cap), u, r, lanepos)
        S.lanes_out!(uo, uc, r, lanepos, nreal)
        for c in b.cells; seen[c] += 1; end
        chk(all(levels[c] == b.j for c in b.cells), "a cell landed in the wrong level's batch")
    end
    chk(all(==(1), seen), "trial $trial: $(count(!=(1), seen)) cells missing or duplicated")
    chk(!any(isnan, uo), "trial $trial: $(count(isnan, uo)) states never written back")
    chk(all(isequal(uo[i], u[i]) for i in eachindex(u)),
        "trial $trial: the window round trip is not bit-identical")
end

# 3. the SubStats monoid
a = S.SubStats(); a.calls = 3; a.cellsteps = 10.0; a.maxlevel = 2
b = S.SubStats(); b.calls = 4; b.cellsteps = 5.0;  b.maxlevel = 7
c = a + b
chk(c.calls == 7 && c.cellsteps == 15.0 && c.maxlevel == 7, "SubStats + is wrong")

# NEGATIVE CONTROL: the round-trip check has to be able to fail. Send ONE lane
# back to the wrong cell and require the same comparison to report it.
let C = maximum(S.SUB_LADDER), r = fake_rung(C, rng), n = min(C, NC)
    cellpos = collect(1:n)
    lanepos = [cellpos[l <= n ? l : mod1(l - n, n)] for l in 1:C]
    uc = S.lanes_in!(fill(NaN, NS * C), u, r, lanepos)
    wrong = copy(lanepos); wrong[1], wrong[2] = wrong[2], wrong[1]
    uo = fill(NaN, NS * NC)
    S.lanes_out!(uo, uc, r, wrong, n)
    bad = count(c -> any(s -> !isequal(uo[(s - 1) * NC + c], u[(s - 1) * NC + c]), 1:NS), cellpos)
    chk(bad > 0, "NEGATIVE CONTROL: two lanes written back swapped went UNNOTICED")
    println("  negative control (two lanes written back swapped): $bad of $n cells differ -- FIRED")
end

println(fails == 0 ? "PERMTEST PASS" : "PERMTEST FAIL ($fails)")
exit(fails == 0 ? 0 : 1)
