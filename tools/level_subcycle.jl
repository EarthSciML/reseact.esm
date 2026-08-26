# ===========================================================================
# level_subcycle.jl -- the DYADIC LEVEL SCHEDULE for the cell-local half.
# ===========================================================================
# The scheduling half of variable per-cell time stepping. It owns no solver and
# no evaluator: given a per-cell "steps this window needs" vector it produces
# the BATCH LIST a driver runs, and the permutation the adjoint replays. Pure
# index arithmetic, so it is testable without a build (`selftest()` below).
#
# WHY DYADIC RATHER THAN FREE-FORM PER-CELL dt. Three reasons, and the first is
# the one that decides it:
#   1. `t` STAYS A SCALAR WITHIN A LEVEL. With a free per-cell dt every cell
#      sits at its own time, so the RHS needs a lane-varying `t` -- which pushes
#      all time-dependent forcing interpolation, knot locate included, out of
#      the shared lane-invariant CSE tier and into the per-lane tier, paying
#      back much of the win. With levels every lane in a call shares a time and
#      the RHS signature is untouched.
#   2. `ros23_step` is untouched: `dt` is still a scalar per call.
#   3. LEVELS NEVER HAVE TO INTERLEAVE. Local time stepping for a hyperbolic PDE
#      must interleave because cells exchange fluxes. A pointwise sub-system does
#      not: each level runs independently to completion and they need only agree
#      at the window boundary.
#
# ===========================================================================
# MEASURED 2026-08-24 (slurm 10127723 / 10128099), AND IT IS THE DECIDING FACT
# FOR THIS WHOLE SCHEME. The 7.53x "ideal" and 5.14x "dyadic" below are computed
# from `need = SUM over the accepted sequence of dd/dtc(t)` -- the substep count
# for a per-cell VARIABLE dt. But a level is `2^j` EQUAL steps across the window,
# so it must resolve the cell's WORST INSTANT, i.e. `W / min_t dtc(t)`. Those are
# not the same number and on ReSEACT they are far apart:
#
#   CONUS 13x7x72 (NC = 6552), one 300 s window, 75 accepted global steps:
#     integral      need: median  10.6, max  130  -> j0=1360 j1=395 j2=320 j3=577
#                                                    j4=1619 j5=2044 j6=194 j7=40 j8=3
#     worst-instant need: median  50.2, max 1.4e4 -> spreads to j9=636, j10=1806
#     ratio worst/integral: median 5.1x, max 655.4x
#     cell-steps vs 4.914e5 global: INTEGRAL 1.177e5 = 4.18x | LEVEL 2.363e6 = 0.21x
#
#   The integral histogram above REPRODUCES this file's own CONUS histogram
#   (slurm 10123660) term for term, which is the control: the two quantities are
#   being computed on the same window by the same rule, and they still differ 20x
#   in the delivered cost.
#
#   6x6x8 (NC = 288), one 300 s window:
#     integral      need (variable dt): median   25.9, max  147   -> levels j5-j8
#     worst-instant need (a LEVEL):     median  938.9, max 6452   -> every cell j10
#     ratio worst/integral:             median 37.5x, max 207.8x
#     cell-steps vs 3.341e4 global: INTEGRAL 1.232e4 = 2.71x | LEVEL 2.949e5 = 0.11x
#
# AND AT CONUS THE SCHEME DOES NOT EVEN TERMINATE at a sane cap: a 512-cell batch
# still REJECTED at level 14 (16,384 equal substeps of 0.018 s), because some cell
# in it needs that dt somewhere inside the window. jmax is an accuracy floor, and
# on this model the floor is far below where the projection put it.
#
# So the projection is an upper bound that an EQUAL-SUBSTEP level cannot reach:
# it banks temporal adaptivity the scheme does not have. The end-to-end forward
# path measures 0.023x at this grid -- and NOT because of padding (0.0% waste) or
# per-call dispatch (2.3-3.1 ms/call, CHEAPER per call than the global program).
# `assign_levels` is honest about what it is given; the caller has to feed it the
# worst-instant demand, and then pay for it.
#
# WHAT WOULD MAKE LEVELS PAY, in the order worth trying: a SHORTER window (the
# gap is temporal variation WITHIN W, so it shrinks with W); re-assigning levels
# at sub-window checkpoints rather than once per macro step; or abandoning equal
# substeps for a per-cell controller, which is a different scheme than this file.
# ===========================================================================
#
# WHY A LADDER OF CAPACITIES AND NOT ONE. Level j costs `2^j x (lanes run)`, and
# a level's lanes are rounded up to a whole chunk of the build's capacity. The
# sparse HIGH levels are therefore ruinous at a single large capacity: measured
# at CONUS (slurm 10122530), levels 6-8 hold 0.5% of the cell-windows and, at
# C = 1024, cost more than half the total -- the dyadic scheme's 5.14x saving is
# entirely spent on padding and it lands at 0.75x, SLOWER than the global
# controller it replaces. On a ladder (8/32/128/512/2048) padding waste is 2.7%
# and the saving is 5.00x. A capacity build is seconds (tools/capacity_chem.jl),
# so the ladder costs a handful of builds, not a grid's worth.
# ===========================================================================
module LevelSubcycle

using Printf

"""
    assign_levels(need, jmax) -> Vector{Int}

`need[c]` is the number of steps cell `c` would need over the window (the
per-cell error norm extrapolated with the embedded pair's own exponent). The
level is `ceil(log2 n)`, so `2^j >= n`: bucketing only ever gives a cell MORE
steps than it asked for, never fewer. Clamped at `jmax`, which is therefore an
accuracy floor and not a free knob -- a window whose cells reach it is a window
where the schedule is under-resolving, and the caller must say so.
"""
function assign_levels(need::AbstractVector{<:Real}, jmax::Int)
    lv = Vector{Int}(undef, length(need))
    @inbounds for c in eachindex(need)
        n = need[c]
        lv[c] = n <= 1 ? 0 : clamp(ceil(Int, log2(n)), 0, jmax)
    end
    return lv
end

"One batch: `2^j` steps of `dt = W/2^j` over `cells`, run on the capacity-`cap` build."
struct Batch
    j::Int
    cap::Int
    cells::Vector{Int}      # reference cell ids, length <= cap; the rest is padding
end
Base.show(io::IO, b::Batch) = @printf(io, "Batch(j=%d dt=W/%d cap=%d cells=%d pad=%d)",
                                      b.j, 2^b.j, b.cap, length(b.cells), b.cap - length(b.cells))

"""
    pack(levels, ladder; jmax) -> Vector{Batch}

Group cells by level and cover each level with ladder rungs: the LARGEST rung
that still fits while one does, then one rung for the remainder. Taking the
smallest rung >= n up front instead is the trap that makes a ladder look worse
the more rungs it has (1,611 cells billed a 4,096 chunk rather than 1,024+1,024).

The returned order is the RECORD the adjoint replays: batches in this order,
cells within a batch in this order.
"""
function pack(levels::Vector{Int}, ladder::Vector{Int}; jmax::Int = maximum(levels))
    L = sort(ladder)
    isempty(L) && throw(ArgumentError("pack: the capacity ladder is empty"))
    batches = Batch[]
    for j in 0:jmax
        cells = findall(==(j), levels)
        isempty(cells) && continue
        pos = 1; rem = length(cells)
        while rem >= L[1]
            cap = L[findlast(<=(rem), L)]
            push!(batches, Batch(j, cap, cells[pos:(pos + cap - 1)]))
            pos += cap; rem -= cap
        end
        rem > 0 && push!(batches, Batch(j, L[1], cells[pos:end]))
    end
    return batches
end

"Lane-steps a batch list costs (padding included) and the ideal it is measured against."
function cost(batches::Vector{Batch})
    lanes = 0.0; used = 0.0
    for b in batches
        lanes += 2.0^b.j * b.cap
        used  += 2.0^b.j * length(b.cells)
    end
    return (; lanes, used, waste = (lanes - used) / max(lanes, eps()))
end

"""
    capacities(batches) -> Vector{Int}

The distinct lane capacities a batch list needs a build for. Call it BEFORE the
window so the builds/compiles are warm; a capacity appearing here for the first
time mid-window is a stall, not a failure.
"""
capacities(batches::Vector{Batch}) = sort(unique(b.cap for b in batches))

# --------------------------------------------------------------------------- #
# Self-test. No build, no solver -- the scheduler's contract is index algebra
# and it is worth being able to check it in a second.
# --------------------------------------------------------------------------- #
function selftest(; verbose::Bool = true)
    ok = true
    chk(c, msg) = (c || (ok = false; println("FAIL: ", msg)); c)

    lv = assign_levels([0.4, 1.0, 1.1, 2.0, 2.1, 8.0, 9.0, 1e9], 6)
    chk(lv == [0, 0, 1, 1, 2, 3, 4, 6], "assign_levels: got $lv")
    chk(all(2 .^ lv[1:7] .>= [1, 1, 1.1, 2, 2.1, 8, 9]), "a level must never under-resolve its cell")

    levels = vcat(fill(0, 700), fill(4, 1600), fill(6, 107), fill(8, 2))
    ladder = [8, 32, 128, 512, 2048]
    bs = pack(levels, ladder; jmax = 8)
    # every cell appears exactly once, in exactly one batch, at its own level
    seen = zeros(Int, length(levels))
    for b in bs
        chk(length(b.cells) <= b.cap, "batch overfull: $b")
        chk(b.cap in ladder, "batch capacity $(b.cap) is not a ladder rung")
        for c in b.cells
            seen[c] += 1
            chk(levels[c] == b.j, "cell $c has level $(levels[c]) but landed in a level-$(b.j) batch")
        end
    end
    chk(all(==(1), seen), "$(count(!=(1), seen)) cells are missing or duplicated")

    co = cost(bs)
    chk(co.used == sum(2.0^levels[c] for c in eachindex(levels)), "cost.used disagrees with the levels")
    chk(co.waste < 0.10, "padding waste $(round(100co.waste, digits=1))% on a five-rung ladder is too high")
    # the whole point: ONE big capacity must be measurably worse
    co1 = cost(pack(levels, [2048]; jmax = 8))
    chk(co1.lanes > 3 * co.lanes,
        "single-capacity cost $(co1.lanes) vs ladder $(co.lanes) -- the ladder should be several times cheaper")
    if verbose
        @printf("  ladder %s: %.4g lane-steps, waste %.1f%%, %d batches, capacities %s\n",
                string(ladder), co.lanes, 100co.waste, length(bs), string(capacities(bs)))
        @printf("  single 2048   : %.4g lane-steps, waste %.1f%%\n", co1.lanes, 100 * co1.waste)
        println(ok ? "  LevelSubcycle.selftest PASS" : "  LevelSubcycle.selftest FAIL")
    end
    return ok
end

end # module
