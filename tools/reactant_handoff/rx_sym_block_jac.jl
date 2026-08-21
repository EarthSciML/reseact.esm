# The SYMBOLIC block Jacobian, as something a TRACED consumer can call.
#
# EarthSciASTDiff's `prepare_jacobian` returns a `JacobianEvaluator` whose call
# operator is HOST-ONLY by construction: it scatters `u` into a persistent
# buffer (`jac.uj[s] = u[i]`), calls the band model, then accumulates into a
# `SparseMatrixCSC`'s `nzval` (`J.nzval[pos] += duj[slot]`). Mutation of host
# scratch, a sparse container, and a scatter -- none of which trace.
#
# Everything it does is nevertheless a STATIC permutation, known once the model
# is built. This module turns that permutation inside out into the two GATHERS a
# traced program can execute, and hands the result to `rx_traced_integrator.jl`
# in the exact shape `blocksolve` already consumes (`Matrix{Any}` NS x NS, each
# entry a length-NC traced vector).
#
#   u  -> uj  : `jac.umap` is a scatter (uj[umap[i]] = u[i]). Inverted, it is a
#               gather `uj[k] = u[gidx[k]]` with gidx[k] = 0 where no state
#               feeds slot k. Padding `u` with a leading 0.0 makes 0 a legal
#               index, so the whole map is ONE gather with no branch -- and the
#               host operator leaves exactly those slots at 0.0 too
#               (`jac.uj = zeros(...)`, only umap positions overwritten), so the
#               pad reproduces host semantics rather than approximating them.
#   duj -> Jb : `jac.scatter` is a list of (band slot, nzval position). Inverted
#               and folded through the block layout it is, per (row species, col
#               species) block entry, a length-NC vector of band slots:
#               `Jb[r,s] = duj[idx[r,s]]`. Several bands CAN land on one matrix
#               entry (the host does `+=`), so an entry is a LIST of gathers,
#               summed.
#
# Both gathers are verified to survive a REVERSE pass, including the repeated
# indices that make the adjoint an accumulate rather than a copy -- which is the
# whole point of this route. `:ad` (coloured forward JVPs) is exact too, but it
# emits an inner `enzyme.fwddiff`, and reverse-over-forward segfaults inside
# Enzyme-MLIR on this model. The symbolic Jacobian has NO nested AD in it at
# all: it is straight-line arithmetic over a gather, so a reverse pass crosses
# it the way it crosses any other expression.
#
# WHAT THIS MODULE REFUSES TO DO SILENTLY. The plan is only valid if the
# Jacobian really is block-diagonal by cell with a uniform per-cell pattern, and
# if the Jacobian's state ordering is the one the runner compiles. Neither is
# guaranteed -- `prepare_jacobian` REBUILDS the model from the raw
# FlattenedSystem while the runner compiles the esm-converted document, so two
# builds and two var maps -- and a mismatch would put every band in the wrong
# cell with no visible symptom. `block_jac_plan` therefore CHECKS all of it and
# throws; it never falls back.
module RxSymBlockJac

using Reactant
const RX = Reactant

export BlockJacPlan, block_jac_plan, gather_uj, block_jac, sym_block_jac,
       validate_plan

# `blocks[r,s]` is a list of length-NC gather vectors, already offset by +1 for
# the leading zero pad; an EMPTY list means the block entry is structurally
# zero. `ugather` is likewise padded (1 == the pad, i.e. "no state feeds this").
struct BlockJacPlan
    NS::Int
    NC::Int
    N::Int
    NJ::Int
    species::Vector{String}
    cells::Vector{String}
    ugather::Vector{Int}
    blocks::Matrix{Vector{Vector{Int}}}
    maxslots::Int
end

Base.show(io::IO, p::BlockJacPlan) = print(io,
    "BlockJacPlan(NS=$(p.NS), NC=$(p.NC), NJ=$(p.NJ), ",
    "used=$(count(!isempty, p.blocks))/$(p.NS^2), maxslots=$(p.maxslots))")

# name[i,j,k] -> ("name", "i,j,k"). The subscript is kept as an opaque STRING:
# nothing here needs to understand the grid, only that the same subscript means
# the same cell. That also keeps this working for lifts that are not 3D.
function _split_name(nm::AbstractString)
    m = match(r"^(.*)\[([^\[\]]*)\]$", nm)
    m === nothing && return nothing
    return (String(m.captures[1]), String(m.captures[2]))
end

"""
    block_jac_plan(jac; runner_names = nothing)

Build the traced gather plan from a prepared `JacobianEvaluator`. Throws with a
specific message if any precondition for a per-cell block Jacobian fails.

`runner_names` is the state-name vector the CONSUMER compiles (in its own index
order). When given, it is checked position by position against `jac.rownames`;
a mismatch is an error, because the plan indexes by position.
"""
function block_jac_plan(jac; runner_names = nothing)
    rn = jac.rownames
    n = length(rn)
    parsed = map(_split_name, rn)
    bad = findall(isnothing, parsed)
    isempty(bad) || error("block_jac_plan: $(length(bad)) state names are not " *
                          "`name[subscript]`, e.g. `$(rn[first(bad)])`; this " *
                          "Jacobian is not a species x cell lift")
    sp = first.(parsed)
    ce = last.(parsed)
    NS = length(unique(sp))
    NC = length(unique(ce))
    NS * NC == n || error("block_jac_plan: $NS species x $NC cells = $(NS*NC) " *
                          "but the Jacobian has $n states -- not a rectangular lift")

    # (a) each species one contiguous run of NC, and (b) the same cell sequence
    # inside every run. That -- and only that -- is what `blocksolve` needs: it
    # slices `x[((s-1)*NC+1):(s*NC)]` and pairs position k with position k. The
    # cells may appear in ANY order, and on ReSEACT they are not lexicographic.
    for s in 1:NS
        rng = ((s - 1) * NC + 1):(s * NC)
        length(unique(sp[rng])) == 1 ||
            error("block_jac_plan: species block $s (states $(first(rng))..$(last(rng))) " *
                  "is not a contiguous run of one species -- got " *
                  "$(length(unique(sp[rng]))) distinct names")
    end
    seq1 = ce[1:NC]
    for s in 2:NS
        ce[((s - 1) * NC + 1):(s * NC)] == seq1 ||
            error("block_jac_plan: species block $s lists its cells in a " *
                  "different order than block 1; position k of one block would " *
                  "not be the same cell as position k of another")
    end
    species = [sp[(s - 1) * NC + 1] for s in 1:NS]

    if runner_names !== nothing
        length(runner_names) == n ||
            error("block_jac_plan: consumer has $(length(runner_names)) states, " *
                  "the Jacobian has $n -- the two builds do not agree")
        k = findfirst(i -> runner_names[i] != rn[i], eachindex(rn))
        k === nothing ||
            error("block_jac_plan: state ordering differs at position $k " *
                  "(consumer `$(runner_names[k])`, Jacobian `$(rn[k])`); the " *
                  "plan indexes by position, so it would be silently wrong")
    end

    srow(i) = (i - 1) ÷ NC + 1
    crow(i) = (i - 1) % NC + 1

    # invert jac.scatter: (band slot, nzval position) -> (row, col) -> slots
    Jp = jac.prototype
    pos_of = Vector{Tuple{Int,Int}}(undef, length(Jp.nzval))
    for c in 1:size(Jp, 2), q in Jp.colptr[c]:(Jp.colptr[c + 1] - 1)
        pos_of[q] = (Jp.rowval[q], c)
    end
    slots = Dict{Tuple{Int,Int},Vector{Int}}()
    for (slot, pos) in jac.scatter
        push!(get!(slots, pos_of[pos], Int[]), slot)
    end

    offcell = count(((r, c),) -> crow(r) != crow(c), keys(slots))
    offcell == 0 ||
        error("block_jac_plan: $offcell Jacobian entries couple DIFFERENT " *
              "cells; this Jacobian is not block-diagonal by cell and cannot " *
              "be consumed as per-cell blocks")

    maxslots = isempty(slots) ? 0 : maximum(length, values(slots))
    # idx[r,s] is NC x maxslots; 0 means "this cell has no slot at this rank"
    idx = [zeros(Int, NC, maxslots) for _ in 1:NS, _ in 1:NS]
    patt = [Set{Tuple{Int,Int}}() for _ in 1:NC]
    for ((r, c), sl) in slots
        rs, cs, cc = srow(r), srow(c), crow(r)
        push!(patt[cc], (rs, cs))
        for (q, s_) in enumerate(sl)
            idx[rs, cs][cc, q] = s_
        end
    end
    ref = patt[1]
    diffs = findall(k -> patt[k] != ref, 1:NC)
    isempty(diffs) ||
        error("block_jac_plan: $(length(diffs)) of $NC cells have a different " *
              "sparsity pattern than cell 1 (e.g. cell $(first(diffs)): " *
              "$(length(patt[first(diffs)])) entries vs $(length(ref))); a " *
              "STATIC per-cell plan cannot represent that")

    blocks = [Vector{Int}[] for _ in 1:NS, _ in 1:NS]
    for rs in 1:NS, cs in 1:NS
        (rs, cs) in ref || continue
        for q in 1:maxslots
            col = idx[rs, cs][:, q]
            all(iszero, col) && continue
            # A rank present for SOME cells but not all would need a per-cell
            # zero, which the pad supplies: index 0 -> the leading 0.0.
            push!(blocks[rs, cs], col .+ 1)
        end
    end

    NJ = length(jac.uj)
    gidx = zeros(Int, NJ)
    for (i, s) in enumerate(jac.umap)
        gidx[s] = i
    end

    return BlockJacPlan(NS, NC, n, NJ, species, seq1, gidx .+ 1, blocks, maxslots)
end

# One leading 0.0, so index 1 of the padded vector is "nothing feeds this".
_pad0(x) = RX.Ops.concatenate([RX.Ops.fill(zero(eltype(x)), (1,)), x], 1)

"""
    gather_uj(plan, u)

Lift a traced state vector into the band model's own (longer) state vector.
"""
function gather_uj(plan::BlockJacPlan, u)
    up = _pad0(u)
    return up[plan.ugather]
end

"""
    block_jac(plan, duj)

Fold the band model's output into the NS x NS `Matrix{Any}` of length-NC traced
vectors that `blocksolve` consumes. Structurally-zero entries come back as
literal zero vectors, matching what the FD and AD builders produce.
"""
function block_jac(plan::BlockJacPlan, duj)
    dp = _pad0(duj)
    NS, NC = plan.NS, plan.NC
    Jb = Matrix{Any}(undef, NS, NS)
    for cs in 1:NS, rs in 1:NS
        gl = plan.blocks[rs, cs]
        if isempty(gl)
            Jb[rs, cs] = RX.Ops.fill(0.0, (NC,))
        else
            acc = dp[gl[1]]
            for q in 2:length(gl)
                add = dp[gl[q]]
                acc = acc .+ add
            end
            Jb[rs, cs] = acc
        end
    end
    return Jb
end

"""
    sym_block_jac(plan, gjb)

The `(u, p, t, bufs) -> Jb` the integrator's `symjac` hook wants, given the band
model in its 4-ARGUMENT form (`EarthSciAST.rhs_with_buffers(jac.fJ!)`). The
4-arg form is not optional: the 3-arg form resolves forcing internally, which
under tracing BAKES the meteorology in as constants.
"""
sym_block_jac(plan::BlockJacPlan, gjb) =
    (u, p, t, bufs) -> block_jac(plan, gjb(gather_uj(plan, u), p, t, bufs))

"""
    validate_plan(plan, jac, u, p, t; gjb = nothing, bufs = nothing)

Run the plan and the HOST `JacobianEvaluator` on the same point, both on the
host, and return the worst relative difference over every block entry. Zero is
the expected answer -- the plan is a permutation of the same numbers, so
anything nonzero is an index error, not roundoff.

Pass `gjb`/`bufs` to exercise the SAME 4-argument band model the traced path
uses; without them the evaluator's own stored form is called instead.
"""
function validate_plan(plan::BlockJacPlan, jac, u, p, t; gjb = nothing, bufs = nothing)
    NS, NC = plan.NS, plan.NC
    ujh = zeros(plan.NJ)
    for (i, s) in enumerate(jac.umap)
        ujh[s] = u[i]
    end
    duj = if gjb !== nothing
        gjb(ujh, p, t, bufs)
    elseif jac.oop
        jac.fJ!(ujh, p, t)
    else
        jac.fJ!(jac.duj, ujh, p, t)
        jac.duj
    end
    dp = vcat(0.0, duj)
    J = copy(jac.prototype)
    jac(J, u, p, t)
    sidx(s, c) = (s - 1) * NC + c
    worst = 0.0
    for rs in 1:NS, cs in 1:NS
        gl = plan.blocks[rs, cs]
        got = isempty(gl) ? zeros(NC) : sum(g -> dp[g], gl)
        for c in 1:NC
            ref = J[sidx(rs, c), sidx(cs, c)]
            worst = max(worst, abs(got[c] - ref) / max(abs(ref), 1e-300))
        end
    end
    return worst
end

end # module
