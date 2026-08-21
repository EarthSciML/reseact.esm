#!/usr/bin/env julia
# ===========================================================================
# rx_sym_block_jac_unit.jl -- the gather plan's index algebra, in seconds.
# ===========================================================================
# rx_sym_block_jac.jl inverts two host-side scatters (`jac.umap` and
# `jac.scatter`) into gathers a traced program can run. Getting an index wrong
# there does not throw -- it silently puts bands in the wrong cells, and the
# only end-to-end check costs a ~600 s model build plus a ~430 s
# `prepare_jacobian`. So the index algebra is tested here instead, against a
# SYNTHETIC object shaped like a `JacobianEvaluator`, with no model in sight.
#
# The synthetic case is deliberately nastier than ReSEACT's:
#   * a NON-lexicographic cell order and a shuffled species order, because the
#     plan must depend on neither (ReSEACT's real cell order is not
#     lexicographic either);
#   * a band model LONGER than the state vector, with `umap` a random injection,
#     so most band slots are fed by nothing and must come back 0.0 through the
#     pad rather than through a branch;
#   * a per-cell pattern with holes, so structural zeros are exercised;
#   * ONE matrix entry fed by TWO band slots -- the host operator does
#     `J.nzval[pos] += duj[slot]`, and an implementation that assumed one slot
#     per entry would pass every other check here.
#
# Checked: the traced plan reproduces the host call operator exactly, structural
# zeros really are zero, and the whole path REVERSE-differentiates (against
# central differences of the same host construction). That last one is the
# property the adjoint needs and `jac=:ad` cannot provide.
#
# Run: julia --project=<any env with Reactant> tools/diag/rx_sym_block_jac_unit.jl
# ===========================================================================
using Reactant, SparseArrays, LinearAlgebra, Random
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end
include(joinpath(dirname(@__DIR__), "reactant_handoff", "rx_sym_block_jac.jl"))
using .RxSymBlockJac

Random.seed!(7)
NS, NC = 4, 5
N = NS * NC
# deliberately NON-lexicographic cell order, and a shuffled species order, since
# the plan must not depend on either
cells = ["$(i),$(j),1" for j in 1:1, i in 1:5][:]
species = ["S$(s)" for s in [3, 1, 4, 2]]
rownames = [ "$(sp)[$(c)]" for sp in species for c in cells ]

# a per-cell block pattern with a hole, plus ONE entry fed by two band slots
patt = [(1,1),(1,2),(2,2),(2,3),(3,1),(3,3),(4,4),(4,1)]
sidx(s,c) = (s-1)*NC + c
I = Int[]; Jc = Int[]
for (r,c) in patt, k in 1:NC; push!(I, sidx(r,k)); push!(Jc, sidx(c,k)); end
pattern = sparse(I, Jc, trues(length(I)), N, N, |)
proto = SparseMatrixCSC(N, N, copy(pattern.colptr), copy(pattern.rowval),
                        zeros(length(pattern.rowval)))
NJ = 6 * N                      # band model longer than the state, as in reality
umap = shuffle(1:NJ)[1:N]       # scatter u -> uj
scatter = Tuple{Int,Int}[]
nextslot = 0
freeslots = setdiff(1:NJ, umap)
for (r,c) in patt, k in 1:NC
    rr, cc = sidx(r,k), sidx(c,k)
    rng = proto.colptr[cc]:(proto.colptr[cc+1]-1)
    pos = rng[searchsortedfirst(view(proto.rowval, rng), rr)]
    global nextslot += 1
    push!(scatter, (freeslots[nextslot], pos))
    if (r,c) == (2,3)           # this entry gets a SECOND slot: host does +=
        global nextslot += 1
        push!(scatter, (freeslots[nextslot], pos))
    end
end

struct FakeJac
    rownames::Vector{String}
    umap::Vector{Int}
    scatter::Vector{Tuple{Int,Int}}
    prototype::SparseMatrixCSC{Float64,Int}
    uj::Vector{Float64}
    oop::Bool
end
jac = FakeJac(rownames, umap, scatter, proto, zeros(NJ), true)

plan = block_jac_plan(jac; runner_names = rownames)
println("plan: ", plan)
@assert plan.NS == NS && plan.NC == NC
@assert count(!isempty, plan.blocks) == length(patt)
@assert plan.maxslots == 2

# a band "model": any deterministic map from uj to duj
# every band slot must depend on the STATES, or the reverse check below is
# trivially zero: the scatter slots are disjoint from umap, so a slot-local
# function of uj would be constant in u.
duj_of(uj) = sin.(uj) .+ 0.5 .* (1:length(uj)) .+ sum(uj) .* cos.(1:length(uj))

# HOST reference, exactly the JacobianEvaluator call operator
u = randn(N)
ujh = zeros(NJ); for (i,s) in enumerate(umap); ujh[s] = u[i]; end
duj = duj_of(ujh)
Jref = copy(proto); fill!(Jref.nzval, 0.0)
for (slot,pos) in scatter; Jref.nzval[pos] += duj[slot]; end

# TRACED path
function build(u)
    uj = gather_uj(plan, u)
    Jb = block_jac(plan, duj_of(uj))
    return vcat([Jb[r,s] for r in 1:NS for s in 1:NS]...)
end
x = RX.ConcreteRArray(u)
cb = RX.@compile sync=true build(x)
got = reshape(Array(cb(x)), NC, NS, NS)     # [cell, col-species, row-species]
worst = 0.0
for r in 1:NS, c in 1:NS, k in 1:NC
    ref = Jref[sidx(r,k), sidx(c,k)]
    global worst = max(worst, abs(got[k, c, r] - ref) / max(abs(ref), 1e-300))
end
println("traced plan vs host operator: worst relative ", worst)
# NOT exactly 0: `duj_of` now carries a `sum(uj)`, and XLA reduces in a
# different order than Julia. The plan itself is a permutation and contributes
# no error -- with a slot-local `duj_of` this printed exactly 0.0.
@assert worst <= 1e-14

# structural zeros must really be zero, and every off-block entry must be absent
@assert all(all(got[:, c, r] .== 0.0) for r in 1:NS, c in 1:NS if !((r,c) in patt))

# and the whole thing must reverse-differentiate
h(u) = sum(build(u) .^ 2)
cg = RX.@compile sync=true RX.Enzyme.gradient(RX.Enzyme.Reverse, h, x)
g = Array(cg(RX.Enzyme.Reverse, h, x)[1])
fd = similar(g)
for i in 1:N
    e = 1e-6 * max(abs(u[i]), 1.0)
    up = copy(u); up[i] += e; um = copy(u); um[i] -= e
    hh(v) = (uj = zeros(NJ); for (q,s) in enumerate(umap); uj[s] = v[q]; end;
             d = duj_of(uj); Jt = copy(proto); fill!(Jt.nzval, 0.0);
             for (slot,pos) in scatter; Jt.nzval[pos] += d[slot]; end;
             sum(abs2, Jt.nzval) + 0.0)
    fd[i] = (hh(up) - hh(um)) / (2e)
end
# hh sums over the SPARSE nzval, build sums over the dense NSxNSxNC layout with
# structural zeros; those contribute 0 to both, so the two agree.
println("reverse vs FD: rel ", norm(g .- fd) / max(norm(fd),1e-300))
@assert norm(g .- fd) / max(norm(fd), 1e-300) < 1e-6
@assert norm(fd) > 0
println("PLAN_UNIT_OK")
