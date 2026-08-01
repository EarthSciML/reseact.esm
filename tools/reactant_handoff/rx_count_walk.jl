#!/usr/bin/env julia
# PROBE (no tracing): why does the :oop transport RHS emit millions of broadcasts?
# Builds the transport half, then measures the compiled IR itself:
#   unique   — distinct _Node objects reachable (the DAG the build produced)
#   expanded — node evaluations a memo-less tree walk performs (what the :oop
#              vec walker actually does per RHS call)
#   structural — distinct nodes up to structure (what PERFECT interning would give)
# expanded >> unique  => the walker re-expands a well-shared DAG (walker-level fix)
# unique >> structural => the build manufactures copies interning missed (build fix)
import Pkg
const HERE  = @__DIR__
const REPO  = normpath(joinpath(HERE, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io=devnull)
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system, stencil_vs_pointwise
using Logging
const EA = EarthSciAST
const MODEL = get(ENV, "RESEACT_MODEL", "/tmp/reseact_7x7x8.esm")
say(s) = (println(s); flush(stdout))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
const T0 = 64800.0

fo1 = nothing
Logging.with_logger(Logging.NullLogger()) do
    global fo1
    file = EA.load(MODEL); flat = EA.flatten(file)
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
    # stencil_following_rule (split_common.jl), not the shipped
    # stencil_vs_pointwise: the air-mass equation reads `divh_fix`, an OBSERVED
    # that is a stencil, and the syntactic rule would post that term to the
    # chemistry half.
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    doc   = EA.flattened_to_esm(parts[1])                          # transport only
    ff = reseact_forcing(CHEMDIR)
    mc = Dict{String,Any}(String(k)=>v for (k,v) in ff.const_arrays)
    mp = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        (EA.provider_is_const(prov) ? mc : mp)[k] = fld
    end
    ov = Dict{String,Float64}(String(k)=>Float64(v) for (k,v) in ff.parameters)
    fo1, _, _, _, _ = EA.build_evaluator(doc; form=:oop,
        parameter_overrides=ov, const_arrays=mc, param_arrays=mp)
end
say("transport :oop built")

# ---- Reach into the RHS closure for the compiled IR --------------------------
const N = EA._Node
rhs = fo1.rhs
say("closure fields: $(collect(fieldnames(typeof(rhs))))")
roots = N[]                      # every node the per-call walk starts from
kernels = EA._AccKernel[]
function harvest!(x, seen=IdDict{Any,Bool}())
    haskey(seen, x) && return; seen[x] = true
    if x isa N
        push!(roots, x)
    elseif x isa EA._AccKernel
        push!(kernels, x)
        harvest!(x.spine, seen)
        for f in fieldnames(typeof(x.cse)); harvest!(getfield(x.cse, f), seen); end
        for s in x.subs; harvest!(s, seen); end
    elseif x isa Vector
        isempty(x) && return
        (x[1] isa N || x[1] isa EA._AccKernel || x[1] isa Tuple) || return
        for e in x; harvest!(e, seen); end
    elseif x isa Tuple
        for e in x; e isa N ? harvest!(e, seen) : (e isa EA._AccKernel && harvest!(e, seen)); end
    end
end
hs = IdDict{Any,Bool}()
for f in fieldnames(typeof(rhs)); harvest!(getfield(rhs, f), hs); end
say("harvested: $(length(roots)) root nodes, $(length(kernels)) acc kernels " *
    "(subs per kernel: $(isempty(kernels) ? 0 : maximum(k -> length(k.subs), kernels)) max)")

# ---- unique vs expanded vs structural ---------------------------------------
const EXPANDED = IdDict{N,Float64}()
const NK_CACHED = EA._NK_CACHED
function expanded(n::N)::Float64
    v = get(EXPANDED, n, -1.0); v >= 0 && return v
    s = 1.0
    if n.kind !== NK_CACHED       # cached slot = O(1) read; recipe counted once via roots
        for c in n.children; s += expanded(c); end
    end
    EXPANDED[n] = s; return s
end
total_expanded = sum(expanded, roots)
unique_nodes = length(EXPANDED)

# structural identity: hash on (kind, op, literal, idx, sym, payload-id, child-keys)
const SKEY = IdDict{N,Int}()
const STAB = Dict{Any,Int}()
function skey(n::N)::Int
    v = get(SKEY, n, 0); v != 0 && return v
    k = (n.kind, n.op, n.literal, n.idx, n.sym,
         n.payload === nothing ? 0 : objectid(n.payload),
         Tuple(skey(c) for c in n.children))
    id = get!(STAB, k, length(STAB) + 1)
    SKEY[n] = id; return id
end
foreach(skey, roots)
say("")
say("RESULT nodes: unique=$(unique_nodes) structural=$(length(STAB)) " *
    "expanded=$(round(total_expanded, sigdigits=4))")
say("RESULT ratios: walker re-expansion (expanded/unique) = " *
    "$(round(total_expanded / max(unique_nodes,1), digits=1))x ; " *
    "identity-sharing miss (unique/structural) = " *
    "$(round(unique_nodes / max(length(STAB),1), digits=2))x")
# per-kernel spine detail, largest first
ks = sort(kernels; by=k -> -get(EXPANDED, k.spine, 0.0))
for k in ks[1:min(end, 6)]
    say("  kernel: spine expanded=$(round(get(EXPANDED,k.spine,0.0), sigdigits=4)) " *
        "cse_recipes=$(sum(f -> getfield(k.cse, f) isa Vector ? length(getfield(k.cse, f)) : 0,
                           fieldnames(typeof(k.cse)))) subs=$(length(k.subs))")
end
say("PROBE DONE")

# ---- Part 2: why 18k kernels? group by semantic shape + lane counts ---------
shapekey(n::N) = hash((n.kind, n.op, length(n.children),
                       Tuple(shapekey(c) for c in n.children)))
nlanes(k) = try; length(k.cells); catch; -1; end
byshape = Dict{UInt,Vector{Int}}()   # shape -> lane counts of kernels in class
for k in kernels
    push!(get!(byshape, shapekey(k.spine), Int[]), nlanes(k))
end
say("kernel semantic classes (spine shape only): $(length(byshape))")
say("lanes-per-kernel histogram:")
lh = Dict{Int,Int}()
for k in kernels; lh[nlanes(k)] = get(lh, nlanes(k), 0) + 1; end
for (l, c) in sort(collect(lh))
    say("  $l lanes: $c kernels")
end
say("largest classes (kernels × lanes):")
for (kk, v) in first(sort(collect(byshape); by=p -> -length(p.second)), 8)
    say("  class: $(length(v)) kernels, lanes=$(sort(unique(v))')")
end
tot_lanes = sum(nlanes, kernels)
say("total lanes across kernels=$tot_lanes (nstates=5096); cells fieldnames=$(fieldnames(typeof(kernels[1].cells)))")
say("PROBE2 DONE")

# ---- Part 3: lanes (fixed) + merge-compatibility signature ------------------
function nlanes2(k)
    c = k.cells
    !isempty(c.outs) && return length(c.outs)
    try; return prod(length, c.ranges); catch; return -1; end
end
# merge signature: shape + access-desc KINDS + literal pattern (values/indices free)
function msig(n::N, K, io::IOBuffer)
    print(io, n.kind, ':', n.op, '(')
    if n.kind === EA._NK_ACCESS
        print(io, "ak", K.acc[n.idx].kind)
    end
    for c in n.children; msig(c, K, io); end
    print(io, ')')
end
msigs = Dict{String,Int}(); lanetot = 0
lh2 = Dict{Int,Int}()
for k in kernels
    io = IOBuffer(); msig(k.spine, k, io)
    s = String(take!(io)); msigs[s] = get(msigs, s, 0) + 1
    l = nlanes2(k); global lanetot += l; lh2[l] = get(lh2, l, 0) + 1
end
say("merge-compatible signature classes: $(length(msigs)) " *
    "(top: $(first(sort(collect(values(msigs)); rev=true), 6)))")
say("lanes histogram: $(sort(collect(lh2)))")
say("total lane-evals=$lanetot vs nstates=5096")
say("PROBE3 DONE")
