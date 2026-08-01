#!/usr/bin/env julia
# PROTOTYPE v2: post-hoc merge of the :oop transport half's fragmented _AccKernels,
# INCLUDING template-body sub-kernels (where ~98% of the nodes live).
#
# Structure discovered by rx_count_walk.jl / v1: 4,119 top-level kernels (mostly
# 1-lane = per-cell), each carrying template-body subs; ~26 semantic classes;
# ~8M nodes total (57,648 in top-level trees, the rest inside per-kernel sub
# copies). This merges each class into ONE lane-batched kernel + lane-batched
# subs, expressing per-member differences with EarthSciAST's OWN table
# descriptors:
#   varying state slot  -> _AccStateTblBox(tbl,1,0,0,1)    (0 entry = ghost 0.0)
#   varying const/lit   -> _AccConstBox(tbl,1,0,0,1)       (frozen per-lane)
#   varying forcing idx -> _AccArrTblBox(buf,tbl,1,0,0,1)  (LIVE re-gather)
# then evaluates with stock _build_oop_acc_plan + _oop_run_acc_vec: per-lane
# semantics (fold order, ghost select, interp) are EarthSciAST's, not ours.
# Sub CSE invariant tiers move to the cell tier (same values, lane-vectorized).
# Scatter is assignment and out-slots are asserted disjoint, so kernel
# reordering is value-preserving; the host check demands bit-identity.
#
# Env: RESEACT_MODEL (default /tmp/reseact_7x7x8.esm), RESEACT_RXENV,
#      RESEACT_MERGE_TRACE=1 to also @compile the merged RHS under Reactant.
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

# ---- Build the transport half :oop ------------------------------------------
fo1 = u0 = p0 = nothing
Logging.with_logger(Logging.NullLogger()) do
    global fo1, u0, p0
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
    fo1, u0, p0, _, _ = EA.build_evaluator(doc; form=:oop,
        parameter_overrides=ov, const_arrays=mc, param_arrays=mp)
end
say("transport :oop built; nstates=$(length(u0))")

const N = EA._Node
rhs0        = fo1.rhs
rhs_list    = getfield(rhs0, :rhs_list)
cse_prelude = getfield(rhs0, :cse_prelude)
kernels     = getfield(rhs0, :acc_kernels)
plans       = getfield(rhs0, :acc_plans)
n_states    = getfield(rhs0, :n_states)
host_keys   = getfield(rhs0, :host_keys)
n_cse       = getfield(rhs0, :n_cse)
say("closure: $(length(kernels)) kernels, $(length(rhs_list)) scalar entries, n_cse=$n_cse, " *
    "max subs=$(maximum(K -> length(K.subs), kernels; init=0))")

# ---- Audit: disjoint outs (assignment scatter => order-free merge) ----------
let allouts = reduce(vcat, (pl.out_slots for pl in plans); init=Int[])
    @assert allunique(allouts) "kernel out-slots overlap — merge would reorder writes"
    say("audit: $(length(allouts)) kernel out-slots, all unique ✓")
end

# ---- Group signature (recurses into subs) -----------------------------------
mergeable_kind(k) = k in (EA._AK_STATE_AFFINE, EA._AK_STATE_TBL_BOX, EA._AK_STATE_FIXED,
                          EA._AK_SCALAR, EA._AK_CONST_AFFINE, EA._AK_CONST_BOX,
                          EA._AK_CONST_CELL, EA._AK_LOOP_IDX, EA._AK_ARR_FIXED,
                          EA._AK_FORCING_BOX, EA._AK_ARR_TBL_BOX)
function sig!(io::IOBuffer, n::N, K, parentsubs, why::Base.RefValue{Symbol})
    k = n.kind
    if k === EA._NK_ACCESS
        a = K.acc[n.idx]
        mergeable_kind(a.kind) || (why[] = :acc_kind)
        # kind-FAMILY, not exact kind: the clone tables all members of a family
        # identically (per-member a.kind branches), so interior AFFINE and
        # boundary TBL_BOX cells may share a class.
        if a.kind in (EA._AK_STATE_AFFINE, EA._AK_STATE_TBL_BOX, EA._AK_STATE_FIXED)
            print(io, "AS")
        elseif a.kind in (EA._AK_ARR_FIXED, EA._AK_FORCING_BOX, EA._AK_ARR_TBL_BOX)
            print(io, "AF", "b", objectid(a.arr))
        else
            print(io, "AC")
        end
    elseif k === EA._NK_LITERAL
        print(io, "L")                     # value free (tabled if varying)
    elseif k === EA._NK_PARAM
        print(io, "P", n.sym)
    elseif k === EA._NK_TIME
        print(io, "T")
    elseif k === EA._NK_CACHED
        print(io, "C", n.payload === K.cse.scratch ? "c" : "i", n.idx)
    elseif k === EA._NK_CONTRACTION
        print(io, "K", n.op, "s", n.literal, "(")
        for c in n.children; sig!(io, c, K, parentsubs, why); end
        print(io, ")")
    elseif k === EA._NK_SUBCALL
        pos = findfirst(s -> s === n.payload, parentsubs)
        pos === nothing && (why[] = :subcall_unknown; return)
        print(io, "S", pos)
    elseif k === EA._NK_REDUCE
        why[] = :reduce; print(io, "X")
    else  # _NK_OP
        print(io, "O", n.op)
        pl = n.payload
        if pl isa Tuple && length(pl) >= 1
            print(io, "@", pl[1])
            length(pl) >= 2 && print(io, "#", try; EA._fn_spec_hash(pl[2]); catch; hash(repr(typeof(pl[2]))); end)
        end
        print(io, "(")
        for c in n.children; sig!(io, c, K, parentsubs, why); end
        print(io, ")")
    end
end
function trees_sig!(io::IOBuffer, K, parentsubs, why)
    print(io, "z", K.zerobar, "|")
    sig!(io, K.spine, K, parentsubs, why); print(io, "|I")
    for r in K.cse.inv_recipes; sig!(io, r, K, parentsubs, why); print(io, ";"); end
    print(io, "|R")
    for r in K.cse.recipes; sig!(io, r, K, parentsubs, why); print(io, ";"); end
end
function kernel_sig(K, plan)
    why = Ref(:ok)
    plan.vectorizable || (why[] = :unvectorizable)
    isempty(plan.red_seg) || (why[] = :reduce)
    io = IOBuffer()
    trees_sig!(io, K, K.subs, why)
    for (si, S) in enumerate(K.subs)
        isempty(S.subs) || (why[] = :nested_sub_subs)
        print(io, "|SUB", si, ":")
        trees_sig!(io, S, K.subs, why)
    end
    return why[] === :ok ? String(take!(io)) : nothing, why[]
end

groups = Dict{String,Vector{Int}}()
passthrough = Int[]
whycount = Dict{Symbol,Int}()
for j in eachindex(kernels)
    s, why = kernel_sig(kernels[j], plans[j])
    whycount[why] = get(whycount, why, 0) + 1
    s === nothing ? push!(passthrough, j) : push!(get!(groups, s, Int[]), j)
end
say("sig outcome: $whycount")
say("groups: $(length(groups)) classes " *
    "(sizes: $(sort(length.(values(groups)); rev=true)[1:min(end,10)]) ...); " *
    "blocked kernels: $(length(passthrough))")

# ---- Merge one group ---------------------------------------------------------
lanes_of(j)::Int = length(plans[j].out_slots)
function merge_group(js::Vector{Int})
    m = length(js); Ls = Int[lanes_of(j) for j in js]; L = sum(Ls)
    rep = kernels[js[1]]
    nsubs = length(rep.subs)
    @assert all(j -> length(kernels[j].subs) == nsubs, js)
    merged_subs = Vector{EA._AccKernel}(undef, nsubs)

    # Merge one aligned tree-family (parent trees or sub-si trees).
    # Kof(i)/Pof(i): member i's context kernel and the plan whose tables resolve
    # that context's descriptors (sub plans are built against PARENT lanes, so
    # every table below has member-lane length by construction).
    function merge_trees(Kof, Pof)
        accvec = EA._AccDesc[]
        n_inv = length(Kof(1).cse.inv_recipes); n_cell = length(Kof(1).cse.recipes)
        @assert all(i -> length(Kof(i).cse.inv_recipes) == n_inv &&
                         length(Kof(i).cse.recipes) == n_cell, 1:m)
        newscr = EA._AccScratch(n_inv + n_cell)
        function clone(nodes::Vector{N})::N
            r = nodes[1]; k = r.kind
            if k === EA._NK_LITERAL
                vals = Float64[n.literal for n in nodes]
                if all(==(vals[1]), vals)
                    return N(k, r.op, r.literal, r.idx, r.sym, nothing, N[])
                end
                tbl = Vector{Float64}(undef, L); q = 0
                for i in 1:m
                    tbl[q+1:q+Ls[i]] .= vals[i]; q += Ls[i]
                end
                push!(accvec, EA._AccConstBox(tbl, 1, 0, 0, 1))
                return N(EA._NK_ACCESS, :acc, 0.0, length(accvec), Symbol(""), nothing, N[])
            elseif k === EA._NK_PARAM || k === EA._NK_TIME
                return N(k, r.op, r.literal, r.idx, r.sym, r.payload, N[])
            elseif k === EA._NK_CACHED
                tier_cell = r.payload === Kof(1).cse.scratch
                @assert all(i -> (nodes[i].payload === Kof(i).cse.scratch) == tier_cell, 1:m)
                nidx = tier_cell ? n_inv + r.idx : r.idx
                return N(k, r.op, r.literal, nidx, r.sym, newscr, N[])
            elseif k === EA._NK_SUBCALL
                pos = findfirst(s -> s === r.payload, kernels[js[1]].subs)
                @assert pos !== nothing
                @assert all(i -> kernels[js[i]].subs[pos] === nodes[i].payload, 1:m)
                @assert isassigned(merged_subs, pos) "nested-first order violated"
                return N(k, r.op, r.literal, r.idx, r.sym, merged_subs[pos], N[])
            elseif k === EA._NK_ACCESS
                akind = Kof(1).acc[r.idx].kind
                if akind in (EA._AK_STATE_AFFINE, EA._AK_STATE_TBL_BOX, EA._AK_STATE_FIXED)
                    tbl = Vector{Int}(undef, L); q = 0
                    for i in 1:m
                        lk = Ls[i]; idx = nodes[i].idx
                        a = Kof(i).acc[idx]; pl = Pof(i)
                        if a.kind === EA._AK_STATE_FIXED
                            tbl[q+1:q+lk] .= a.idx
                        elseif a.kind === EA._AK_STATE_AFFINE
                            tbl[q+1:q+lk] .= pl.gathers[idx]
                        else # TBL_BOX: reconstruct raw (0 = ghost)
                            g = pl.gathers[idx]; gh = pl.ghost[idx]
                            if isempty(gh); tbl[q+1:q+lk] .= g
                            else; for mm in 1:lk; tbl[q+mm] = gh[mm] ? 0 : g[mm]; end; end
                        end
                        q += lk
                    end
                    push!(accvec, EA._AccStateTblBox(tbl, 1, 0, 0, 1))
                elseif akind in (EA._AK_SCALAR, EA._AK_CONST_AFFINE, EA._AK_CONST_BOX,
                                 EA._AK_CONST_CELL, EA._AK_LOOP_IDX)
                    tbl = Vector{Float64}(undef, L); q = 0
                    for i in 1:m
                        lk = Ls[i]; idx = nodes[i].idx
                        a = Kof(i).acc[idx]; pl = Pof(i)
                        if a.kind === EA._AK_SCALAR
                            tbl[q+1:q+lk] .= a.v
                        else
                            tbl[q+1:q+lk] .= pl.consts[idx]
                        end
                        q += lk
                    end
                    push!(accvec, EA._AccConstBox(tbl, 1, 0, 0, 1))
                else # ARR_FIXED / FORCING_BOX / ARR_TBL_BOX (live; same buffer by sig)
                    arr = Kof(1).acc[r.idx].arr
                    tbl = Vector{Int}(undef, L); q = 0
                    for i in 1:m
                        lk = Ls[i]; idx = nodes[i].idx
                        a = Kof(i).acc[idx]; pl = Pof(i)
                        if a.kind === EA._AK_ARR_FIXED
                            tbl[q+1:q+lk] .= a.idx
                        else
                            tbl[q+1:q+lk] .= pl.forc[idx]
                        end
                        q += lk
                    end
                    push!(accvec, EA._AccArrTblBox(arr, tbl, 1, 0, 0, 1))
                end
                return N(EA._NK_ACCESS, :acc, 0.0, length(accvec), Symbol(""), nothing, N[])
            else # _NK_OP / _NK_CONTRACTION
                ch = Vector{N}(undef, length(r.children))
                for ci in eachindex(r.children)
                    ch[ci] = clone(N[n.children[ci] for n in nodes])
                end
                return N(k, r.op, r.literal, r.idx, r.sym, r.payload, ch)
            end
        end
        spine = clone(N[Kof(i).spine for i in 1:m])
        recipes = Vector{N}(undef, n_inv + n_cell)
        for i2 in 1:n_inv
            recipes[i2] = clone(N[Kof(i).cse.inv_recipes[i2] for i in 1:m])
        end
        for i2 in 1:n_cell
            recipes[n_inv + i2] = clone(N[Kof(i).cse.recipes[i2] for i in 1:m])
        end
        return spine, recipes, accvec, newscr
    end

    for si in 1:nsubs
        # NOTE: names must not collide with clone's locals — a nested function
        # assigning a name bound in this scope REBINDS it (shared box), which
        # silently corrupted every cloned node until renamed (msp_/mrc_/...).
        msp_, mrc_, mav_, msc_ = merge_trees(i -> kernels[js[i]].subs[si],
                                             i -> plans[js[i]].sub_plans[si])
        repsub = rep.subs[si]
        merged_subs[si] = EA._AccKernel(repsub.cells, msp_, mav_, repsub.bound, repsub.zerobar,
                                        EA._AccCSE(mrc_, msc_, N[], EA._AccScratch(0)),
                                        EA._AccKernel[])
    end
    msp_, mrc_, mav_, msc_ = merge_trees(i -> kernels[js[i]], i -> plans[js[i]])
    outs = reduce(vcat, (plans[j].out_slots for j in js))
    return EA._AccKernel(EA._outs_cells(outs), msp_, mav_, rep.bound, rep.zerobar,
                         EA._AccCSE(mrc_, msc_, N[], EA._AccScratch(0)), merged_subs)
end

merged = EA._AccKernel[]
group_list = Vector{Int}[]
for js in values(groups)
    push!(merged, length(js) == 1 ? kernels[js[1]] : merge_group(js))
    push!(group_list, js)
end
# debug: compare rep vs merged tree ops for the first multi-member class
let gi = findfirst(js -> length(js) > 1, group_list)
    js = group_list[gi]; rep = kernels[js[1]]; mk = merged[gi]
    ops(n, acc) = (n.kind === EA._NK_OP && push!(acc, n.op); foreach(c -> ops(c, acc), n.children); acc)
    say("DEBUG class $gi ($(length(js)) members):")
    say("  rep spine root: kind=$(Int(rep.spine.kind)) op=$(repr(rep.spine.op)) nch=$(length(rep.spine.children))")
    say("  mrg spine root: kind=$(Int(mk.spine.kind)) op=$(repr(mk.spine.op)) nch=$(length(mk.spine.children))")
    say("  rep spine ops: $(join(unique(ops(rep.spine, Symbol[])), ','))")
    say("  mrg spine ops: $(join(unique(ops(mk.spine, Symbol[])), ','))")
    say("  rep #subs=$(length(rep.subs)) sub1 spine root op=$(isempty(rep.subs) ? "-" : repr(rep.subs[1].spine.op))")
    say("  mrg #subs=$(length(mk.subs)) sub1 spine root op=$(isempty(mk.subs) ? "-" : repr(mk.subs[1].spine.op))")
    if !isempty(rep.subs)
        say("  rep sub1 ops: $(join(unique(ops(rep.subs[1].spine, Symbol[])), ','))")
        say("  mrg sub1 ops: $(join(unique(ops(mk.subs[1].spine, Symbol[])), ','))")
        say("  rep sub1 recipes: $(length(rep.subs[1].cse.recipes)) inv=$(length(rep.subs[1].cse.inv_recipes))")
        r1ops = Symbol[]; foreach(r -> ops(r, r1ops), rep.subs[1].cse.recipes)
        m1ops = Symbol[]; foreach(r -> ops(r, m1ops), mk.subs[1].cse.recipes)
        say("  rep sub1 recipe ops: $(join(unique(r1ops), ','))")
        say("  mrg sub1 recipe ops: $(join(unique(m1ops), ','))")
    end
end
append!(merged, kernels[passthrough])
mplans = EA._OopAccPlan[EA._build_oop_acc_plan(K) for K in merged]
@assert all(pl -> pl.vectorizable, mplans) "a merged kernel failed the vec-plan gate"

function countnodes(ks)
    seen = IdDict{N,Bool}()
    function walk(n); haskey(seen, n) && return; seen[n] = true; foreach(walk, n.children); nothing; end
    todo = EA._AccKernel[]
    append!(todo, ks)
    for K in ks; append!(todo, K.subs); end
    for K in todo
        walk(K.spine); foreach(walk, K.cse.recipes); foreach(walk, K.cse.inv_recipes)
    end
    return length(seen)
end
say("MERGE: $(length(kernels)) kernels -> $(length(merged)) " *
    "($(count(js -> length(js) > 1, values(groups))) real merges, " *
    "$(count(js -> length(js) == 1, values(groups))) singletons, $(length(passthrough)) blocked)")
say("MERGE: unique nodes (incl subs) $(countnodes(kernels)) -> $(countnodes(merged))")


# ---- Static sanity check: every merged node must be acck-evaluable ----------
const ACCK_KINDS = (EA._NK_ACCESS, EA._NK_LITERAL, EA._NK_PARAM, EA._NK_TIME,
                    EA._NK_CACHED, EA._NK_SUBCALL, EA._NK_REDUCE, EA._NK_CONTRACTION,
                    EA._NK_OP)
function check_trees(ks, label)
    bad = Dict{Tuple{Int,Symbol},Int}()
    hist = Dict{Int,Int}()
    seen = IdDict{N,Bool}()
    printed_path = Ref(false)
    path = Ref("")
    function walk(n, evaluated::Bool)
        haskey(seen, n) && return; seen[n] = true
        opath = path[]; path[] = opath * "/" * string(Int(n.kind)) * (n.kind === EA._NK_OP ? string(n.op) : "")
        hist[Int(n.kind)] = get(hist, Int(n.kind), 0) + 1
        if evaluated && !(n.kind in ACCK_KINDS)
            bad[(Int(n.kind), n.op)] = get(bad, (Int(n.kind), n.op), 0) + 1
        end
        if n.kind === EA._NK_OP && n.op === Symbol("")
            key = (5, evaluated ? :EVAL_emptyop : :lazy_emptyop)
            bad[key] = get(bad, key, 0) + 1
            if evaluated && !printed_path[]
                printed_path[] = true
                say("  path to EVALUATED empty-op node: $(path[])")
            end
        end
        if n.kind === EA._NK_OP && n.op === :fn
            pl = n.payload
            lazy = pl isa Tuple && length(pl) >= 2 && !(pl[2] === nothing)
            for (ci, c) in enumerate(n.children)
                walk(c, evaluated && (!lazy || ci <= 2))
            end
        elseif n.kind === EA._NK_OP && (n.op === :^ || n.op === :pow) && length(n.children) == 2
            walk(n.children[1], evaluated)
            walk(n.children[2], evaluated)   # literal-exponent arm skips eval but keep strict
        else
            for c in n.children; walk(c, evaluated); end
        end
        path[] = opath
    end
    todo = EA._AccKernel[]; append!(todo, ks)
    for K in ks; append!(todo, K.subs); end
    for K in todo
        walk(K.spine, true); foreach(r -> walk(r, true), K.cse.recipes)
        foreach(r -> walk(r, true), K.cse.inv_recipes)
    end
    say("check[$label]: kind histogram $(sort(collect(hist)))")
    isempty(bad) || say("check[$label]: NON-ACCK EVALUATED NODES: $bad")
    return bad
end
check_trees(kernels, "original")
check_trees(merged, "merged")

# ---- The merged RHS (mirror of _make_rhs_oop's closure body) ----------------
function merged_rhs(u, p, t, buffers)
    fb = EA._OopForcing(buffers, host_keys)
    T = EA._oop_value_type(u, p, t)
    cache = Vector{T}(undef, n_cse)
    @inbounds for s in 1:n_cse
        cache[s] = EA._oop_eval(cse_prelude[s], u, p, t, cache, fb)
    end
    du = EA._oop_du_zeros(u, T, n_states)
    @inbounds for k in eachindex(rhs_list)
        slot, node = rhs_list[k]
        du = EA._oop_store(du, slot, EA._oop_eval(node, u, p, t, cache, fb))
    end
    for j in eachindex(merged)
        du = EA._oop_run_acc_vec(du, u, p, t, merged[j], mplans[j], T, fb)
    end
    return du
end

# ---- Per-kernel isolation: find exactly which merged kernel fails -----------
let bufs = EA.forcing_buffers(fo1)
    fb = EA._OopForcing(bufs, host_keys)
    du = zeros(n_states)
    for j in eachindex(merged)
        try
            EA._oop_run_acc_vec(du, u0, p0, T0, merged[j], mplans[j], Float64, fb)
        catch e
            say("FAIL at merged kernel $j: $(sprint(showerror, e))")
            K = merged[j]
            ops = Dict{Symbol,Int}()
            seenX = IdDict{N,Bool}()
            function wops(n)
                haskey(seenX, n) && return; seenX[n] = true
                n.kind === EA._NK_OP && (ops[n.op] = get(ops, n.op, 0) + 1)
                foreach(wops, n.children)
            end
            wops(K.spine); foreach(wops, K.cse.recipes)
            for S in K.subs; wops(S.spine); foreach(wops, S.cse.recipes); end
            say("  ops in failing kernel: $(sort(collect(ops); by=last, rev=true))")
            break
        end
    end
end

# ---- Host bit-identity check ------------------------------------------------
du_ref = fo1(u0, p0, T0)
du_m   = merged_rhs(u0, p0, T0, EA.forcing_buffers(fo1))
d = maximum(abs, du_m .- du_ref)
nexact = count(du_m .== du_ref)
say("VERIFY host: maxabs(merged-ref)=$d exact-equal=$nexact/$(length(du_ref)) " *
    (d == 0.0 ? "BIT-IDENTICAL ✓" : "NOT bit-identical ✗"))
t1 = @elapsed for _ in 1:5; fo1(u0, p0, T0); end
t2 = @elapsed for _ in 1:5; merged_rhs(u0, p0, T0, EA.forcing_buffers(fo1)); end
say("host RHS time: original=$(round(t1/5*1000, digits=1))ms merged=$(round(t2/5*1000, digits=1))ms")

# ---- Optional: trace the merged RHS under Reactant --------------------------
if get(ENV, "RESEACT_MERGE_TRACE", "0") == "1"
    include(joinpath(HERE, "rx_merge_trace_inc.jl"))
end
say("MERGE DONE")
