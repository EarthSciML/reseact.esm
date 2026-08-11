# Reusable library form of the kernel-merge prototype (rx_merge_kernels.jl).
# Post-hoc lane-batching merge of a form=:oop evaluator's fragmented _AccKernels,
# including template-body sub-kernels. See rx_merge_kernels.jl for the annotated
# prototype and RESULTS.md §2026-07-21 for measurements (4,119→346 kernels,
# 7.93M→1.06M nodes, host bit-identical, 9.5× faster host RHS, transport half
# then @compiles under Reactant in ~393 s).
#
# Usage:
#   include("rx_merge_lib.jl")
#   mg = RxOopMerge.merge_oop_rhs(fo.rhs)      # fo from build_evaluator(form=:oop)
#   mg.rhs(u, p, t, buffers)                   # same contract as rhs_with_buffers(fo)
#   mg.stats                                   # NamedTuple of merge statistics
module RxOopMerge

import EarthSciAST
const EA = EarthSciAST
const N = EA._Node

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
        # identically, so interior AFFINE and boundary TBL_BOX cells share a class.
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

# Merge one class of lockstep-identical kernels into a single lane-batched kernel.
# js indexes into kernels/plans; per-member varying leaves become table descriptors.
function merge_group(kernels, plans, js::Vector{Int})
    m = length(js); Ls = Int[length(plans[j].out_slots) for j in js]; L = sum(Ls)
    rep = kernels[js[1]]
    nsubs = length(rep.subs)
    @assert all(j -> length(kernels[j].subs) == nsubs, js)
    merged_subs = Vector{EA._AccKernel}(undef, nsubs)

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
        # assigning a name bound in this scope REBINDS it (shared box).
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

"""
    merge_oop_rhs(rhs0; count_nodes=false) -> (; rhs, merged, mplans, stats)

`rhs0` is the `_OopRHS` closure of a `form=:oop` evaluator (`fo.rhs`).
Groups its acc kernels by lockstep signature, merges each class into one
lane-batched kernel, and returns a drop-in `rhs(u, p, t, buffers)` with the
same contract as `EarthSciAST.rhs_with_buffers(fo)`. Merge is value-exact
(host output bit-identical to the original; assignment scatter + disjoint
out-slots make kernel reordering safe — asserted here).
"""
function merge_oop_rhs(rhs0; count_nodes::Bool=false)
    rhs_list    = getfield(rhs0, :rhs_list)
    cse_prelude = getfield(rhs0, :cse_prelude)
    kernels     = getfield(rhs0, :acc_kernels)
    plans       = getfield(rhs0, :acc_plans)
    n_states    = getfield(rhs0, :n_states)
    host_keys   = getfield(rhs0, :host_keys)
    n_cse       = getfield(rhs0, :n_cse)

    # MATERIALIZED ARRAY OBSERVEDS (EarthSciAST 66b8e9a6, `mat_levels`): the
    # reconstruction below predates that closure layout — it rebuilds the RHS
    # from rhs_list/cse_prelude/kernels only, so it would never run the level
    # fills and every reader of an observed slot would see undef (NaN). The
    # bit-identity gate downstream then rejects the result with maxabs=NaN,
    # which LOOKS like a merge defect but is this staleness. The in-package
    # kernel-class merge already ran at build (both emitters receive merged
    # kernels), so there is nothing left for this pass to do — say so instead
    # of reconstructing an RHS that silently skips part of the model.
    if hasfield(typeof(rhs0), :mat_levels) && !isempty(getfield(rhs0, :mat_levels))
        error("merge_oop_rhs: rhs carries materialized array-observed levels " *
              "(mat_levels), which this driver-side reconstruction predates; " *
              "the in-package merge already ran at build — use the stock rhs")
    end

    allouts = reduce(vcat, (pl.out_slots for pl in plans); init=Int[])
    @assert allunique(allouts) "kernel out-slots overlap — merge would reorder writes"

    groups = Dict{String,Vector{Int}}()
    passthrough = Int[]
    whycount = Dict{Symbol,Int}()
    for j in eachindex(kernels)
        s, why = kernel_sig(kernels[j], plans[j])
        whycount[why] = get(whycount, why, 0) + 1
        s === nothing ? push!(passthrough, j) : push!(get!(groups, s, Int[]), j)
    end

    merged = EA._AccKernel[]
    for js in values(groups)
        push!(merged, length(js) == 1 ? kernels[js[1]] : merge_group(kernels, plans, js))
    end
    append!(merged, kernels[passthrough])
    mplans = EA._OopAccPlan[EA._build_oop_acc_plan(K) for K in merged]
    @assert all(pl -> pl.vectorizable, mplans) "a merged kernel failed the vec-plan gate"

    stats = (; n_kernels = length(kernels), n_merged = length(merged),
             n_classes = length(groups), n_blocked = length(passthrough),
             whycount,
             nodes_before = count_nodes ? countnodes(kernels) : -1,
             nodes_after  = count_nodes ? countnodes(merged) : -1)

    function rhs(u, p, t, buffers)
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

    return (; rhs, merged, mplans, stats)
end

end # module
