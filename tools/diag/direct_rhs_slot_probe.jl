#!/usr/bin/env julia
# ===========================================================================
# direct_rhs_slot_probe.jl -- WHICH SLOTS DOES THE DIRECT EMITTER READ BEFORE
# ANYTHING WRITES THEM, AND IS ANYTHING GOING TO WRITE THEM AT ALL?
# ===========================================================================
# The direct StableHLO emitter (`EarthSciASTReactantExt.direct_rhs`) keeps a
# SLOT MAP instead of a flat buffer: every extended-state slot is either an
# already-emitted value or `nothing`. A read of a `nothing` slot used to be an
# unconditional refusal, on the argument that the ordering must come from the
# fill levels. That is only half true. The interpreter's extended vector starts
# at ZERO, so a slot NO level ever writes legitimately reads 0.0 -- a structural
# zero, not an ordering bug. A slot some LATER level writes is the ordering bug.
#
# This probe separates the two WITHOUT compiling anything: it builds the two
# `:oop` halves, reflects on the closure's `mat_levels` / `acc_plans` /
# `scan_folds`, and replays the emitter's walk order over a Bool "written"
# vector, mirroring `ext/reactant_direct/emit.jl` unit for unit. It reports
# every read-before-write, and for each one whether the slot is in the
# GLOBAL write set of the emission (ordering bug) or not (structural zero).
#
# No Reactant, no MLIR, no XLA -- the build is the only cost.
#
# Env:
#   RESEACT_NLON/NLAT/NLEV     grid (default 6/6/8)
#   RESEACT_RES                GEOS-FP grid row (default 4x5)
#   RESEACT_T0                 seconds from the epoch (default 5400)
#   ESS_OOP_SSA                read at build time (default 1)
#   RESEACT_RXENV              the Julia environment to activate
#   RESEACT_SLOT_NAMES         optional JSON file mapping extended slot ->
#                              name, as dumped by a temporary build patch.
#                              Without it mat-block slots print as numbers.
# ===========================================================================
import Pkg
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, JSON3
using EarthSciAST, EarthSciIO
using EarthSciASTSplitter: split_system
const EA = EarthSciAST

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "ESS_OOP_SSA", "1")
const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
_envi(k) = haskey(ENV, "RESEACT_$k") ? parse(Int, ENV["RESEACT_$k"]) : nothing
const RES   = get(ENV, "RESEACT_RES", "4x5")
const SLICE = native_slice(res = RES, lon0 = _envi("LON0"), lat0 = _envi("LAT0"),
                           nlon = _envi("NLON"), nlat = _envi("NLAT"),
                           nlev = _envi("NLEV"))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const NDAYS = forcing_days_for(T0, T0 + 16 * 3600)

say("="^78)
say(@sprintf("DIRECT-RHS SLOT PROBE  grid=%dx%dx%d  res=%s  ESS_OOP_SSA=%s",
             GRID_MP["NLON"], GRID_MP["NLAT"], NLEV_EFF, RES, ENV["ESS_OOP_SSA"]))
say("="^78)

# --------------------------------------------------------------------------- #
# 1. Build -- the census probe's build block, minus the band model.
# --------------------------------------------------------------------------- #
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = docs = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing; splitparts = nothing; merged_const = nothing; ov = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff
    global splitparts, merged_const, ov, docs
    file = EA.load_path(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in splitparts]
    f0 = reseact_forcing(CHEMDIR; ndays = NDAYS, res = SLICE.res)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld; discrete[k] = prov
        end
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part 2 var_map != part 1")
        end
    end
end
foreach(d -> d.materialize!(), dms)
say(@sprintf("BUILD (two :oop halves) %.2f s   nstates=%d  nparams=%d",
             time() - tb, length(u0), length(p)))

# --------------------------------------------------------------------------- #
# 1b. IS THE SPLIT A SPLIT?
# --------------------------------------------------------------------------- #
# EarthSciASTSplitter's lifted-derivative recogniser matched the pre-1.1.0
# `aggregate` node tag. EarthSciAST normalizes that alias away on load, so after
# the rename the recogniser returned `nothing` for every state equation and
# `split_equations` copied the WHOLE tendency into BOTH parts. Nothing downstream
# noticed: two identical halves build, compile and run. Every measurement taken
# on them is a measurement of the unsplit model, so this check comes first and
# the probe says outright when the split is degenerate.
function split_sanity(docs)
    eqs(d) = begin
        ms = d["models"]
        m = ms[first(keys(ms))]
        get(m, "equations", Any[])
    end
    say("\n  SPLIT SANITY")
    nbytes = Int[]
    for (i, d) in enumerate(docs)
        E = eqs(d)
        tot = sum(length(JSON3.write(get(e, "rhs", nothing))) for e in E; init = 0)
        push!(nbytes, tot)
        say(@sprintf("    part %d: %d equations, %d bytes of right-hand side", i, length(E), tot))
    end
    # `SuperFast.NO`'s tendency is the sharpest single probe: in a correct split
    # it is the mechanism term alone (a few kB), and in the degenerate one it
    # carries the whole advection stencil with it (megabytes of `index`/`ifelse`/
    # `abs`/`min` from the PPM limiters).
    for (i, d) in enumerate(docs)
        for e in eqs(d)
            lhs = JSON3.write(get(e, "lhs", nothing))
            occursin("\"SuperFast.NO\"", lhs) || continue
            r = JSON3.write(get(e, "rhs", nothing))
            cnt(op) = length(collect(eachmatch(Regex("\"" * op * "\""), r)))
            say(@sprintf("    part %d  SuperFast.NO rhs: %d bytes  index=%d ifelse=%d abs=%d min=%d",
                         i, length(r), cnt("index"), cnt("ifelse"), cnt("abs"), cnt("min")))
            break
        end
    end
    if length(nbytes) == 2 && nbytes[1] == nbytes[2]
        say("    *** THE TWO PARTS ARE BYTE-IDENTICAL: the split did not split. " *
            "Everything below is a measurement of the WHOLE model. ***")
    end
    return nothing
end
split_sanity(docs)

# --------------------------------------------------------------------------- #
# 2. Slot names.
# --------------------------------------------------------------------------- #
# `var_map` is the PUBLIC (ODE-only) map, and it is the SAME for both halves. The
# materialized-observed block above it is build-owned scratch, has no public name
# source, and is DIFFERENT in each half -- the two halves materialize different
# observeds, so slot 4705 names one thing in the transport half and another in the
# chemistry half. `RESEACT_SLOT_NAMES` therefore points at a DIRECTORY of
# `extvarmap-<n_total>.json` files (one per build, dumped by a temporary build
# patch) and the map is selected per half by its own `n_total`. Naming a
# transport slot out of the chemistry half's map is exactly the mistake this
# indirection exists to prevent.
const STATENAMES = Dict{Int,String}()
for (nm, ix) in var_map; STATENAMES[Int(ix)] = String(nm); end
const EXTNAMES = Dict{Int,Dict{Int,String}}()
let d = get(ENV, "RESEACT_SLOT_NAMES", "")
    if !isempty(d) && isdir(d)
        for f in readdir(d; join = true)
            m = match(r"extvarmap-(\d+)\.json$", f)
            m === nothing && continue
            nt = parse(Int, m.captures[1])
            EXTNAMES[nt] = Dict{Int,String}(ix => nm for (nm, ix) in
                JSON3.read(read(f, String), Dict{String,Int}))
            say("  slot names: $(length(EXTNAMES[nt])) for n_total=$nt from $(basename(f))")
        end
    end
    isempty(EXTNAMES) &&
        say("  slot names: ODE states only (set RESEACT_SLOT_NAMES to the dump directory)")
end
const CURNAMES = Ref(STATENAMES)
slotname(s::Int) = get(CURNAMES[], s, "flat slot $s")

# --------------------------------------------------------------------------- #
# 3. The read walkers -- `_de_scalar` / `_de_acck`, restricted to slot reads.
# --------------------------------------------------------------------------- #
const NK = (state = EA._NK_STATE, sg = EA._NK_STATE_GATHER,
            cl = EA._NK_CONTRACTION_LOOP, access = EA._NK_ACCESS,
            subcall = EA._NK_SUBCALL, reduce = EA._NK_REDUCE)
const AK_STATE = (EA._AK_STATE_AFFINE, EA._AK_STATE_INDIRECT,
                  EA._AK_STATE_INDIRECT_COL, EA._AK_STATE_TBL_BOX)

function reads_scalar!(acc::Set{Int}, nd)
    k = nd.kind
    if k === NK.state
        push!(acc, nd.idx)
    elseif k === NK.sg
        sg = nd.payload::EA._StateGather
        off = 0; ok = true
        for d in eachindex(nd.children)
            sub = EA._oop_index_int(nd.children[d], nothing, nothing, 0.0,
                                    Float64[], EA._OOP_NO_FORCING)
            if !(sg.lo[d] <= sub <= sg.hi[d]); ok = false; break; end
            off += (sub - sg.lo[d]) * sg.strides[d]
        end
        ok && push!(acc, sg.slot_flat[off + 1])   # out of range => a ghost 0.0
    elseif k === NK.cl
        spec = nd.payload::EA._ContractLoop
        for kk in spec.lo:spec.step:spec.hi
            spec.ref[] = kk
            reads_scalar!(acc, nd.children[1])
        end
    else
        for ch in nd.children; reads_scalar!(acc, ch); end
    end
    return acc
end

function reads_acck!(acc::Set{Int}, nd, K, plan)
    k = nd.kind
    if k === NK.access
        a = K.acc[nd.idx]
        if a.kind in AK_STATE
            union!(acc, plan.gathers[nd.idx])
        elseif a.kind === EA._AK_STATE_FIXED
            push!(acc, a.idx)
        end
    elseif k === NK.subcall
        S = nd.payload::EA._AccKernel
        j = findfirst(x -> x === S, plan.subs)
        j === nothing && error("sub-kernel not in the parent plan")
        Sp = plan.sub_plans[j]
        for r in S.cse.inv_recipes; reads_acck!(acc, r, S, Sp); end
        for r in S.cse.recipes;     reads_acck!(acc, r, S, Sp); end
        reads_acck!(acc, S.spine, S, Sp)
    elseif k === NK.reduce
        Ep = plan.red_plan[1]
        reads_acck!(acc, nd.children[1], K, Ep)
    else
        for ch in nd.children; reads_acck!(acc, ch, K, plan); end
    end
    return acc
end

function reads_kernel(K, plan)
    acc = Set{Int}()
    for j in eachindex(plan.subs)
        S = plan.subs[j]; Sp = plan.sub_plans[j]
        for r in S.cse.inv_recipes; reads_acck!(acc, r, S, Sp); end
    end
    for r in K.cse.inv_recipes; reads_acck!(acc, r, K, plan); end
    for r in K.cse.recipes;     reads_acck!(acc, r, K, plan); end
    reads_acck!(acc, K.spine, K, plan)
    return acc
end

# --------------------------------------------------------------------------- #
# 4. Replay `_de_emit!`'s walk order over a "written" vector.
# --------------------------------------------------------------------------- #
# Units in emission order, each tagged with its SECTION: materialization level
# `li` for a fill, `nlev + 1` for everything from the CSE prelude on. The
# section is what separates the two readings of a read-before-write. Both
# evaluators run these units in this order over an extended vector that starts
# at zero, so an unwritten slot reads 0.0 on host:
#
#   * nothing writes it, or only THIS section does (an in-place prefix scan
#     reads its own slot and then overwrites it) -- the zero is correct and the
#     emitter must supply one;
#   * a LATER section writes it -- the level plan is wrong, and the two
#     interpreters stop agreeing (out-of-place folds in a zero, in-place folds
#     in the previous call's value out of its reused buffer).
struct Unit
    what::String
    section::Int
    reads_ue::Set{Int}
    writes_ue::Vector{Int}
    reads_du::Set{Int}
    writes_du::Vector{Int}
    scan::Any               # the `_ScanFold` when this unit is one, else nothing
end
Unit(what, section, rue, wue, rdu, wdu) = Unit(what, section, rue, wue, rdu, wdu, nothing)

function units_of(f)
    rhs = EA.rhs_with_buffers(f)
    mat  = getfield(rhs, :mat_levels)
    nlev = length(mat)
    U = Unit[]
    for (li, lvl) in enumerate(mat)
        scalars, kernels, plans, scans = lvl[1], lvl[2], lvl[3], lvl[4]
        for (slot, nd) in scalars
            push!(U, Unit("the observed fill of $(slotname(slot)) (materialization level $li)",
                          li, reads_scalar!(Set{Int}(), nd), Int[slot], Set{Int}(), Int[]))
        end
        for j in eachindex(kernels)
            os = plans[j].out_slots
            push!(U, Unit("the observed fill kernel writing $(slotname(first(os)))" *
                          (length(os) == 1 ? "" : " … $(slotname(last(os))) ($(length(os)) lanes)") *
                          " (materialization level $li)",
                          li, reads_kernel(kernels[j], plans[j]), copy(os),
                          Set{Int}(), Int[]))
        end
        for (si, S) in enumerate(scans)
            # `_de_scan!` gathers each scan position's lanes BEFORE writing them
            # back, so every one of the fold's own slots is a read.
            push!(U, Unit("a prefix scan at materialization level $li (fold $si of $(length(scans)))",
                          li, Set{Int}(S.slots), copy(S.slots), Set{Int}(), Int[], S))
        end
    end
    sec = nlev + 1
    prelude = getfield(rhs, :cse_prelude)
    for s in eachindex(prelude)
        push!(U, Unit("shared subexpression $s of the CSE prelude", sec,
                      reads_scalar!(Set{Int}(), prelude[s]), Int[], Set{Int}(), Int[]))
    end
    for (slot, nd) in getfield(rhs, :rhs_list)
        push!(U, Unit("the state equation for $(slotname(slot))", sec,
                      reads_scalar!(Set{Int}(), nd), Int[], Set{Int}(), Int[slot]))
    end
    kernels = getfield(rhs, :acc_kernels); plans = getfield(rhs, :acc_plans)
    for j in eachindex(kernels)
        os = plans[j].out_slots
        push!(U, Unit("the access kernel writing $(slotname(first(os)))" *
                      (length(os) == 1 ? "" : " … $(slotname(last(os))) ($(length(os)) lanes)"),
                      sec, reads_kernel(kernels[j], plans[j]), Int[], Set{Int}(), copy(os)))
    end
    for (si, S) in enumerate(getfield(rhs, :scan_folds))
        push!(U, Unit("a prefix scan over the state equations (fold $si)", sec,
                      Set{Int}(), Int[], Set{Int}(S.slots), copy(S.slots)))
    end
    return U
end

function report(name, f)
    rhs = EA.rhs_with_buffers(f)
    n_states = getfield(rhs, :n_states)::Int
    n_total  = getfield(rhs, :n_total)::Int
    mat = getfield(rhs, :mat_levels)
    nlev = length(mat)
    # This half's own extended map, or the ODE names alone if none was dumped.
    CURNAMES[] = haskey(EXTNAMES, n_total) ?
                 merge(EXTNAMES[n_total], STATENAMES) : STATENAMES
    say("\n" * "-"^78)
    say("  $name: n_states=$n_states  n_total=$n_total  mat block=$(n_total - n_states) cells  " *
        "levels=$nlev")
    say(@sprintf("    state acc kernels=%d  state scan folds=%d  state array contractions=%d",
                 length(getfield(rhs, :acc_kernels)),
                 length(getfield(rhs, :scan_folds)),
                 length(getfield(rhs, :array_contractions))))
    for (li, lvl) in enumerate(mat)
        nsc = length(lvl[1]); nk = length(lvl[2]); nS = length(lvl[4]); nac = length(lvl[5])
        say(@sprintf("    level %2d: scalars=%-5d kernels=%-3d scans=%-3d contractions=%d",
                     li, nsc, nk, nS, nac))
        for (si, S) in enumerate(lvl[4])
            nl = div(length(S.slots), S.len)
            say(@sprintf("       scan %d: len=%d lanes=%d oplus=%s inclusive=%s slots %d..%d",
                         si, S.len, nl, S.oplus, S.inclusive,
                         minimum(S.slots), maximum(S.slots)))
        end
    end

    U = units_of(f)
    # First writer PER SECTION, exactly as `_de_plan_writes!` computes it.
    wsec_ue = zeros(Int, n_total); wsec_du = zeros(Int, n_states)
    for u in U
        for s in u.writes_ue; wsec_ue[s] == 0 && (wsec_ue[s] = u.section); end
        for s in u.writes_du; wsec_du[s] == 0 && (wsec_du[s] = u.section); end
    end

    written_ue = falses(n_total); written_ue[1:n_states] .= true
    written_du = falses(n_states)
    nviol = 0
    firstviol = nothing
    never = Set{Int}(); samesec = Set{Int}(); later = Set{Int}()
    for u in U
        for s in sort!(collect(u.reads_ue))
            written_ue[s] && continue
            nviol += 1
            w = wsec_ue[s]
            cls = w == 0 ? never : (w <= u.section ? samesec : later)
            push!(cls, s)
            firstviol === nothing && (firstviol = (u, s, :ue, w))
        end
        for s in sort!(collect(u.reads_du))
            written_du[s] && continue
            nviol += 1
            w = wsec_du[s]
            cls = w == 0 ? never : (w <= u.section ? samesec : later)
            push!(cls, s)
            firstviol === nothing && (firstviol = (u, s, :du, w))
        end
        for s in u.writes_ue; written_ue[s] = true; end
        for s in u.writes_du; written_du[s] = true; end
    end

    say("    units=$(length(U))  read-before-write occurrences=$nviol")
    say("      no section writes it at all             (ZERO, correct): $(length(never)) slots")
    say("      only THIS section writes it, later on   (ZERO, correct): $(length(samesec)) slots")
    say("      a LATER section writes it            (ORDERING BUG):  $(length(later)) slots")
    if firstviol !== nothing
        u, s, which, w = firstviol
        say("    FIRST slot the emitter meets unwritten:")
        say("      rule    : $(u.what)")
        say("      slot    : $(slotname(s))  ($which map)")
        say("      section : reading in section $(u.section); first written in section " *
            (w == 0 ? "(none)" : string(w)))
        say("      class   : " * (w == 0 ? "never written -- a zero, and correct" :
                                  w <= u.section ?
                                  "written by THIS section, later on -- a zero, and correct" :
                                  "written by a LATER section -- ORDERING BUG"))
    end

    # Per-scan anatomy: for each fold, which scan POSITIONS were already filled
    # by this level's kernels and which the fold meets unwritten. A fold whose
    # unwritten positions are exactly its last one is reading the term slot the
    # exclusive accumulation never uses -- dead on both sides.
    filled = falses(n_total); filled[1:n_states] .= true
    for u in U
        S = u.scan
        if S !== nothing
            len = S.len; nl = div(length(S.slots), len)
            nunw(kk) = count(l -> !filled[S.slots[(l - 1) * len + kk]], 1:nl)
            unw = Int[kk for kk in 1:len if nunw(kk) > 0]
            say("    $(u.what):")
            say("      len=$len lanes=$nl; positions the fold meets UNWRITTEN: " *
                (isempty(unw) ? "(none)" : string(unw)) * " of 1..$len; " *
                "lanes affected per position: " * string(Int[nunw(kk) for kk in unw]))
            say("      slot at (lane 1, position $len) = $(slotname(S.slots[len]))" *
                ";  at (lane $nl, position $len) = $(slotname(S.slots[end]))")
        end
        for s in u.writes_ue; filled[s] = true; end
    end

    for (label, set) in (("never written", never), ("same section", samesec),
                         ("LATER section", later))
        isempty(set) && continue
        ss = sort!(collect(set))
        say("    $label: $(length(ss)) slots, range $(first(ss))..$(last(ss)), " *
            "$(count(s -> s > n_states, ss)) in the mat block")
        for s in ss[1:min(end, 6)]
            owners = String[]
            for (li, lvl) in enumerate(mat), (si, S) in enumerate(lvl[4])
                if s in S.slots
                    ii = findfirst(==(s), S.slots)
                    push!(owners, "level $li scan $si lane $(div(ii - 1, S.len) + 1) " *
                                  "position $((ii - 1) % S.len + 1) of $(S.len)")
                end
            end
            say("      $(slotname(s)): " * (isempty(owners) ? "(in no scan)" : join(owners, "; ")))
        end
    end
    return (; n_states, n_total, nviol, never, samesec, later)
end

const R1 = report("transport", fo[1])
const R2 = report("chemistry", fo[2])

say("\n" * "="^78)
for (nm, R) in (("transport", R1), ("chemistry", R2))
    say(@sprintf("%-10s read-before-write=%d  never=%d  same-section=%d  LATER-section=%d",
                 nm, R.nviol, length(R.never), length(R.samesec), length(R.later)))
end
say("="^78)
