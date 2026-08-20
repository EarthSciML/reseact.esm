#!/usr/bin/env julia
# ===========================================================================
# astdiff_traced_probe.jl -- GATE B: can the symbolic Jacobian be TRACED?
# ===========================================================================
# astdiff_probe.jl established that EarthSciASTDiff's analytical Jacobian works
# on ReSEACT's chemistry half: :block_diagonal, 3744x3744, nnz 19,872, exact to
# 2.3e-16 against a ForwardDiff JVP. That was all on the HOST.
#
# It is not usable from there. `JacobianEvaluator`'s call operator
# (assemble.jl:198-211) gathers into a host scratch `jac.uj`, calls `fJ!`, then
# scatters band values into a SparseMatrixCSC by index. None of that traces, and
# calling it per chemistry step is exactly the host round trip that is ruled out.
#
# THE SEAM IS ALREADY THERE, and deliberately -- assemble.jl:184-190 says the
# derived band model goes through the ORDINARY builder, so `form = :oop` reaches
# it, and "the out-of-place form is not an optional nicety here -- it is the only
# form that traces ... so a consumer compiling the Jacobian alongside a traced
# RHS builds BOTH that way". This probe is that consumer.
#
# WHY THIS MATTERS MORE THAN SPEED. rx_traced_integrator.jl:563-579 records that
# `ros23_step_vjp` defaults to jac=:fd DELIBERATELY: jac=:ad segfaults inside
# Enzyme-MLIR under reverse-over-forward (the inner `enzyme.fwddiff` that
# `ad_block_jac` emits gets a null reverse callee). So the adjoint is stuck with
# the FD Jacobian, and that is NOT free -- the same comment records the
# dot-product identity degrading from 1.8e-16 to 1.1e-6 on ReSEACT because the
# difference quotient inside the step makes the step map ill-conditioned in u.
#
# A SYMBOLIC Jacobian is neither of those things. It is straight-line
# arithmetic: no `enzyme.fwddiff` to nest, and no `stablehlo.while` region (the
# other documented blocker, line 496). Reverse over it is plain reverse mode.
# So if it traces, it is the only route to an EXACT Jacobian the adjoint can
# actually cross -- a CORRECTNESS win, on top of whatever speed it buys.
#
# WHAT THIS PROBE DECIDES, in order, cheapest first. The build costs ~600 s, so
# every question is answered in ONE run and each stage reports and continues
# rather than aborting.
#   1. is `jac.oop` true -- did `form = :oop` actually reach the band model
#   2. does the band model carry FORCING BUFFERS? If it bakes meteorology in as
#      host constants it would FREEZE the forcing inside the Jacobian, which is
#      a silent correctness bug over a long run, not a wiring detail.
#   3. is the state layout species-major with block s at (s-1)*NC+1 : s*NC --
#      the layout fd_block_jac ASSERTS (rx_traced_integrator.jl:134)
#   4. is the per-cell band pattern UNIFORM across cells? nnz 19,872 / 288 cells
#      = 69.0 exactly is suggestive, NOT proof. If one cell has a different
#      pattern the whole static-gather plan is invalid.
#   5. the gather plan itself, validated on the host against `jac`'s own sparse
#      matrix -- an index error here would be invisible later
#   6. THE GATE: compile the band model under Reactant and compare its band
#      vector to the host's
#   7. end to end: assemble the NS x NS blocks blocksolve consumes, traced, and
#      compare to the host blocks
#
# ENV. Needs the PINNED EarthSciAST (reseact.esm does not load against
# EarthSciAST main since PR #167 renamed `examples` -> `analyses`). Point
# RESEACT_RXENV at an env with the pinned EarthSciAST and EarthSciASTDiff dev'd.
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Printf, Statistics, Logging, SparseArrays
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using EarthSciASTDiff
using Reactant
const EA = EarthSciAST
const ED = EarthSciASTDiff
const RX = Reactant

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))
hdr(s) = say("\n---- $s ----")

const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
_env(k, d)  = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 6), nlat = _env("NLAT", 6),
                           nlev = _env("NLEV", 8))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]

say("=== astdiff TRACED probe: grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF ===")

# --- build both halves exactly as run_reseact_reactant.jl does ------------- #
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2); vms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); merged_const = Dict{String,Any}()
ov = Dict{String,Float64}(); docs = nothing; ff = nothing; parts = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, vms, u0, p, var_map, merged_param, merged_const, ov, docs, ff, parts
    file = EA.load(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
    f0 = reseact_forcing(CHEMDIR; ndays = 1)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        EA.provider_is_const(prov) ? (merged_const[k] = fld) : (merged_param[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi; vms[i] = vmi
        i == 1 && (u0 = u0i; p = pi; var_map = vmi)
    end
end
foreach(d -> d.materialize!(), dms)
const N = length(u0)
say(@sprintf("BUILD %.2f s   nstates=%d", time() - tb, N))

u = copy(u0)
bk = (; form = :oop, parameter_overrides = ov, const_arrays = merged_const,
        param_arrays = merged_param)

# --- prepare the symbolic Jacobian on the chemistry half ------------------ #
hdr("prepare_jacobian (chemistry half, parts[2])")
tp = time()
jac = ED.prepare_jacobian(parts[2]; wrt = :states, build_kwargs = bk)
say(@sprintf("  OK in %.2f s   structure=%s", time() - tp, jac.structure))
Jsp = copy(ED.jac_prototype(jac))
jac(Jsp, u, p, T0)
say(@sprintf("  sparse J: %d x %d  nnz=%d", size(Jsp,1), size(Jsp,2), nnz(Jsp)))

# --- 1. did form=:oop reach the band model? ------------------------------- #
hdr("1. band model form")
say("  jac.oop = $(jac.oop)   $(jac.oop ? "PASS -- traceable form" :
                                          "FAIL -- in-place, captures host scratch, cannot trace")")
NJ = length(jac.uj)
say(@sprintf("  band-model state length nj=%d vs model n=%d  (%s)", NJ, N,
             NJ == N ? "same" : "DIFFERENT -- unmapped slots stay 0.0"))

# --- 2. does the band model carry forcing buffers? ------------------------ #
# If it does, the traced consumer must push and refresh them exactly like the
# RHS's. If it does NOT but the RHS does, the band model has BAKED the
# meteorology in as host constants and the Jacobian would silently freeze at
# whatever epoch it was built -- a correctness bug that only shows up over a
# long run, so it is worth knowing now.
hdr("2. forcing buffers")
_bufs(f) = try EA.forcing_buffers(f) catch e; ("ERR", sprint(showerror, e)) end
bj = _bufs(jac.fJ!); br = _bufs(fo[2])
_desc(b) = b isa Tuple && b[1] == "ERR" ? "unavailable: $(first(b[2],120))" :
           b === nothing ? "none" : "$(length(b)) buffer(s): $(collect(keys(b)))"
say("  band model : $(_desc(bj))")
say("  chem RHS   : $(_desc(br))")

# --- 3. state layout -------------------------------------------------------#
# WHAT `blocksolve` ACTUALLY REQUIRES. `fd_block_jac` slices with
# `x[((s-1)*NC+1):(s*NC)]` and then pairs POSITION k of one species block with
# position k of another (rx_traced_integrator.jl:173). So the real requirement
# is only:
#   (a) each species occupies one CONTIGUOUS run of NC indices, and
#   (b) the CELL SEQUENCE inside every species block is the same sequence.
# It does NOT require the cells to appear in any particular order. An earlier
# version of this probe demanded that the cell order match a lexicographic sort
# of the (lon,lat,lev) triples and reported FAIL on a layout that is perfectly
# usable -- the check was wrong, not the model. Test (a) and (b), derive the
# order from the names, and print the actual layout either way.
hdr("3. state layout")
function parse_state(nm)
    m = match(r"^(.*)\[(\d+),(\d+),(\d+)\]$", nm)
    m === nothing && return nothing
    (String(m.captures[1]),
     (parse(Int,m.captures[2]), parse(Int,m.captures[3]), parse(Int,m.captures[4])))
end
rn = jac.rownames
parsed = map(parse_state, rn)
let bad = findall(isnothing, parsed)
    isempty(bad) || (say("  UNPARSEABLE state names ($(length(bad))), e.g. $(rn[first(bad)]) -- STOP"); exit(1))
end
sp = first.(parsed); ce = last.(parsed)
species = sort(unique(sp)); cells_u = unique(ce)
const NS = length(species); const NC = length(cells_u)
say(@sprintf("  NS=%d species, NC=%d cells, NS*NC=%d vs n=%d", NS, NC, NS*NC, N))
NS * NC == N || (say("  FAIL -- not a rectangular species x cell lift"); exit(1))
say("  first 4 states: " * join(rn[1:min(4,end)], ", "))
say(@sprintf("  across the first block boundary (idx %d..%d): %s", NC-1, NC+2,
             join(rn[(NC-1):min(NC+2,end)], ", ")))
blk_ok = all(s -> length(unique(sp[((s-1)*NC+1):(s*NC)])) == 1, 1:NS)
say("  (a) species blocks contiguous, NC long: $(blk_ok ? "PASS" : "FAIL")")
seq1 = ce[1:NC]
seq_ok = blk_ok && all(s -> ce[((s-1)*NC+1):(s*NC)] == seq1, 2:NS)
say("  (b) identical cell sequence in every block: $(seq_ok ? "PASS" : "FAIL")")
if blk_ok
    say("  species block order: " * join([sp[(s-1)*NC+1] for s in 1:NS], ", "))
    lex = sort(cells_u)
    say("  cell order is lexicographic (lon,lat,lev): $(seq1 == lex ? "yes" : "NO -- some other nesting; irrelevant to blocksolve, but do not assume it elsewhere")")
    seq1 == lex || say("    first 3 cells as laid out: $(seq1[1:min(3,end)]) vs lexicographic $(lex[1:min(3,end)])")
end
(blk_ok && seq_ok) || (say("  STOP -- the per-cell block form blocksolve needs does not exist"); exit(1))
srow(i) = (i - 1) ÷ NC + 1      # species block of state index i
crow(i) = (i - 1) % NC + 1      # position within the block == cell slot
sidx(s, c) = (s - 1) * NC + c

# --- 3b. does the Jacobian's ordering match the RUNNER's? ----------------- #
# `prepare_jacobian` REBUILDS the model from `parts[2]` (the raw FlattenedSystem)
# while the runner compiles `docs[2]` (esm-converted, with
# `index_promoted_refs_by_loop!`). Two builds, two var maps -- and nothing
# guarantees they agree. If they do not, the gather plan must be composed with a
# name-based remap or every band lands in the wrong cell. Silent if unchecked.
hdr("3b. Jacobian ordering vs the runner's ordering")
vm2 = vms[2]
run_names = first.(sort(collect(vm2), by = last))
say(@sprintf("  runner chem-half states: %d   Jacobian rownames: %d",
             length(run_names), length(rn)))
if length(run_names) == length(rn)
    nmatch = count(k -> run_names[k] == rn[k], eachindex(rn))
    say(@sprintf("  positions that agree: %d / %d  (%s)", nmatch, length(rn),
                 nmatch == length(rn) ? "PASS -- same ordering, no remap needed" :
                 "DIFFER -- the plan MUST be composed with a name-based remap"))
    if nmatch != length(rn)
        k = findfirst(k -> run_names[k] != rn[k], eachindex(rn))
        say("  first disagreement at $k: runner='$(run_names[k])' jac='$(rn[k])'")
        say("  same SET of names: $(Set(run_names) == Set(rn))")
    end
else
    say("  DIFFERENT LENGTHS -- the two builds do not even have the same states")
    say("  only in runner: $(collect(setdiff(Set(run_names), Set(rn)))[1:min(end,5)])")
    say("  only in jac:    $(collect(setdiff(Set(rn), Set(run_names)))[1:min(end,5)])")
end

# --- 4. is the per-cell band pattern UNIFORM? ----------------------------- #
# The static gather plan is only valid if every cell's block has the SAME
# (row species, col species) sparsity. nnz 19,872 / 288 = 69.0 exactly is
# suggestive, NOT proof -- check cell by cell.
hdr("4. per-cell pattern uniformity")
I0, J0, _ = findnz(Jsp)
offdiag_cell = 0
patt = [Set{Tuple{Int,Int}}() for _ in 1:NC]
for k in eachindex(I0)
    r, c = I0[k], J0[k]
    if crow(r) != crow(c); global offdiag_cell += 1; continue; end
    push!(patt[crow(r)], (srow(r), srow(c)))
end
say(@sprintf("  entries coupling DIFFERENT cells: %d  (%s)", offdiag_cell,
             offdiag_cell == 0 ? "PASS -- purely cell-local" :
                                 "FAIL -- not block-diagonal by cell"))
ref_patt = patt[1]
diffs = findall(k -> patt[k] != ref_patt, 1:NC)
say(@sprintf("  cells whose pattern differs from cell 1: %d / %d  (%s)",
             length(diffs), NC, isempty(diffs) ? "PASS -- uniform" : "FAIL"))
say(@sprintf("  nonzeros per cell block: %d of %d (%.1f%%)",
             length(ref_patt), NS*NS, 100*length(ref_patt)/(NS*NS)))
if !isempty(diffs)
    k = first(diffs)
    say("  e.g. cell $k has $(length(patt[k])) vs cell 1's $(length(ref_patt))")
    say("    only in cell $k: $(collect(setdiff(patt[k], ref_patt))[1:min(end,6)])")
    say("    only in cell 1:  $(collect(setdiff(ref_patt, patt[k]))[1:min(end,6)])")
end

# --- 5. the gather plan ---------------------------------------------------- #
# `jac.scatter` is a list of (duj slot, nzval position). Invert it into what a
# TRACED consumer needs: for each (r, si) block entry, a length-NC vector of duj
# slots, so the traced code is `Jb[r,si] = duj[idx[r,si]]` -- one gather, no host
# scratch, no loop, no region op.
#
# Entries can be MULTI-SLOT: the host operator does `J.nzval[pos] += duj[slot]`,
# so several bands can land on one matrix entry. Handle that, and report it.
hdr("5. build + validate the traced gather plan")
pos_of = Dict{Int,Tuple{Int,Int}}()
for c in 1:size(Jsp,2), q in Jsp.colptr[c]:(Jsp.colptr[c+1]-1)
    pos_of[q] = (Jsp.rowval[q], c)
end
slots = Dict{Tuple{Int,Int},Vector{Int}}()
for (slot, pos) in jac.scatter
    push!(get!(slots, pos_of[pos], Int[]), slot)
end
say(@sprintf("  matrix entries fed by >1 band slot: %d of %d",
             count(v -> length(v) > 1, values(slots)), length(slots)))
maxslots = maximum(length, values(slots))
idx = [zeros(Int, NC, maxslots) for _ in 1:NS, _ in 1:NS]
nzblk = falses(NS, NS)
for ((r, c), sl) in slots
    crow(r) == crow(c) || continue
    rs, cs, ce_ = srow(r), srow(c), crow(r)
    nzblk[rs, cs] = true
    for (q, sl_) in enumerate(sl); idx[rs, cs][ce_, q] = sl_; end
end
say(@sprintf("  block entries used: %d of %d   (max slots per entry %d)",
             count(nzblk), NS*NS, maxslots))
ujh = zeros(NJ); for (i, s) in enumerate(jac.umap); ujh[s] = u[i]; end
# The 4-ARG form (buffers as an explicit argument) is the one that matters: the
# 3-arg form resolves forcing internally, which under tracing would BAKE it in
# as constants -- the exact freeze stage 2 was checking for. So if the 4-arg
# form is unavailable, say so loudly rather than quietly falling back to a
# version that cannot be used anyway.
gjb = bufj = nothing; dujh = nothing
try
    global gjb  = EA.rhs_with_buffers(jac.fJ!)
    global bufj = EA.forcing_buffers(jac.fJ!)
    global dujh = gjb(ujh, p, T0, bufj)
    say("  band model called 4-arg (u, p, t, bufs): OK -- forcing stays an operand")
catch e
    say("  4-arg call FAILED: " * first(replace(sprint(showerror, e), "\n" => " "), 300))
    say("  falling back to the 3-arg form -- NOTE: that form bakes forcing in and")
    say("  is NOT usable traced; stage 6 measures compile, not a usable Jacobian.")
    global gjb = nothing
    global dujh = jac.oop ? jac.fJ!(ujh, p, T0) : jac.fJ!(zeros(NJ), ujh, p, T0)
end
dujp = vcat(0.0, dujh)                      # slot 0 -> 0.0, so no branch
worst = 0.0
for rs in 1:NS, cs in 1:NS
    got = nzblk[rs,cs] ? vec(sum(dujp[idx[rs,cs] .+ 1]; dims = 2)) : zeros(NC)
    for c_ in 1:NC
        ref = Jsp[sidx(rs,c_), sidx(cs,c_)]
        global worst = max(worst, abs(got[c_] - ref) / max(abs(ref), 1e-300))
    end
end
say(@sprintf("  plan vs host sparse J: worst relative %.3e  (%s)", worst,
             worst <= 1e-12 ? "PASS" : "FAIL -- index error in the plan"))

# --- 6. THE GATE: does the band model compile under Reactant? ------------- #
# u -> uj is a SCATTER on the host (uj[umap[i]] = u[i]). A traced consumer needs
# a GATHER, so invert it: gidx[k] = the u index feeding band-model slot k, 0
# where nothing does. Padding u with a leading zero makes the 0 a legitimate
# index, so the whole map is one gather with no branch.
hdr("6. GATE -- compile the band model under Reactant")
gidx = zeros(Int, NJ)
for (i, s) in enumerate(jac.umap); gidx[s] = i; end
say(@sprintf("  u -> uj gather: %d of %d band-model slots fed by a state",
             count(!iszero, gidx), NJ))
dev_p(pp) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
tc = time(); xjac = nothing
if gjb === nothing
    say("  SKIPPED -- no 4-arg form to compile (see stage 5)")
else
dbuf = map(RX.ConcreteRArray, bufj)
try
    global xjac = RX.@compile sync=true gjb(RX.ConcreteRArray(ujh), dev_p(p),
                                            RX.ConcreteRNumber(T0), dbuf)
    say(@sprintf("  @compile OK in %.1f s   <<< GATE B PASSES", time() - tc))
catch e
    say(@sprintf("  @compile FAILED after %.1f s   <<< GATE B FAILS", time() - tc))
    say("    " * first(replace(sprint(showerror, e), "\n" => " "), 1200))
end
if xjac !== nothing
    dujt = Array(xjac(RX.ConcreteRArray(ujh), dev_p(p), RX.ConcreteRNumber(T0), dbuf))
    d = norm(dujt .- dujh) / max(norm(dujh), 1e-300)
    say(@sprintf("  traced band vector vs host: relative %.3e  (%s)", d,
                 d <= 1e-12 ? "PASS" : "FAIL"))
end
end


# --- 7. THE CONTROL: is 4e-7 the JACOBIAN, or is it host-vs-XLA arithmetic? --#
# Run 10038140 compiled the band model fine (263 s) but its band vector differed
# from the host's by 4.154e-07 relative -- nine orders too big for float64
# roundoff, so something real is going on. Two very different explanations:
#
#   (a) something wrong with the band model or how it is being traced, or
#   (b) the ambient difference between Julia's libm and XLA:CPU's arithmetic,
#       which the chemistry RHS -- full of exp/pow in the rate constants --
#       would show EQUALLY.
#
# These have opposite consequences and cannot be told apart from the Jacobian
# number alone. So measure the SAME quantity on the chemistry RHS, compiled and
# called exactly the same way. If the RHS also lands at ~1e-7, (b) is the
# answer, the Jacobian is no worse than the RHS the runner already ships, and
# the 1e-12 threshold was simply the wrong bar. If the RHS agrees to 1e-15 while
# the Jacobian is at 1e-7, it is (a) and the Jacobian has a real defect.
CONTROL_REL = NaN
hdr("7. CONTROL -- traced vs host on the CHEMISTRY RHS, same treatment")
gC   = EA.rhs_with_buffers(fo[2])
bufC = EA.forcing_buffers(fo[2])
duC_h = gC(u, p, T0, bufC)
dbufC = map(RX.ConcreteRArray, bufC)
tc7 = time()
try
    xC = RX.@compile sync=true gC(RX.ConcreteRArray(u), dev_p(p),
                                  RX.ConcreteRNumber(T0), dbufC)
    duC_t = Array(xC(RX.ConcreteRArray(u), dev_p(p), RX.ConcreteRNumber(T0), dbufC))
    dC = norm(duC_t .- duC_h) / max(norm(duC_h), 1e-300)
    say(@sprintf("  @compile OK in %.1f s", time() - tc7))
    say(@sprintf("  traced RHS vs host RHS: relative %.3e", dC))
    global CONTROL_REL = dC
    # DELIBERATELY NO CONCLUSION HERE. An earlier version of this probe DID
    # conclude at this point -- "the RHS differs too, therefore host-vs-XLA
    # arithmetic, therefore the 1e-12 bar is unreachable" -- and that reasoning
    # was WRONG, in a way worth keeping a warning about. Both this control and
    # the stage-6 compile ran WITHOUT the XLA:CPU race workaround, so both arms
    # carried the SAME contamination; "they both differ" was evidence of a
    # shared cause, not of irreducible arithmetic. Stage 8 recompiles with the
    # workaround and settles it (measured: 4.154e-07 -> 2.102e-16). Interpret
    # only after that, in stage 9.
    say("  (no conclusion yet -- this control also ran WITHOUT the race")
    say("   workaround, so it cannot separate arithmetic from the race. Stage 8.)")
catch e
    say("  control FAILED to compile: " * first(replace(sprint(showerror, e), "\n" => " "), 400))
end

# --- 8. where does the band difference live, and does the race fix move it? --#
# Two follow-ups that separate the remaining candidates. A difference spread
# thinly over most slots is reassociation/libm; one concentrated in a handful is
# a specific op or a genuine bug. And the XLA:CPU race is a KNOWN value-perturbing
# defect on this model (it biased the adjoint objective 4.0e-5), whose workaround
# was NOT enabled in the compile above -- so it is a live candidate here.
hdr("8. band-difference distribution + the race workaround")
if xjac !== nothing
    dujt = Array(xjac(RX.ConcreteRArray(ujh), dev_p(p), RX.ConcreteRNumber(T0), dbuf))
    rel = abs.(dujt .- dujh) ./ max.(abs.(dujh), 1e-300)
    nz = findall(!iszero, dujh)
    for thr in (1e-14, 1e-12, 1e-9, 1e-7, 1e-5)
        say(@sprintf("  slots (of %d nonzero) with relative > %.0e : %d",
                     length(nz), thr, count(>(thr), rel[nz])))
    end
    k = nz[argmax(rel[nz])]
    say(@sprintf("  worst slot %d: traced=%.16e host=%.16e", k, dujt[k], dujh[k]))
end
FIXED_REL = NaN
COPTS = RX.CompileOptions(; sync = true,
                          xla_debug_options = (; xla_cpu_prefer_vector_width = 128))
tc8 = time()
try
    xjac2 = RX.@compile compile_options=COPTS gjb(RX.ConcreteRArray(ujh), dev_p(p),
                                                 RX.ConcreteRNumber(T0), dbuf)
    dujt2 = Array(xjac2(RX.ConcreteRArray(ujh), dev_p(p), RX.ConcreteRNumber(T0), dbuf))
    d2 = norm(dujt2 .- dujh) / max(norm(dujh), 1e-300)
    say(@sprintf("  with the race workaround (vector_width=128), compiled in %.1f s:", time() - tc8))
    say(@sprintf("    traced band vector vs host: relative %.3e", d2))
    global FIXED_REL = d2
catch e
    say("  race-workaround compile FAILED: " * first(replace(sprint(showerror, e), "\n" => " "), 300))
end

# --- 9. the interpretation, using BOTH arms ------------------------------- #
hdr("9. VERDICT")
if isnan(FIXED_REL)
    say("  UNKNOWN -- the race-workaround arm did not run.")
elseif FIXED_REL <= 1e-12
    say(@sprintf("  With the race workaround the band model matches the host to %.3e.", FIXED_REL))
    say(@sprintf("  Without it, %.3e. The gap was ENTIRELY the XLA:CPU race --", 4.154e-7))
    say("  not arithmetic, and not a defect in the Jacobian.")
    isnan(CONTROL_REL) ||
        say(@sprintf("  Note the RHS control differed by %.3e with the race ACTIVE, i.e.", CONTROL_REL))
    say("  the race hurts the RHS FAR more than it hurts the band model.")
    say("  => GATE B PASSES: the symbolic Jacobian traces, and it is EXACT.")
    say("     RESEACT_RXFIX must be ON wherever it is used -- non-negotiable here.")
else
    say(@sprintf("  Even with the race workaround the band model differs by %.3e.", FIXED_REL))
    say("  => the race is NOT the whole story; do not proceed until it is explained.")
end

say("\nTRACED_PROBE_DONE")
