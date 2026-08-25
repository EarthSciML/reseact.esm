#!/usr/bin/env julia
# ===========================================================================
# p4_pass_bisect.jl -- WHICH default EnzymeXLA pattern rewrites the emitter's
#                      DUS into whole-buffer concatenates?
# ===========================================================================
# CONTEXT (measured previously): the ROS23 step's out-of-place emitter writes
# its whole-buffer updates as `stablehlo.dynamic_update_slice` (298 DUS, 4
# concatenates in the UNOPTIMIZED module at 6x6x8, job 10129790). The default
# Reactant/Enzyme-JAX StableHLO pipeline hands XLA a module with 79
# whole-buffer concatenates and 0 DUS; XLA cannot alias a concatenate in
# place, so each is a ~MB-scale copy -- the largest single traffic term in the
# step. `CONCATS_TO_DUS[]=true` recovers only ~43% (1.17x at CONUS). Ruled
# out already: the `dus_to_concat` kwarg (default false) and the
# `DUS_TO_CONCAT[]` Ref -- the culprit is some OTHER pattern in the default
# `enzyme-hlo-generate-td{patterns=...}` list.
#
# QUESTION: what is the MINIMAL `excluded_passes` set (CompileOptions field;
# entries are removed by base-name from the generated pattern list) that keeps
# the emitter's DUS intact through the pipeline -- optimized module:
# concatenates ~ few, DUS ~ hundreds -- changing nothing else?
#
# METHOD (one process, one build; each probe round is one @code_hlo):
#   1. harvest the full default pattern list from
#      Reactant.Compiler.optimization_passes (both sroa variants, every
#      patterns= block); save to logs/.
#   2. census the optimized step module (concat vs DUS with result sizes,
#      big = >=400k elements) with no exclusions.
#   3. sanity: exclude EVERY pattern -- DUS must survive, else the rewrite
#      lives outside this list and excluded_passes is not the knob.
#   4. singleton tests of name-candidates (concat/dus/slice/pad), then greedy
#      chunk minimization (delta debugging) from a working superset.
#   5. verify the winner: structural census + functional bit-for-bit compare
#      of one compiled step (default COPTS vs COPTS + excluded_passes).
#
# Every CompileOptions constructed here keeps sync=true and
# xla_debug_options=(; xla_cpu_prefer_vector_width=128) -- the XLA:CPU race
# workaround (REQUIRED for correctness); exclusions are the only change.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 6 6 8; do NOT run CONUS here)
#   P4_MAXROUNDS             cap on @code_hlo probe rounds (default 60)
#   P4_BIGN                  "big result" element threshold (default 400000)
#
# RESULTS (grid 6x6x8, Reactant v0.2.280, 2026-08-25; round 2 = p4_round2.jl):
#   * baseline: concat 79 / DUS 0.  exclude-all(493): concat 8 / DUS 292.
#   * the ONLY effective singleton of 58 tried: `dynamic_update_to_concat`
#     (concat 79 -> 40, DUS 0 -> 205) -- a C-side default pattern, invisible
#     from the Julia dus_to_concat kwarg/Ref that were ruled out earlier.
#   * round-2 ddmin from exclude-all with that winner held: ONE cooperator,
#     `sub_const_prop`; the pair reaches the floor (concat 8, big 4 / DUS 292,
#     big 232). Necessity-checked; 16 rounds.
#   * step time (p4_steptime_excluded.jl, interleaved): default 10.98 ms ->
#     excluded 5.22 ms = 2.10x, bit-for-bit equal unew.
#   * TRAP: BIGN=400k classifies nothing at 6x6x8 (extended observed buffer
#     is 57708 elems) -- round 1's "rewrite lives outside this list" verdict
#     was this artifact. TRAP 2: excluding `slice_concat` INCREASES concats
#     (it is the collapse pattern) -- 79 -> 252.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu")
get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "p4bisect"

include(joinpath(@__DIR__, "_env.jl"))
using Printf
say(s) = (println(s); flush(stdout))

const GRID = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT = joinpath(REPO, "logs", "p4-bisect-$GRID-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)
say("P4_PASS_BISECT grid=$GRID out=$OUT")

const tb = time()
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
say(@sprintf("driver loaded in %.1f s", time() - tb))

const BIGN = parse(Int, get(ENV, "P4_BIGN", "400000"))
const MAXROUNDS = parse(Int, get(ENV, "P4_MAXROUNDS", "60"))

# --------------------------------------------------------------------------- #
# 1. Harvest the default pattern list.
# --------------------------------------------------------------------------- #
basename_of(p) = String(first(split(p, '<')))
function pattern_names()
    names = String[]
    for sroa in (true, false)
        s = RX.Compiler.optimization_passes(D.COPTS; sroa, backend = "cpu")
        for m in eachmatch(r"patterns=([^}]*)\}", s)
            for p in split(m.captures[1], ';')
                isempty(strip(p)) && continue
                b = basename_of(strip(p))
                b in names || push!(names, b)
            end
        end
    end
    return names
end

const ALLPATS = pattern_names()
open(joinpath(OUT, "patterns_default.txt"), "w") do io
    foreach(p -> println(io, p), ALLPATS)
end
say(@sprintf("default pattern list: %d unique base names -> %s",
             length(ALLPATS), joinpath(OUT, "patterns_default.txt")))
const CANDS = filter(p -> occursin(r"concat|dus|dynamic_update|slice|pad"i, p), ALLPATS)
say("name candidates (concat/dus/slice/pad): " * join(CANDS, ", "))

# --------------------------------------------------------------------------- #
# 2. One probe round = one @code_hlo of the step under `excluded_passes`.
# --------------------------------------------------------------------------- #
const UR = RX.ConcreteRArray(copy(D.UBASE))

function sizes_for(txt, op)
    out = Int[]
    for ln in split(txt, '\n')
        occursin(op, ln) || continue
        m = match(r"->\s*tensor<([0-9x]+)x[a-z]", ln)
        m === nothing && (m = match(r":\s*tensor<([0-9x]+)x[a-z]", ln))
        m === nothing && continue
        n = 1
        for d in split(m.captures[1], 'x'); n *= parse(Int, d); end
        push!(out, n)
    end
    return out
end

copts_excl(excl) = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
    excluded_passes = collect(String, excl))

const CACHE = Dict{Vector{String},NamedTuple}()
const NROUNDS = Ref(0)
function census(excl::Vector{String}; label = "", save = "")
    key = sort(excl)
    save == "" && haskey(CACHE, key) && return CACHE[key]
    NROUNDS[] += 1
    NROUNDS[] > MAXROUNDS && error("P4_MAXROUNDS=$MAXROUNDS exhausted")
    copts = copts_excl(excl)
    t0 = time()
    txt = try
        sprint(show, RX.@code_hlo(compile_options = copts, D.ros_step(UR, D.THC, D.T_R, D.DTC_R)))
    catch e
        say(@sprintf("  [%3d] excl=%-3d %-34s FAILED: %s", NROUNDS[], length(excl),
                     label, first(split(sprint(showerror, e), '\n'))))
        r = (ok = false, nc = -1, nbigc = -1, bigc = 0, nd = -1, nbigd = -1, bigd = 0)
        CACHE[key] = r
        return r
    end
    cc = sizes_for(txt, "stablehlo.concatenate")
    du = sizes_for(txt, "stablehlo.dynamic_update_slice")
    bigc = filter(>=(BIGN), cc); bigd = filter(>=(BIGN), du)
    r = (ok = true, nc = length(cc), nbigc = length(bigc), bigc = sum(bigc; init = 0),
         nd = length(du), nbigd = length(bigd), bigd = sum(bigd; init = 0))
    say(@sprintf("  [%3d] excl=%-3d %-34s %5.0fs  concat %4d (big %3d, %6.0f MB)  DUS %4d (big %3d)",
                 NROUNDS[], length(excl), label, time() - t0,
                 r.nc, r.nbigc, r.bigc * 8 / 1e6, r.nd, r.nbigd))
    if save != ""
        open(joinpath(OUT, save), "w") do io; write(io, txt); end
    end
    CACHE[key] = r
    return r
end

# DUS restored = the module still carries the emitter's whole-buffer updates
# as DUS and the big concatenates are (near) gone.
restored(r) = r.ok && r.nbigd >= 50 && r.nbigc <= 10

# --------------------------------------------------------------------------- #
# 3. Search phases (functions: top-level soft scope is a known probe trap).
# --------------------------------------------------------------------------- #
function find_singleton()
    say("\n---- singleton name-candidates ----")
    for c in CANDS
        restored(census([c]; label = c)) && return [c]
    end
    return nothing
end

# Greedy chunk removal (ddmin on the complement): end state is 1-minimal --
# removing any single remaining name breaks restoration.
function minimize(E::Vector{String})
    chunk = max(1, length(E) ÷ 2)
    while true
        i = 1
        while i <= length(E)
            j = min(i + chunk - 1, length(E))
            trial = vcat(E[1:(i - 1)], E[(j + 1):end])
            if !isempty(trial) && restored(census(trial; label = "drop $(i):$(j)"))
                E = trial
            else
                i = j + 1
            end
        end
        chunk == 1 && break
        chunk = max(1, chunk ÷ 2)
    end
    return E
end

function search(allx)
    w = find_singleton()
    w !== nothing && return w
    say("\n---- the candidate set together ----")
    if restored(census(copy(CANDS); label = "all-candidates"))
        say("\n---- minimizing within the candidate set ----")
        return minimize(copy(CANDS))
    elseif restored(allx)
        say("\n---- delta-debugging the full pattern list ----")
        return minimize(copy(ALLPATS))
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# 4. Baseline, sanity, search.
# --------------------------------------------------------------------------- #
say("\n---- baseline (no exclusions) ----")
const BASE = census(String[]; label = "baseline", save = "step.opt.default.mlir")
say("\n---- sanity: exclude EVERYTHING ----")
const ALLX = census(copy(ALLPATS); label = "exclude-all")
restored(ALLX) || say("!! excluding every generated pattern does NOT restore DUS " *
                      "-- the rewrite lives outside this list.")

const WINNER = search(ALLX)
if WINNER === nothing
    say("\nNO MINIMAL SET FOUND -- see the log above.")
    exit(1)
end
say("\n==== WINNER: excluded_passes = [" * join(WINNER, ", ") * "] ====")
const WR = census(copy(WINNER); label = "winner", save = "step.opt.excluded.mlir")
open(joinpath(OUT, "winner.txt"), "w") do io
    foreach(p -> println(io, p), WINNER)
end

# --------------------------------------------------------------------------- #
# 5. Functional verification: compile both, compare one step bit-for-bit.
#    Fresh input buffers per call -- donated_args=:auto may invalidate inputs.
# --------------------------------------------------------------------------- #
say("\n---- functional check: one step, default vs excluded ----")
const CEXC = let copts = copts_excl(WINNER)
    t0 = time()
    c = RX.@compile compile_options = copts D.ros_step(D.U_R, D.THC, D.T_R, D.DTC_R)
    say(@sprintf("  @compile excluded %.1f s", time() - t0))
    c
end
const u_def = Array(D.CROS(RX.ConcreteRArray(copy(D.UBASE)), D.THC,
                           RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(D.DT0C)))
const u_exc = Array(CEXC(RX.ConcreteRArray(copy(D.UBASE)), D.THC,
                         RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(D.DT0C)))
const SAME = u_def == u_exc
say(@sprintf("  bit-for-bit equal: %s", SAME))
if !SAME
    say(@sprintf("  max abs diff %.3e   max rel diff %.3e",
                 maximum(abs.(u_def .- u_exc)),
                 maximum(abs.(u_def .- u_exc) ./ max.(abs.(u_def), 1e-30))))
end

say("\n==== SUMMARY ====")
@printf("  default : big concat %3d (%6.0f MB)  DUS %4d (big %3d)\n",
        BASE.nbigc, BASE.bigc * 8 / 1e6, BASE.nd, BASE.nbigd)
@printf("  excluded: big concat %3d (%6.0f MB)  DUS %4d (big %3d)\n",
        WR.nbigc, WR.bigc * 8 / 1e6, WR.nd, WR.nbigd)
say("  excluded_passes = [" * join(map(p -> "\"$p\"", WINNER), ", ") * "]")
say("  rounds used: $(NROUNDS[])")
say("P4_BISECT_DONE")
