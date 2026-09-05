#!/usr/bin/env julia
# ===========================================================================
# p5_vjp_bisect.jl -- minimal EXTRA `excluded_passes` set that brings the
#                     VJP module's whole-buffer traffic down to its floor.
# ===========================================================================
# Companion to p5_vjp_census.jl, which measures WHAT the optimized VJP module
# carries. This one asks, for ONE program (default ros_vjp): starting from the
# driver's default exclusion set (always held), which additional default
# Enzyme-JAX patterns are responsible for the big whole-buffer writes that
# remain -- if any pattern is. Method is p4_round2's: census the FLOOR by
# excluding EVERY generated pattern, then ddmin the extra exclusions down from
# exclude-all with the driver default held, then necessity-check each survivor.
#
# The metric is the number of "big" (>= P5_BIGN elements) results among
# concatenate + pad + copy, plus their element total. If exclude-all does NOT
# lower it, the traffic is not produced by a pattern (Enzyme's reverse-pass
# codegen, or XLA itself) and `excluded_passes` is not the knob -- the probe
# says so and exits 2.
#
# Every CompileOptions keeps sync=true and xla_cpu_prefer_vector_width=128.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 6 6 8; do NOT run CONUS here)
#   P5_PROGRAM               ros_vjp | ssp_vjp | ros_step | ssp_step
#   P5_BIGN                  big-result threshold (default 40000)
#   P5_MAXROUNDS             @code_hlo round cap (default 90)
#   P5_SLACK                 allowed big-count excess over the floor (default 2)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p5bisect")
include(joinpath(@__DIR__, "_env.jl"))
using Printf
say(s) = (println(s); flush(stdout))
const PROG = get(ENV, "P5_PROGRAM", "ros_vjp")
const GRID = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT = joinpath(REPO, "logs", "p5-bisect-$PROG-$GRID-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)
say("P5_VJP_BISECT program=$PROG grid=$GRID out=$OUT")
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
const BIGN = parse(Int, get(ENV, "P5_BIGN", "40000"))
const MAXROUNDS = parse(Int, get(ENV, "P5_MAXROUNDS", "90"))
const SLACK = parse(Int, get(ENV, "P5_SLACK", "2"))
const ALWAYS = collect(String, D.EXCLP)
say("always-excluded (driver default): " * (isempty(ALWAYS) ? "<none>" : join(ALWAYS, ",")))

basename_of(p) = String(first(split(p, '<')))
function pattern_names()
    names = String[]
    for sroa in (true, false)
        s = RX.Compiler.optimization_passes(D.COPTS; sroa, backend = "cpu")
        for m in eachmatch(r"patterns=([^}]*)\}", s), p in split(m.captures[1], ';')
            isempty(strip(p)) && continue
            b = basename_of(strip(p)); b in names || push!(names, b)
        end
    end
    return names
end
const ALLPATS = pattern_names()
open(joinpath(OUT, "patterns_default.txt"), "w") do io; foreach(p -> println(io, p), ALLPATS); end
say("default pattern list: $(length(ALLPATS)) unique base names")

copts_excl(excl) = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
    (isempty(excl) ? (;) : (; excluded_passes = collect(String, excl)))...)
const UR = RX.ConcreteRArray(copy(D.UBASE))
const LR = RX.ConcreteRArray(copy(D.WOBJ))
function hlo_of(copts)
    PROG == "ros_step" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ros_step(UR, D.THC, D.T_R, D.DTC_R)))
    PROG == "ssp_step" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ssp_step(UR, D.THT, D.T_R, D.DTT_R)))
    PROG == "ros_vjp" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ros_vjp(UR, D.THC, LR, D.T_R, D.DTC_R)))
    PROG == "ssp_vjp" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ssp_vjp(UR, D.THT, LR, D.T_R, D.DTT_R)))
    error("unknown program $PROG")
end
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
const NROUNDS = Ref(0)
const CACHE = Dict{Vector{String},NamedTuple}()
function census(excl::Vector{String}; label = "", save = "")
    key = sort(unique(excl))
    save == "" && haskey(CACHE, key) && return CACHE[key]
    NROUNDS[] += 1
    NROUNDS[] > MAXROUNDS && error("P5_MAXROUNDS=$MAXROUNDS exhausted")
    t0 = time()
    txt = try
        hlo_of(copts_excl(key))
    catch e
        say(@sprintf("  [%3d] excl=%-3d %-30s FAILED: %s", NROUNDS[], length(key), label,
                     first(split(sprint(showerror, e), '\n'))))
        r = (ok = false, big = typemax(Int) ÷ 4, bigel = 0, concat = -1, dus = -1, pad = -1)
        CACHE[key] = r; return r
    end
    cc = sizes_for(txt, "stablehlo.concatenate"); pp = sizes_for(txt, "stablehlo.pad")
    cp = sizes_for(txt, "stablehlo.copy"); dd = sizes_for(txt, "stablehlo.dynamic_update_slice")
    bigv = filter(>=(BIGN), vcat(cc, pp, cp))
    r = (ok = true, big = length(bigv), bigel = sum(bigv; init = 0),
         concat = length(cc), dus = length(dd), pad = length(pp))
    CACHE[key] = r
    say(@sprintf("  [%3d] excl=%-3d %-30s %4.0fs  big(concat+pad+copy) %4d (%7.1f MB)   concat %4d  DUS %4d  pad %4d",
                 NROUNDS[], length(key), label, time() - t0, r.big, r.bigel * 8e-6, r.concat, r.dus, r.pad))
    save == "" || open(joinpath(OUT, save), "w") do io; write(io, txt); end
    return r
end

say("\n---- baseline: driver default ----")
const BASE = census(ALWAYS; label = "driver-default", save = "$PROG.default.mlir")
say("\n---- floor: exclude EVERY generated pattern ----")
const extras0 = setdiff(ALLPATS, ALWAYS)
const FULL = census(vcat(ALWAYS, extras0); label = "exclude-all", save = "$PROG.exclude-all.mlir")
if !FULL.ok || FULL.big >= BASE.big
    say(@sprintf("\nP5_VERDICT NOT-A-PATTERN: exclude-all leaves big %d vs default %d -- " *
                 "the whole-buffer traffic in %s is not produced by a default pattern; " *
                 "excluded_passes is not the knob.", FULL.big, BASE.big, PROG))
    exit(2)
end
say(@sprintf("\nexclude-all lowers big writes %d -> %d (%.1f -> %.1f MB): a pattern set is responsible",
             BASE.big, FULL.big, BASE.bigel * 8e-6, FULL.bigel * 8e-6))
pred(r) = r.ok && r.big <= FULL.big + SLACK

say("\n---- singleton name-candidates ----")
const CANDS = filter(p -> occursin(r"concat|dus|dynamic_update|slice|pad|copy|transpose|reshape|broadcast"i, p), extras0)
single = nothing
for c in CANDS
    if pred(census(vcat(ALWAYS, [c]); label = c))
        global single = c; break
    end
end
single === nothing || say("  singleton winner: $single")

function minimize(work0)
    work = copy(work0); n = 2
    try
        while true
            chunksz = max(1, length(work) ÷ n); reduced = false
            for i in 1:n
                lo = (i - 1) * chunksz + 1; hi = i == n ? length(work) : i * chunksz
                lo > length(work) && break
                cand = vcat(work[1:lo-1], work[hi+1:end])
                if pred(census(vcat(ALWAYS, cand); label = "drop $(lo):$(hi)"))
                    work = cand; n = max(n - 1, 2); reduced = true; break
                end
            end
            reduced && continue
            n >= length(work) && break
            n = min(2n, length(work))
        end
    catch e
        say("!! stopped early: $(sprint(showerror, e))")
    end
    say("P5_MINSET n=$(length(work)): " * join(work, ","))
    for p in copy(work)
        r = census(vcat(ALWAYS, setdiff(work, [p])); label = "need? -$p")
        pred(r) && (work = setdiff(work, [p]))
    end
    return work
end
const FINAL = single === nothing ? minimize(extras0) : [single]
const WR = census(vcat(ALWAYS, FINAL); label = "winner", save = "$PROG.winner.mlir")
open(joinpath(OUT, "winner.txt"), "w") do io; foreach(p -> println(io, p), FINAL); end
say("\nP5_WINNER extra=" * join(FINAL, ",") * "   full=" * join(vcat(ALWAYS, FINAL), ","))
say(@sprintf("  default big %d (%.1f MB) -> winner big %d (%.1f MB); floor %d",
             BASE.big, BASE.bigel * 8e-6, WR.big, WR.bigel * 8e-6, FULL.big))
say("P5_BISECT_DONE rounds=$(NROUNDS[])")
