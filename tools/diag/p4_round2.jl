#!/usr/bin/env julia
# ===========================================================================
# p4_round2.jl -- minimize the exclusion set DOWN from exclude-all, keeping
#                 the round-1 winner always excluded.
# ===========================================================================
# Round 1 (p4_pass_bisect.jl, 6x6x8): the ONLY singleton that restores DUS is
# `dynamic_update_to_concat` (concat 79 -> 40, DUS 0 -> 205); excluding all
# 493 patterns reaches concat 8 / DUS 292, so a COOPERATING set produces the
# remaining ~32 concatenates. Round 1's "!! rewrite lives outside this list"
# message was wrong -- its criterion used the big>=400k threshold, which no
# 6x6x8 tensor reaches; counts said otherwise. This probe ddmin-minimizes the
# extra exclusions needed to match the exclude-all census.
#
#   P4_ALWAYS     always-excluded base set (default dynamic_update_to_concat)
#   P4_MAXROUNDS  @code_hlo round cap (default 90)
#   P4_BIGN       "big result" element threshold (default 40000 -- the 6x6x8
#                 extended observed buffer is 57708)
#
# RESULTS: see the P4_WINNER line of the run log.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "p4round2"
include(joinpath(@__DIR__, "_env.jl"))
using Printf
say(s) = (println(s); flush(stdout))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
const BIGN = parse(Int, get(ENV, "P4_BIGN", "40000"))
const MAXROUNDS = parse(Int, get(ENV, "P4_MAXROUNDS", "90"))
const ALWAYS = String.(split(get(ENV, "P4_ALWAYS", "dynamic_update_to_concat"), ','))

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
const UR = RX.ConcreteRArray(copy(D.UBASE))
const NROUNDS = Ref(0)
const CACHE = Dict{Vector{String},NamedTuple}()
function census(excl::Vector{String}; label = "")
    key = sort(excl)
    haskey(CACHE, key) && return CACHE[key]
    NROUNDS[] += 1
    NROUNDS[] > MAXROUNDS && error("P4_MAXROUNDS=$MAXROUNDS exhausted")
    copts = RX.CompileOptions(; sync = true,
        xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
        excluded_passes = collect(String, excl))
    t0 = time()
    txt = sprint(show, RX.@code_hlo(compile_options = copts, D.ros_step(UR, D.THC, D.T_R, D.DTC_R)))
    cc = sizes_for(txt, "stablehlo.concatenate")
    dd = sizes_for(txt, "stablehlo.dynamic_update_slice")
    r = (concat = length(cc), cbig = count(>=(BIGN), cc), cmb = sum(cc; init = 0) * 8 / 1e6,
         dus = length(dd), dbig = count(>=(BIGN), dd))
    CACHE[key] = r
    say(@sprintf("  [%3d] excl=%-3d %-28s %3.0fs  concat %4d (big %3d, %7.1f MB)  DUS %4d (big %3d)",
                 NROUNDS[], length(excl), label, time() - t0, r.concat, r.cbig, r.cmb, r.dus, r.dbig))
    return r
end

const base = census(ALWAYS; label = "always-only")
const extras0 = setdiff(ALLPATS, ALWAYS)
const full = census(vcat(ALWAYS, extras0); label = "exclude-all")
pred(r) = r.concat <= full.concat + 2 && r.dus >= full.dus - 5

function minimize(extras0)
    # soft-scope trap: all mutation lives inside this function on purpose
    work = copy(extras0)
    n = 2
    try
        while true
            chunksz = max(1, length(work) ÷ n)
            reduced = false
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
    say("P4_MINSET n=$(length(work)): " * join(work, ","))
    for p in copy(work)   # necessity check of each survivor
        r = census(vcat(ALWAYS, setdiff(work, [p])); label = "need? -$p")
        pred(r) && (work = setdiff(work, [p]))
    end
    return work
end
const FINAL = minimize(extras0)
say("P4_WINNER " * join(vcat(ALWAYS, FINAL), ","))
say("P4_ROUND2_DONE rounds=$(NROUNDS[])")
