#!/usr/bin/env julia
# ===========================================================================
# p5_vjp_census.jl -- WHAT does the optimized VJP module carry that the
#                     primal step no longer does?
# ===========================================================================
# MOTIVATION (slurm 10359755, 48 h CONUS, 2026-09-04): with the driver defaults
# ESS_OOP_SSA=1 + RESEACT_EXCLUDED_PASSES=dynamic_update_to_concat,sub_const_prop
# the forward pass sped up 5.2x and the replay 5.5x against August, but the
# VJPs only 2.5x (0.4585 -> 0.1847 s/VJP) and are now 67% of wall. `COPTS` is
# shared by every compile site, so ros_vjp/ssp_vjp ALREADY compile with that
# exclusion. The question is therefore empirical: does the optimized VJP
# module still carry whole-buffer concatenates / pads / copies that the primal
# lost, and if so are they produced by a pattern (bisectable) or by Enzyme's
# reverse-pass codegen (not)?
#
# This probe censuses FOUR optimized modules in ONE process at 6x6x8:
#   ros_step, ros_vjp, ssp_step, ssp_vjp
# each under (a) the stock pipeline (no exclusions) and (b) the current driver
# default, and prints per program: op count, an op histogram weighted by result
# ELEMENTS (so a single 57,708-element concatenate outweighs a thousand
# scalars), and the whole-buffer families -- concatenate, dynamic_update_slice,
# pad, slice, copy, gather, select -- with counts and "big" (>= P5_BIGN
# elements) subsets. `stablehlo.while`, `func.call`, `enzyme.*` counts are
# printed too: a reverse pass that materialises a tape or recomputes the primal
# shows up there, not in the concat count.
#
# Every CompileOptions keeps sync=true and xla_cpu_prefer_vector_width=128 (the
# XLA:CPU race workaround, REQUIRED); exclusions are the only change.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 6 6 8; CONUS goes through sbatch)
#   P5_BIGN                  "big result" element threshold (default 40000;
#                            the 6x6x8 extended observed buffer is 57708)
#   P5_PROGRAMS              comma list of ros_step,ros_vjp,ssp_step,ssp_vjp
#   P5_ARMS                  comma list of stock,default,<extra exclusion set
#                            written as a+b+c> (default "stock,default")
#
# RESULTS (grid 6x6x8, Reactant v0.2.280, 2026-09-04, logs/p5-census-local.out;
# big = >= 40,000 elements, the 6x6x8 extended buffer is 57,708):
#
#   program   arm      ops    big concat     big DUS       big pad    scatter
#   ros_step  stock    8555   4 (2.1 MB)     0             4 (2.2 MB)    7
#   ros_step  default  8821   0              8 (4.4 MB)    4 (2.2 MB)    7
#   ros_vjp   stock   14115   3 (1.7 MB)     0             3 (1.7 MB)   61
#   ros_vjp   default 14342   0             12 (6.9 MB)    3 (1.7 MB)   61
#   ssp_step  default 12062   0              0             0            12
#   ssp_vjp   default 32517   0              0             0           628
#
# VERDICT: THE VJPs ARE ALREADY DUS-CLEAN under the driver default. The
# exclusion set that fixed the primal reaches the VJP module too (the concat
# family goes to 0 big in both), and the only big whole-buffer writes left in
# ros_vjp are the SAME three pads the primal carries. There is nothing here
# for a second pass bisect to remove, so p5_vjp_bisect.jl was NOT run.
#
# What the VJP carries that the primal does not: 61 scatters against 7 (the
# reverse of the chemistry plan's gathers), 72 reduces against 0, 78 dots
# against 13, and 3.5x the `add` result-elements (6.06e6 vs 1.71e6: adjoint
# accumulation). ssp_vjp: 628 scatters against 12 (the reverse of 619
# transport gathers). Per-program timings: p5_vjp_time.jl.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p5census")
include(joinpath(@__DIR__, "_env.jl"))
using Printf
say(s) = (println(s); flush(stdout))

const GRID = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT = joinpath(REPO, "logs", "p5-census-$GRID-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)
say("P5_VJP_CENSUS grid=$GRID out=$OUT")
const tb = time()
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
say(@sprintf("driver loaded in %.1f s  (driver default excluded_passes = %s)",
             time() - tb, isempty(D.EXCLP) ? "<none>" : join(D.EXCLP, ",")))

const BIGN = parse(Int, get(ENV, "P5_BIGN", "40000"))
const PROGS = String.(split(get(ENV, "P5_PROGRAMS", "ros_step,ros_vjp,ssp_step,ssp_vjp"), ','))
const ARMS = String.(split(get(ENV, "P5_ARMS", "stock,default"), ','))

copts_excl(excl) = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
    (isempty(excl) ? (;) : (; excluded_passes = collect(String, excl)))...)
arm_excl(a) = a == "stock" ? String[] : a == "default" ? collect(String, D.EXCLP) :
              String.(split(a, '+'))

const UR = RX.ConcreteRArray(copy(D.UBASE))
const LR = RX.ConcreteRArray(copy(D.WOBJ))
function hlo_of(prog, copts)
    prog == "ros_step" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ros_step(UR, D.THC, D.T_R, D.DTC_R)))
    prog == "ssp_step" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ssp_step(UR, D.THT, D.T_R, D.DTT_R)))
    prog == "ros_vjp" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ros_vjp(UR, D.THC, LR, D.T_R, D.DTC_R)))
    prog == "ssp_vjp" && return sprint(show, RX.@code_hlo(compile_options = copts,
        D.ssp_vjp(UR, D.THT, LR, D.T_R, D.DTT_R)))
    error("unknown program $prog")
end

# One op per line `%x = dialect.op(...) ... -> tensor<AxBxf64>` (or `: tensor<...>`).
# Returns Dict op => Vector of result element counts (0 when no tensor result).
function census(txt)
    h = Dict{String,Vector{Int}}()
    for ln in split(txt, '\n')
        m = match(r"=\s+\"?([a-zA-Z_][\w]*\.[\w.]+)\"?", ln)
        m === nothing && continue
        op = m.captures[1]
        n = 0
        s = match(r"->\s*tensor<([0-9x]+)x[a-z]", ln)
        s === nothing && (s = match(r":\s*tensor<([0-9x]+)x[a-z]", ln))
        if s !== nothing
            n = 1
            for d in split(s.captures[1], 'x'); n *= parse(Int, d); end
        end
        push!(get!(h, op, Int[]), n)
    end
    return h
end
const FAMILIES = ["stablehlo.concatenate", "stablehlo.dynamic_update_slice", "stablehlo.pad",
                  "stablehlo.slice", "stablehlo.copy", "stablehlo.gather", "stablehlo.select",
                  "stablehlo.while", "stablehlo.dynamic_slice", "func.call", "stablehlo.custom_call"]

function report(prog, arm, txt)
    h = census(txt)
    tot = sum(length, values(h); init = 0)
    elems = sum(sum, values(h); init = 0)
    say(@sprintf("  %-8s %-8s  ops %6d   result-elements %10.3e   lines %d",
                 prog, arm, tot, elems, count(==('\n'), txt)))
    for f in FAMILIES
        v = get(h, f, Int[])
        isempty(v) && continue
        big = filter(>=(BIGN), v)
        say(@sprintf("      %-32s n %5d  elems %9.3e   big n %4d  big elems %9.3e",
                     f, length(v), Float64(sum(v)), length(big), Float64(sum(big; init = 0))))
    end
    enz = sum(length(v) for (k, v) in h if startswith(k, "enzyme"); init = 0)
    enz > 0 && say(@sprintf("      %-32s n %5d", "enzyme.* (any)", enz))
    say("      top ops by result elements:")
    for (k, v) in sort(collect(h); by = kv -> -sum(kv[2]))[1:min(12, length(h))]
        say(@sprintf("        %-34s n %5d  elems %9.3e", k, length(v), Float64(sum(v))))
    end
    open(joinpath(OUT, "$prog.$arm.mlir"), "w") do io; write(io, txt); end
    return h
end

const RESULTS = Dict{Tuple{String,String},Dict{String,Vector{Int}}}()
for prog in PROGS, arm in ARMS
    t0 = time()
    txt = try
        hlo_of(prog, copts_excl(arm_excl(arm)))
    catch e
        say(@sprintf("  %-8s %-8s FAILED: %s", prog, arm, first(split(sprint(showerror, e), '\n'))))
        continue
    end
    say(@sprintf("\n---- %s / %s   (@code_hlo %.0f s) ----", prog, arm, time() - t0))
    RESULTS[(prog, arm)] = report(prog, arm, txt)
end

say("\n==== SUMMARY: big whole-buffer writes (>= $BIGN elems), per program and arm ====")
for prog in PROGS, arm in ARMS
    haskey(RESULTS, (prog, arm)) || continue
    h = RESULTS[(prog, arm)]
    f(op) = (v = filter(>=(BIGN), get(h, op, Int[])); (length(v), sum(v; init = 0)))
    c = f("stablehlo.concatenate"); d = f("stablehlo.dynamic_update_slice"); p = f("stablehlo.pad")
    s = f("stablehlo.select"); g = f("stablehlo.gather")
    say(@sprintf("  %-8s %-8s  concat %3d (%6.1f MB)  DUS %3d (%6.1f MB)  pad %3d (%6.1f MB)  select %3d (%6.1f MB)  gather %3d (%6.1f MB)",
                 prog, arm, c[1], c[2] * 8e-6, d[1], d[2] * 8e-6, p[1], p[2] * 8e-6,
                 s[1], s[2] * 8e-6, g[1], g[2] * 8e-6))
end
say("P5_CENSUS_DONE")
