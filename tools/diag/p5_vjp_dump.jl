#!/usr/bin/env julia
# ===========================================================================
# p5_vjp_dump.jl -- what does XLA:CPU actually EXECUTE for the VJP, versus the
#                   primal step? (post-optimization HLO census)
# ===========================================================================
# p5_vjp_census.jl looked at the StableHLO handed to XLA and found the VJPs
# already free of whole-buffer concatenates. p5_vjp_time.jl then measured the
# VJPs at 4.4x (chemistry) and 9x (transport) the cost of their primal steps at
# 6x6x8 for 1.6-1.7x the ops, with MORE cores busy (cpu/wall 4.9 vs 3.3), i.e.
# the extra cost is work, not idle cores. XLA's own passes can add work the
# StableHLO does not show: copy-insertion materialises `copy` instructions
# where a buffer is updated while still live, fusion decides how many kernels
# run and of what kind, and scatter/gather/dot stay outside fusions. This probe
# compiles each program with `xla_dump_to` and censuses the
# `*after_optimizations.txt` module: fusion count by kind, every non-fused
# instruction by opcode with its result shape, and `copy` shapes in particular.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 6 6 8)
#   P5_PROGRAMS              default ros_step,ros_vjp,ssp_step,ssp_vjp
#   P5_BIGN                  big-result threshold in elements (default 40000)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p5dump")
include(joinpath(@__DIR__, "_env.jl"))
using Printf
say(s) = (println(s); flush(stdout))
const PROGS = String.(split(get(ENV, "P5_PROGRAMS", "ros_step,ros_vjp,ssp_step,ssp_vjp"), ','))
const BIGN = parse(Int, get(ENV, "P5_BIGN", "40000"))
const GRID = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT = joinpath(REPO, "logs", "p5-dump-$GRID-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
say("P5_VJP_DUMP grid=$GRID out=$OUT")

function copts_dump(dir)
    RX.CompileOptions(; sync = true,
        xla_debug_options = (; xla_cpu_prefer_vector_width = 128,
                               xla_dump_to = dir, xla_dump_hlo_as_text = true),
        (isempty(D.EXCLP) ? (;) : (; excluded_passes = collect(String, D.EXCLP)))...)
end
function compile_prog(prog, copts)
    prog == "ros_step" && return RX.@compile compile_options = copts D.ros_step(D.U_R, D.THC, D.T_R, D.DTC_R)
    prog == "ssp_step" && return RX.@compile compile_options = copts D.ssp_step(D.U_R, D.THT, D.T_R, D.DTT_R)
    prog == "ros_vjp"  && return RX.@compile compile_options = copts D.ros_vjp(D.U_R, D.THC, D.LAM_R, D.T_R, D.DTC_R)
    prog == "ssp_vjp"  && return RX.@compile compile_options = copts D.ssp_vjp(D.U_R, D.THT, D.LAM_R, D.T_R, D.DTT_R)
    error("unknown program $prog")
end

shape_elems(s) = (m = match(r"f64\[([0-9,]*)\]", s); m === nothing ? 0 :
                  (isempty(m.captures[1]) ? 1 : prod(parse.(Int, split(m.captures[1], ',')))))

# Census of the ENTRY computation of an after-optimizations HLO text: every
# instruction line `%name = TYPE opcode(...)`, with fusion kind for fusions.
function census_hlo(txt)
    lines = split(txt, '\n')
    # ENTRY computation spans from "ENTRY" to the next blank line
    i0 = findfirst(l -> startswith(l, "ENTRY"), lines)
    i0 === nothing && return nothing
    i1 = findnext(l -> isempty(strip(l)), lines, i0)
    i1 === nothing && (i1 = length(lines))
    ops = Dict{String,Vector{Int}}()
    fkind = Dict{String,Int}()
    for l in lines[(i0 + 1):(i1 - 1)]
        m = match(r"^\s*(?:ROOT\s+)?%[\w.\-]+\s*=\s*(\S+)\s+([a-z\-]+)\(", l)
        m === nothing && continue
        ty, op = m.captures
        push!(get!(ops, op, Int[]), shape_elems(ty))
        if op == "fusion"
            k = match(r"kind=(k\w+)", l)
            key = k === nothing ? "?" : k.captures[1]
            fkind[key] = get(fkind, key, 0) + 1
        end
    end
    return ops, fkind, i1 - i0 - 1
end

for prog in PROGS
    dir = joinpath(OUT, prog); mkpath(dir)
    t0 = time()
    try
        compile_prog(prog, copts_dump(dir))
    catch e
        say("  $prog compile FAILED: " * first(split(sprint(showerror, e), '\n'))); continue
    end
    say(@sprintf("\n---- %s  (@compile %.0f s) ----", prog, time() - t0))
    files = filter(f -> occursin("after_optimizations.txt", f), readdir(dir))
    isempty(files) && (say("  no after_optimizations dump found in $dir"); continue)
    # the largest module is the program itself (helpers are tiny)
    f = files[argmax([filesize(joinpath(dir, f)) for f in files])]
    txt = read(joinpath(dir, f), String)
    r = census_hlo(txt)
    r === nothing && (say("  no ENTRY computation in $f"); continue)
    ops, fkind, n = r
    say(@sprintf("  %s: %d ENTRY instructions", f, n))
    say("  fusions by kind: " * join(["$k=$v" for (k, v) in sort(collect(fkind))], "  "))
    say("  non-fused instructions by opcode (n, total elems, big n):")
    for (op, v) in sort(collect(ops); by = kv -> -length(kv[2]))
        op in ("fusion", "parameter", "constant", "get-tuple-element", "tuple", "bitcast") && continue
        say(@sprintf("      %-26s n %5d   elems %9.3e   big %4d", op, length(v), Float64(sum(v)), count(>=(BIGN), v)))
    end
    cp = get(ops, "copy", Int[])
    say(@sprintf("  COPY instructions: n %d, elems %.3e (%.1f MB), big %d", length(cp), Float64(sum(cp; init = 0)),
                 sum(cp; init = 0) * 8e-6, count(>=(BIGN), cp)))
    say(@sprintf("  fusion count %d, non-fused %d", sum(values(fkind); init = 0),
                 sum(length(v) for (k, v) in ops if !(k in ("fusion", "parameter", "constant", "get-tuple-element", "tuple", "bitcast")); init = 0)))
end
say("P5_DUMP_DONE")
