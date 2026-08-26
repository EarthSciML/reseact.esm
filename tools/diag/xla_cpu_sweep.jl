#!/usr/bin/env julia
# ===========================================================================
# xla_cpu_sweep.jl -- WHY is one chemistry step 256 ms when it is 8,920 ops?
# ===========================================================================
# MEASURED FIRST (slurm 10104808 / 10104837, CONUS 13x7x72, NC=6552):
#   * one ros_step  = 256.1 ms of XLA execution, 0.4% host transfer
#   * one chem RHS  =  51.6 ms
#   * ros_step optimized StableHLO = 8,920 ops (107,882 unoptimized: XLA's own
#     CSE already collapses the emitter's verbosity 12x, so the op COUNT is not
#     the problem)
#   => 256.1 ms / (8,920 ops x 6,552 cells) = 4.4 ns per element-op ~ 13 cycles.
# A fused, vectorised loop should be well under one cycle per element-op, so
# there is roughly a 30-100x execution gap and it is entirely inside XLA:CPU.
#
# This sweep asks two questions the timings above cannot answer:
#
#   1. IS THE THREAD POOL DOING ANYTHING? Measured as CPU time / wall time
#      across the timed region (CLOCK_PROCESS_CPUTIME_ID). ~1 means the step
#      runs on one core and the other 15 are idle -- which would be the single
#      cheapest speedup available. ~16 means it is already parallel and the
#      per-core work itself is the problem.
#   2. HOW MUCH OF IT IS THE BLOCKER-4 WORKAROUND? `xla_cpu_prefer_vector_width
#      =128` halves the SIMD width the fusion emitters may use. It was recorded
#      as costing nothing, but that was measured on the jittered base point
#      where it also removed spurious rejected steps; the traced 24 h run put it
#      at 6.4% SLOWER. Here it is priced directly, against 256/512 and against
#      turning the fusion emitters off entirely.
#
# Every variant is checked to return the SAME unew as the baseline, so a
# "faster" flag that quietly changed the arithmetic cannot pass unnoticed.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default CONUS 13x7x72)
#   RESEACT_SWEEP_REPS       timed repetitions per variant (default 10)
#   RESEACT_SWEEP_DUMP       dir for XLA's post-optimization HLO dump ("" = off)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "xlasweep"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX
say(s) = (println(s); flush(stdout))

const REPS = parse(Int, get(ENV, "RESEACT_SWEEP_REPS", "10"))
const DUMPDIR = get(ENV, "RESEACT_SWEEP_DUMP", "")

# Process CPU time (all threads). CLOCK_PROCESS_CPUTIME_ID == 2 on Linux.
function cpu_ns()
    ts = Ref((Clong(0), Clong(0)))
    ccall(:clock_gettime, Cint, (Cint, Ptr{Cvoid}), 2, ts)
    t = ts[]
    return Int64(t[1]) * 1_000_000_000 + Int64(t[2])
end

UD = RX.ConcreteRArray(copy(D.UBASE))
TD = RX.ConcreteRNumber(D.T0)
DD = RX.ConcreteRNumber(D.DT0C)

# (label, extra debug options). `sync = true` everywhere so the timing brackets
# the whole execution and not a queued handle.
variants = Any[
    ("base vw=128 (today)", (; xla_cpu_prefer_vector_width = 128)),
    ("vw default",          (;)),
    ("vw=256",              (; xla_cpu_prefer_vector_width = 256)),
    ("vw=512",              (; xla_cpu_prefer_vector_width = 512)),
    ("fusion emitters off", (; xla_cpu_use_fusion_emitters = false)),
]

say("\n" * "="^78)
say(@sprintf("XLA:CPU SWEEP  grid=%sx%sx%s  NS=%d NC=%d N=%d  reps=%d",
             ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"],
             D.NS, D.NC, D.N, REPS))
say(@sprintf("  affinity mask allows %d cpus; julia threads = %d",
             length(filter(!isnothing, [nothing])) + length(Sys.cpu_info()), Threads.nthreads()))
say("="^78)

function run_sweep(variants)
    ref = nothing
    results = Any[]
    for (label, dbg) in variants
        local copts
        try
            copts = RX.CompileOptions(; sync = true, xla_debug_options = dbg)
        catch e
            say(@sprintf("  %-22s  CompileOptions REJECTED: %s", label,
                         first(split(sprint(showerror, e), '\n'))))
            continue
        end
        tc = time()
        c = try
            RX.@compile compile_options=copts D.ros_step(UD, D.THC, TD, DD)
        catch e
            say(@sprintf("  %-22s  COMPILE FAILED: %s", label,
                         first(split(sprint(showerror, e), '\n'))))
            continue
        end
        tcomp = time() - tc
        r = c(UD, D.THC, TD, DD)                       # warm
        out = Array(r[1])
        ref === nothing && (ref = copy(out))
        dmax = maximum(abs.(out .- ref) ./ max.(abs.(ref), 1e-300))
        ts = Float64[]; w0 = time_ns(); c0 = cpu_ns()
        for _ in 1:REPS
            t0 = time_ns(); c(UD, D.THC, TD, DD); push!(ts, (time_ns() - t0) / 1e6)
        end
        wall = (time_ns() - w0) / 1e9; cpu = (cpu_ns() - c0) / 1e9
        med = median(ts)
        push!(results, (label, med, cpu / wall, tcomp, dmax))
        @printf("  %-22s  step %8.2f ms   cpu/wall %5.2f   compile %6.1f s   vs base %.2e\n",
                label, med, cpu / wall, tcomp, dmax)
        flush(stdout)
    end
    return results
end
results = run_sweep(variants)

say("\n" * "-"^78)
if !isempty(results)
    b = results[1][2]
    say("  relative to the current default:")
    for (label, med, par, _, dmax) in results
        @printf("    %-22s %6.2fx   (%.2f ms, cpu/wall %.2f, max rel diff %.1e)\n",
                label, b / med, med, par, dmax)
    end
    say("")
    say(@sprintf("  best step time %.2f ms over %d ops x %d cells = %.2f ns/element-op (%.1f cycles at 3 GHz)",
                 minimum(r[2] for r in results), 8920, D.NC,
                 minimum(r[2] for r in results) * 1e6 / (8920 * D.NC),
                 minimum(r[2] for r in results) * 1e6 / (8920 * D.NC) * 3))
end

# ---- XLA's own view: how many FUSIONS does the step compile to? --------------
# The 8,920 figure is StableHLO, i.e. before XLA's backend fuses anything. What
# actually runs is a list of thunks, and if that list is thousands long then each
# one is a separate parallel-for over 6,552 doubles and the temporaries never
# stay in cache. This is the number that decides whether "make XLA fuse" is even
# the right sentence.
if !isempty(DUMPDIR)
    mkpath(DUMPDIR)
    say("\n---- dumping XLA post-optimization HLO to $DUMPDIR ----")
    try
        copts = RX.CompileOptions(; sync = true,
                    xla_debug_options = (; xla_cpu_prefer_vector_width = 128,
                                           xla_dump_to = DUMPDIR,
                                           xla_dump_hlo_as_text = true))
        RX.@compile compile_options=copts D.ros_step(UD, D.THC, TD, DD)
        for f in sort(readdir(DUMPDIR))
            endswith(f, ".txt") || continue
            txt = read(joinpath(DUMPDIR, f), String)
            nf = length(collect(eachmatch(r"fusion\(", txt)))
            nl = count(==('\n'), txt)
            @printf("    %-58s %7d lines  %5d fusion(\n", f, nl, nf)
        end
    catch e
        say("  dump FAILED: " * first(split(sprint(showerror, e), '\n')))
    end
end
say("\nSWEEP_DONE")
