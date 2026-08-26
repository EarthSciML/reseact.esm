#!/usr/bin/env julia
# ===========================================================================
# bufcopy_vs_arith.jl -- is the ROS23 step bound by EXTENDED-BUFFER COPIES
#                        rather than by chemistry arithmetic?
# ===========================================================================
# Derived from the CONUS HLO dump logs/hlodump-10105946:
#
#   * the module's ENTRY computation materialises 155 buffers of >= 500k
#     elements: 47 concatenate, 35 scatter, 21 slice+concat, 21 slice, 4 pad,
#     12 compare/select, ...  = 231.8 M elements = 1855 MB WRITTEN per step.
#   * the extended ("observed") state is f64[1310641] = 200.04 x NC, i.e. 13
#     species + ~187 observed quantities per cell, vs a real state of 85176.
#   * only 40.9 M element-ops of the step's 1005 M are lane-sized arithmetic
#     (multiply/add/subtract/divide on f64[6552]).  4.1%.
#
# So the prediction is: a program that does ONLY the whole-buffer data movement,
# with NO chemistry at all, should already cost most of the 253.4 ms step; and a
# program that does ONLY the step's real arithmetic should be nearly free.
#
# Both arms run IN ONE PROCESS (the only comparison that is trustworthy here).
#
#   FALSIFIED if the buffer arm is fast (say < 40 ms): then whole-buffer
#   materialisation is not what the step is spending its time on.
# ===========================================================================
# STATUS 2026-08-24: DID NOT COMPLETE. Killed after 24 min wall in @compile with
# no output -- 47 chained whole-buffer `vcat`s on a f64[1310641] appear to put
# XLA:CPU into a pathological compile. Kept because that is itself a (weak) data
# point for the diagnosis, and because the arm-B side is reusable. If you revive
# it, cut N_CONCAT to ~5 first and DO NOT run it on a login node.
#
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV",
                 normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io = devnull)
using Reactant, Printf, Statistics
const RX = Reactant

say(s) = (println(s); flush(stdout))

const NC   = 6552            # CONUS cells
const NEXT = 1310641         # extended/observed buffer, from the dump
const NST  = 13 * NC         # 85176, the real state

# op counts lifted from the ENTRY histogram of the dump
const N_CONCAT  = 47
const N_SCATTER = 35
const N_SLCAT   = 21
const N_SLICE   = 21
const N_PAD     = 4

# ---- arm A: whole-buffer data movement only, no arithmetic -----------------
# Mirrors _oop_scatter (du[out] = res) and _oop_prefix_copy (vcat(u, ue[n+1:end])).
function _setidx(b, idx, vals)
    b[idx] = vals
    return b
end

function buffer_traffic2(ue, u, vals)
    b = ue
    for _ in 1:N_CONCAT
        b = vcat(u, b[(NST + 1):NEXT])
    end
    idx = collect(1:NC)
    for _ in 1:N_SCATTER
        b = _setidx(b, idx, vals)
    end
    for _ in 1:N_SLCAT
        b = vcat(b[1:NC], vals, b[(2NC + 1):NEXT])
    end
    acc = b[1:NC]
    for k in 1:N_SLICE
        o = (k - 1) * NC
        acc = acc .+ b[(o + 1):(o + NC)]
    end
    for _ in 1:N_PAD
        b = vcat(b, zeros(eltype(b), 91))
        b = b[1:NEXT]
    end
    return b[1:NC] .+ acc
end

# ---- arm B: the step's real lane arithmetic only ---------------------------
# 40.9 M element-ops of multiply/add/subtract/divide on f64[6552] lanes.
# 40.9e6 / 6552 = 6242 ops.  Emit them as 13 interacting lanes, like blocksolve.
const N_ARITH_OPS = 6242
function lane_arith(u, vals)
    ls = [u[((s - 1) * NC + 1):(s * NC)] for s in 1:13]
    n = 0
    while n < N_ARITH_OPS
        for i in 1:13
            j = (i % 13) + 1
            m = ls[i] .* vals
            ls[j] = ls[j] .- m
            n += 2
            n >= N_ARITH_OPS && break
        end
    end
    r = ls[1]
    for i in 2:13
        r = r .+ ls[i]
    end
    return r
end

function timeit(f, args...; reps = 5)
    f(args...)                                   # warm
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        r = f(args...)
        RX.synchronize(r)
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return minimum(ts), median(ts)
end

say("\n" * "="^78)
say(@sprintf("BUFFER TRAFFIC vs LANE ARITHMETIC   NC=%d  NEXT=%d  threads=%d",
             NC, NEXT, Threads.nthreads()))
say("="^78)

ue0   = RX.ConcreteRArray(rand(NEXT))
u0    = RX.ConcreteRArray(rand(NST))
vals0 = RX.ConcreteRArray(rand(NC))

say("compiling arm A (buffer traffic, no arithmetic) ...")
tA = @elapsed cA = @compile buffer_traffic2(ue0, u0, vals0)
say(@sprintf("  compiled in %.1f s", tA))

say("compiling arm B (lane arithmetic, no buffers) ...")
tB = @elapsed cB = @compile lane_arith(u0, vals0)
say(@sprintf("  compiled in %.1f s", tB))

mA, dA = timeit(cA, ue0, u0, vals0)
mB, dB = timeit(cB, u0, vals0)

say("")
say(@sprintf("arm A  buffer traffic only : min %8.2f ms   median %8.2f ms", mA, dA))
say(@sprintf("arm B  lane arithmetic only: min %8.2f ms   median %8.2f ms", mB, dB))
say("")
say(@sprintf("measured full ROS23 step (slurm 10104808)      : %8.2f ms", 253.40))
say(@sprintf("arm A as fraction of the full step             : %8.1f %%", 100 * mA / 253.4))
say(@sprintf("arm B as fraction of the full step             : %8.1f %%", 100 * mB / 253.4))
say(@sprintf("A / B                                          : %8.1f x", mA / mB))
bytes = 231.8e6 * 8 * 2 / 1e9
say(@sprintf("implied bandwidth if A is copy-bound: %.2f GB / %.3f s = %.1f GB/s",
             bytes, mA / 1e3, bytes / (mA / 1e3)))
say("="^78)
