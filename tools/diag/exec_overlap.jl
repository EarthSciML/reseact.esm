#!/usr/bin/env julia
# ===========================================================================
# exec_overlap.jl -- can independent XLA:CPU executions use the idle cores?
# ===========================================================================
# MEASURED (slurm 10104808 vs 10105170, CONUS, identical build):
#     ros23 step   16 cores 253.4 ms   1 core 362.2 ms   =>  1.43x for 16x cores
#     chem RHS     16 cores  51.6 ms   1 core  83.6 ms   =>  1.62x
# i.e. ~9% parallel efficiency: the step is 8,920 StableHLO ops, each an
# elementwise pass over 6,552 doubles (52 kB). Split 16 ways that is 3.3 kB per
# thread per op, far below the point where a parallel-for pays for itself, so
# XLA runs it essentially on one core.
#
# THE OBVIOUS FIX IS TO CHANGE THE GRANULARITY: chemistry is EXACTLY block
# diagonal, so the domain can be cut into S shards of cells and each shard run
# as its own execution of the same program, concurrently. Each cell's arithmetic
# is untouched (every op in ros23_step is elementwise in the cell index, and
# `blocksolve` is per cell), so the STATE is bit-identical however the cells are
# grouped; only the EEst reduction reassociates, and that can be summed back on
# the host.
#
# Before building that, one thing has to be true: PJRT:CPU must actually overlap
# independent executions rather than serialising them behind a lock. That is
# what this measures -- N concurrent executions of the SAME program on N
# DISTINCT input buffers, N = 1, 2, 4, 8, 12. If throughput rises with N the
# shard plan is real; if wall time rises linearly with N it is not, and the
# fix has to be inside the compiled program instead.
#
#   RESEACT_OVERLAP_REPS   repetitions per concurrency level (default 5)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "overlap"
include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX
say(s) = (println(s); flush(stdout))

const REPS = parse(Int, get(ENV, "RESEACT_OVERLAP_REPS", "5"))

say("\n" * "="^78)
say(@sprintf("EXECUTION OVERLAP  NC=%d  julia threads=%d", D.NC, Threads.nthreads()))
say("="^78)

TD = RX.ConcreteRNumber(D.T0); DD = RX.ConcreteRNumber(D.DT0C)
maxn = min(12, Threads.nthreads())
# distinct input buffers so no two tasks share a device buffer
bufs = [RX.ConcreteRArray(copy(D.UBASE)) for _ in 1:maxn]
D.CROS(bufs[1], D.THC, TD, DD)                       # warm

function run_overlap(maxn, bufs, TD, DD)
    base = 0.0
    for n in (1, 2, 4, 8, 12)
        n > maxn && continue
        ts = Float64[]
        for _ in 1:REPS
            t0 = time_ns()
            @sync for k in 1:n
                Threads.@spawn D.CROS(bufs[k], D.THC, TD, DD)
            end
            push!(ts, (time_ns() - t0) / 1e6)
        end
        w = median(ts)
        n == 1 && (base = w)
        @printf("  %2d concurrent steps: wall %8.2f ms   throughput %5.2f steps/s   speedup vs serial %.2fx\n",
                n, w, 1000n / w, n * base / w)
        flush(stdout)
    end
end
run_overlap(maxn, bufs, TD, DD)

say("")
say("  Read: `speedup vs serial` is what sharding could buy. 1.0x means PJRT")
say("  serialises independent executions and the cores stay idle; N means the")
say("  runtime overlaps them perfectly and a shard-per-core split of the cell")
say("  dimension should recover close to N.")
say("\nOVERLAP_DONE")
