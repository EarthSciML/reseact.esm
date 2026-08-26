#!/usr/bin/env julia
# ===========================================================================
# host_device_overlap.jl -- can XLA:CPU be made to use the cores it is given?
# ===========================================================================
# MEASURED on the real model (slurm 10105948, CONUS): 8 concurrent executions of
# `ros_step` on the single default CPU device gave 1.34x throughput, i.e. PJRT
# does not overlap them, and a single execution keeps only ~3 of 16 cores busy
# (cpu/wall 3.0, and 1 core vs 16 cores is 362 ms vs 253 ms). So the cores are
# there and nothing is using them.
#
# `XLA_FLAGS=--xla_force_host_platform_device_count=N` splits the CPU platform
# into N addressable devices. This is a MODEL-FREE probe of whether executions
# then genuinely overlap -- it needs no build, so it can be iterated in minutes
# rather than the ~22 min a CONUS build costs. The body is a long elementwise
# chain over a 6,552-element vector, deliberately the same shape and the same
# measured cost per element-op (~1.3 cycles) as one fusion of the real step.
#
# It matters because chemistry is EXACTLY block diagonal: if N devices overlap,
# the cell dimension can be sharded N ways with the per-cell arithmetic
# bit-identical (only the EEst reduction reassociates, and that can be summed
# on the host).
#
#   NDEV   host devices to request (default 8); also set XLA_FLAGS to match
#   NVEC   vector length (default 6552, one CONUS cell block)
# ===========================================================================
import Pkg; Pkg.activate(ARGS[1]; io=devnull)
using Reactant, Printf, Statistics
const RX = Reactant
NDEV = parse(Int, get(ENV, "NDEV", "8"))
try
    RX.XLA.update_global_state!(; xla_force_host_platform_device_count = NDEV)
catch e
    @info "update_global_state! rejected" e
end
RX.set_default_backend("cpu")
devs = RX.devices()
@printf("devices visible: %d\n", length(devs))

# a chain of elementwise ops over a 6552-element vector, the shape of one fusion
f(x) = begin
    y = x
    for _ in 1:1500; y = @. y * 1.0000001 + 0.5; y = @. sqrt(abs(y)) + y * 0.5; end
    y
end
n = parse(Int, get(ENV, "NVEC", "6552"))
nd = min(NDEV, length(devs))
xs = [RX.ConcreteRArray(rand(n); device = devs[i]) for i in 1:nd]
cs = [RX.@compile(compile_options=RX.CompileOptions(; sync=true), f(xs[i])) for i in 1:nd]
for i in 1:nd; cs[i](xs[i]); end
function timeit(k)
    ts = Float64[]
    for _ in 1:8
        t0 = time_ns()
        @sync for i in 1:k; Threads.@spawn cs[i](xs[i]); end
        push!(ts, (time_ns()-t0)/1e6)
    end
    median(ts)
end
b = timeit(1)
for k in (1, 2, 4, 8)
    k > nd && continue
    w = timeit(k)
    @printf("  %2d concurrent on %2d devices: wall %7.3f ms  speedup vs serial %.2fx\n", k, nd, w, k*b/w)
end
