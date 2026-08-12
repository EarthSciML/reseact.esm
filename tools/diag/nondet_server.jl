#!/usr/bin/env julia
# ===========================================================================
# nondet_server.jl -- one build, many experiments.
# ===========================================================================
# The ReSEACT build is ~530 s and the two primal compiles another ~170 s, so a
# nondeterminism hunt that needs sample size cannot afford one process per
# question. This builds once (through tools/adjoint_gradient.jl's own build and
# compile path, STAGES=none) and then watches a directory: drop `foo.jl` into
# `tools/diag/jobs/` and it is `include`d in this process, with CSSP/CROS/THT/
# THC/UBASE/var_map/GROUP_OF and everything else the driver defines in scope.
# Write `STOP` to quit.
#
# Env: RESEACT_JOBDIR (default tools/diag/jobs), plus adjoint_gradient.jl's.
# ===========================================================================
ENV["RESEACT_ADJ_STAGES"] = get(ENV, "RESEACT_ADJ_STAGES", "none")
const _WANT_VJP = get(ENV, "RESEACT_HAMMER_VJP", "0") == "1"
_WANT_VJP && (ENV["RESEACT_ADJ_STAGES"] = "vjponly")
include(joinpath(@__DIR__, "..", "adjoint_gradient.jl"))

using Printf, Random, Dates

# The VJPs, compiled here rather than by the driver so no stage machinery runs.
const CSSPV2 = _WANT_VJP ?
    timed_compile("ssp_vjp", () -> RX.@compile sync=true ssp_vjp(U_R, THT, LAM_R, T_R, DTT_R)) : nothing
const CROSV2 = _WANT_VJP ?
    timed_compile("ros_vjp", () -> RX.@compile sync=true ros_vjp(U_R, THC, LAM_R, T_R, DTC_R)) : nothing

const JOBDIR = get(ENV, "RESEACT_JOBDIR", joinpath(@__DIR__, "jobs"))
mkpath(JOBDIR)

loadavg() = try; parse(Float64, split(read("/proc/loadavg", String))[1]); catch; NaN; end
callstep(c, u::Vector{Float64}, TH, t, dt) =
    c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t), RX.ConcreteRNumber(dt))
bitdiff(a::Vector{Float64}, b::Vector{Float64}) = findall(i -> !(a[i] === b[i]), eachindex(a))

# A realistic point one transport + one chemistry step past the base point.
const USTEP = let r = callstep(CSSP, UBASE, THT, T0, DT0T)
    Array(callstep(CROS, Array(r[1]), THC, T0, DT0C)[1])
end

"""Report a non-finite (or otherwise interesting) output vector: which state
groups, which grid cells, and whether an immediate re-issue reproduces it."""
function report_bad(nm, out, u, t, dt, c, TH; kind = "non-finite",
                    pred = !isfinite)
    idx = findall(pred, out)
    isempty(idx) && return
    groups = Dict{String,Int}()
    for i in idx; groups[GROUP_OF[i]] = get(groups, GROUP_OF[i], 0) + 1; end
    say(@sprintf("     %s: %d/%d entries at t=%.6f dt=%.10g, %d groups",
                 kind, length(idx), length(out), t, dt, length(groups)))
    say("     per group: " * join(("$k=$v" for (k, v) in sort(collect(groups))), " "))
    cells = Tuple{Int,Int,Int}[]
    lin = Int[]
    for (vn, vi) in var_map
        vi in idx || continue
        m = match(r"\[(\d+),(\d+),(\d+)\]$", vn)
        m === nothing && continue
        push!(cells, (parse(Int, m.captures[1]), parse(Int, m.captures[2]),
                      parse(Int, m.captures[3])))
    end
    ucells = sort(unique(cells))
    say(@sprintf("     %d distinct cells (i,j,k): %s", length(ucells),
                 join(ucells[1:min(end, 60)], " ")))
    # the within-species-block linear cell offsets, which is what a partitioned
    # loop over the NC axis would show as a contiguous run
    off = sort(unique(((i - 1) % NC) + 1 for i in idx))
    say(@sprintf("     cell offsets in [1,%d]: %s  (contiguous=%s)", NC,
                 join(off[1:min(end, 60)], ","),
                 string(length(off) > 1 && off[end] - off[1] + 1 == length(off))))
    flush(stdout)
end

say("\n" * "="^75)
say("NONDET SERVER ready.  jobdir=$JOBDIR  threads=$(Threads.nthreads())  $(now())")
say("  N=$N NS=$NS NC=$NC  |USTEP|max=$(maximum(abs, USTEP))")
say("="^75); flush(stdout)

const DONE = joinpath(JOBDIR, "done")
mkpath(DONE)
while true
    fs = sort(filter(f -> endswith(f, ".jl"), readdir(JOBDIR)))
    if isfile(joinpath(JOBDIR, "STOP"))
        say("STOP seen; exiting."); break
    end
    if isempty(fs)
        sleep(2); continue
    end
    for f in fs
        p = joinpath(JOBDIR, f)
        say("\n>>>> JOB $f  $(now())  load=$(loadavg())"); flush(stdout)
        try
            Base.include(Main, p)
        catch e
            say("!!!! JOB $f FAILED: $(sprint(showerror, e))")
            for l in Base.stacktrace(catch_backtrace())[1:min(end, 12)]; say("      $l"); end
        end
        say("<<<< JOB $f done $(now())"); flush(stdout)
        mv(p, joinpath(DONE, f); force = true)
    end
end
