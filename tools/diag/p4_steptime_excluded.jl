#!/usr/bin/env julia
# ===========================================================================
# p4_steptime_excluded.jl -- ROS23 step time: default pipeline vs the P4
#                            `excluded_passes` set that keeps DUS intact.
# ===========================================================================
# Companion to p4_pass_bisect.jl. That probe found the minimal pattern
# exclusion that stops the DUS->concat rewrite at the source (optimized module
# keeps the emitter's whole-buffer dynamic_update_slices; the ~79 big
# concatenates are gone). This probe times what that is WORTH: it compiles the
# ROS23 step twice in one process -- default COPTS, and COPTS plus the winning
# `excluded_passes` -- and times both on the same inputs.
#
# Both compiles keep sync=true and xla_cpu_prefer_vector_width=128 (the
# XLA:CPU race workaround, REQUIRED); the exclusion list is the only change.
#
#   RESEACT_NLON/NLAT/NLEV    grid (default 6 6 8; the CONUS arm is run by the
#                             operator, not by probe sessions)
#   P4_EXCLUDE                comma-separated pattern base-names to exclude
#                             (default: the P4 winner, see below)
#   P4_STEPTIME_REPS          timing reps (default 60)
#
# RESULTS (grid 6x6x8, Reactant v0.2.280, 2026-08-25):
#   default 10.98 ms -> excluded 5.22 ms median (best-of-two, interleaved)
#   = 2.10x, bit-for-bit equal unew. CONUS arm: see the campaign notes.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu")
get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"
ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p4steptime")

include(joinpath(@__DIR__, "_env.jl"))
using Printf, Statistics
say(s) = (println(s); flush(stdout))

# The P4 winner (p4_pass_bisect.jl). Overridable so other exclusion sets can
# be timed without editing the script.
const DEFAULT_EXCLUDE = "dynamic_update_to_concat,sub_const_prop"
const EXCL = String.(split(get(ENV, "P4_EXCLUDE", DEFAULT_EXCLUDE), ','))
const REPS = parse(Int, get(ENV, "P4_STEPTIME_REPS", "60"))

Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX

const COPTS_EXCL = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
    excluded_passes = EXCL)
say("excluded_passes = [" * join(EXCL, ", ") * "]")
const t0c = time()
const CEXC = RX.@compile compile_options = COPTS_EXCL D.ros_step(D.U_R, D.THC, D.T_R, D.DTC_R)
say(@sprintf("@compile excluded %.1f s", time() - t0c))

function time_steps(tag, cstep, reps)
    u = RX.ConcreteRArray(copy(D.UBASE))
    tt = RX.ConcreteRNumber(D.T0); dd = RX.ConcreteRNumber(D.DT0C)
    for _ in 1:3
        cstep(u, D.THC, tt, dd)
    end
    ts = Float64[]
    for _ in 1:reps
        t0 = time()
        cstep(u, D.THC, tt, dd)
        push!(ts, time() - t0)
    end
    say(@sprintf("P4_STEPTIME %-8s NC=%d median %.2f ms  min %.2f ms  reps=%d",
                 tag, D.NC, 1000 * median(ts), 1000 * minimum(ts), reps))
    return median(ts)
end

# Interleave the two arms A/B/A/B to cancel slow drift on a shared node.
const m1a = time_steps("default", D.CROS, REPS)
const m2a = time_steps("excluded", CEXC, REPS)
const m1b = time_steps("default", D.CROS, REPS)
const m2b = time_steps("excluded", CEXC, REPS)
const md = min(m1a, m1b); const mx = min(m2a, m2b)
say(@sprintf("P4_STEPTIME_RATIO default/excluded = %.3f  (best-of-two medians %.2f / %.2f ms)",
             md / mx, 1000 * md, 1000 * mx))

# One-step functional spot check on the same inputs (fresh buffers -- donation).
const u_def = Array(D.CROS(RX.ConcreteRArray(copy(D.UBASE)), D.THC,
                           RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(D.DT0C))[1])
const u_exc = Array(CEXC(RX.ConcreteRArray(copy(D.UBASE)), D.THC,
                         RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(D.DT0C))[1])
say("bit-for-bit equal: $(u_def == u_exc)")
u_def == u_exc || say(@sprintf("max abs diff %.3e", maximum(abs.(u_def .- u_exc))))
say("P4_STEPTIME_DONE")
