#!/usr/bin/env julia
# ===========================================================================
# cell_stiffness.jl -- HOW MUCH IS THE GLOBAL dt COSTING?
# ===========================================================================
# The chemistry half is EXACTLY block-diagonal: every cell's 13-species ODE is
# independent of every other cell's. But the step size is chosen ONCE for the
# whole domain, from the Hairer RMS error norm over all 85,176 states. Measured
# at CONUS (slurm 10104808): one 300 s macro step takes 75 accepted chemistry
# steps, median dt 4.96 s, max 6.55 s -- never anywhere near the 300 s window.
#
# If the per-cell error is broadly distributed, most of those 6,552 cells are
# being stepped far below their own accuracy limit purely because some other
# cell needs a small step, and the wasted work grows as the grid grows: adding
# cells can only lower the global dt, never raise it. That is the term that
# makes this cost superlinear in grid size, which is exactly what "amenable to
# increasing the spatial scale" has to fix.
#
# This prices it. Following the SAME accepted step sequence the global
# controller takes, it records the per-cell error norm and reports, per cell,
# the step it COULD have taken: for the 2nd-order embedded pair,
#   dt_cell = dt * (1/EEst_cell)^(1/3)   (the same exponent the PI controller uses)
# and then compares
#   global work  = NC * (number of global steps)
#   per-cell work = sum over cells of ceil(window / dt_cell)
#
# It is an UPPER BOUND on the saving, not a promise: a per-cell controller still
# pays for the cells that are genuinely stiff, still has to re-synchronise at
# the macro-step boundary for transport, and the estimate above is a local
# extrapolation from one step rather than a re-integration. It is meant to say
# whether the lever is worth 1.5x or 10x before anyone builds it.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default CONUS)
#   RESEACT_STIFF_NMACRO     macro steps to sample (default 2)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "cellstiff"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX; RTI = D.RTI
say(s) = (println(s); flush(stdout))

const NMAC = parse(Int, get(ENV, "RESEACT_STIFF_NMACRO", "2"))

# The same step the driver compiles, asking for the per-cell error as well.
ros_step_cw(u, th, t, dt) = RTI.ros23_step((uu, tt) -> D.gC(uu, th, tt), u, t, dt,
                                           D.NS, D.NC, D.MASKS, D.ATOL_C, D.RTOL;
                                           unrolled = true, jac = D.JACMODE,
                                           symjac = D.SYMJAC ? ((uu, tt) -> D.gJ(uu, th, tt)) : nothing,
                                           cellwise = true)

UD = RX.ConcreteRArray(copy(D.UBASE))
TD = RX.ConcreteRNumber(D.T0); DD = RX.ConcreteRNumber(D.DT0C)
t0 = time()
CCW = RX.@compile compile_options=D.COPTS ros_step_cw(UD, D.THC, TD, DD)
say(@sprintf("  @compile ros_step(cellwise) %.1f s", time() - t0))

say("\n" * "="^78)
say(@sprintf("CELL STIFFNESS  grid=%sx%sx%s  NC=%d cells, NS=%d species",
             ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"], D.NC, D.NS))
say(@sprintf("  rtol=%.1e  atol_chem=%.1e  macro window %.0f s", D.RTOL, D.ATOL_C, D.MACRO_DT))
say("="^78)

function sample_stiffness(NMAC)
    D.refresh_forcing(D.T0)
    u = copy(D.UBASE); tcur = D.T0
    global_steps = 0
    percell_steps = zeros(Float64, D.NC)
    for m in 1:NMAC
        t1 = tcur + D.MACRO_DT
        # transport first, exactly as macro_step does, so chemistry sees the real state
        uT, _, _, naT, _ = D.host_adaptive!(D.CSSP, u, tcur, t1, D.DT0T,
                                            RTI.pictrl_ssprk43(), D.THT; clamp_nonneg = D.CLAMP[])
        seq = D.StepSeq()
        uC, _, _, naC, nrC = D.host_adaptive!(D.CROS, uT, tcur, t1, D.DT0C,
                                              RTI.pictrl_ros23(), D.THC;
                                              seq = seq, clamp_nonneg = D.CLAMP[])
        global_steps += naC
        say(@sprintf("\n  macro step %d: transport %d steps, chemistry %d accepted / %d rejected",
                     m, naT, naC, nrC))
        uu = copy(uT)
        allcell = Float64[]
        for (tt, dd) in seq
            r = CCW(RX.ConcreteRArray(uu), D.THC, RX.ConcreteRNumber(tt), RX.ConcreteRNumber(dd))
            raw = Array(r[1]); ce = Array(r[3])
            # the step this cell could have taken here: same exponent the PI
            # controller uses for a 2nd-order embedded pair
            dtc = dd .* (1.0 ./ max.(ce, 1e-300)) .^ (1 / 3)
            percell_steps .+= dd ./ min.(max.(dtc, 1e-6), D.MACRO_DT)
            append!(allcell, ce)
            uu = D.CLAMP[] ? max.(raw, 0.0) : raw
        end
        q = quantile(allcell, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        @printf("    per-cell EEst over %d (step, cell) pairs:\n", length(allcell))
        @printf("      min %.3e  p25 %.3e  med %.3e  p75 %.3e  p90 %.3e  p99 %.3e  max %.3e\n",
                q[1], q[2], q[3], q[4], q[5], q[6], q[7])
        @printf("      spread max/median = %.1fx   p99/median = %.1fx\n",
                q[7] / max(q[3], 1e-300), q[6] / max(q[3], 1e-300))
        flush(stdout)
        u = uC; tcur = t1
    end
    return global_steps, percell_steps
end
global_steps, percell_steps = sample_stiffness(NMAC)

window = NMAC * D.MACRO_DT
gw = float(global_steps) * D.NC
pw = sum(percell_steps)
say("\n" * "-"^78)
@printf("  over %.0f s of simulation:\n", window)
@printf("    global dt   : %d chemistry steps x %d cells = %.4g cell-steps\n", global_steps, D.NC, gw)
@printf("    per-cell dt : %.4g cell-steps  (sum over cells of window/dt_cell)\n", pw)
@printf("    UPPER-BOUND SAVING %.2fx\n", gw / max(pw, 1e-300))
q = quantile(percell_steps ./ (window / D.MACRO_DT), [0.5, 0.9, 0.99, 1.0])
@printf("    steps per macro window, per cell: median %.1f  p90 %.1f  p99 %.1f  max %.1f  (global takes %.1f)\n",
        q[1], q[2], q[3], q[4], global_steps / NMAC)
say("\nSTIFF_DONE")
