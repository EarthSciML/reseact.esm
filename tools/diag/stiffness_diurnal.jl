#!/usr/bin/env julia
# ===========================================================================
# stiffness_diurnal.jl -- PER-CELL CHEMISTRY STEP DEMAND ACROSS A FULL DAY
# ===========================================================================
# cell_stiffness.jl priced the stiffest-cell dictatorship on 2-3 pre-dawn
# windows. That is the wrong sample for the question that decides the
# stiffness-sharded design: which cells are stiff CHANGES as the sun moves,
# so the realizable saving depends on how well window m's demand predicts
# window m+1's -- and the interesting dynamics (sunrise, sunset, the
# terminator band) are exactly what a pre-dawn sample never sees.
#
# This records, for every macro window across RESEACT_ADJ_NMACRO windows
# (default 288 = 24 h of 300 s), the per-cell required chemistry steps
#   s[c, m] = sum over accepted steps of dt / clamp(dt_cell, 1e-6, window)
#   dt_cell = dt * (1/EEst_cell)^(1/3)      (the PI controller's own exponent)
# along the REAL global-controller trajectory. The offline companion
# (stiffness_policy_replay.py) then replays bucketing policies against the
# recorded matrix: static spatial shards, re-sort-by-previous-window, oracle.
# The predictor uses ONLY realized controller state -- deliberately NO model
# variables (no SZA or similar): it must work for any model, not just ones
# with photolysis named the way this one names it.
#
# One pass, not two: ros23_step(cellwise=true) returns (u, EEst, cell_err),
# and host_adaptive! consumes r[1]/r[2] positionally, so the cellwise program
# can BE the controller's step function; the per-cell error of each ACCEPTED
# step is harvested from the same call. (cell_stiffness.jl re-ran the accepted
# sequence through a second program instead -- 2x the cost.)
#
# Output: RESEACT_STIFF_OUT/percell_steps.bin, one record per window:
#   6 Float64 header (tcur, tnext, naT, nrT, naC, nrC) + NC Float64 s[:, m]
# plus meta.txt. Window count = filesize / (8*(6+NC)).
#
#   RESEACT_NLON/NLAT/NLEV    grid (default CONUS 13x7x72)
#   RESEACT_ADJ_NMACRO        windows (default 288 = 24 h)
#   RESEACT_STIFF_OUT         output dir (default logs/stiffdiurnal)
#
# MEASURED 2026-08-24 (local ccc0232, 62 min, logs/stiffdiurnal + replay_report.txt):
#   speedup in chemistry cell-steps vs the ACTUAL global controller (8.46e7):
#              K=4    K=16   K=64   K=128
#     spatial  1.39   1.77   1.91   2.09   (never re-sorted)
#     prev     1.67   2.87   3.91   4.36   (re-sort by previous window)
#     trend    1.63   2.91   4.06   4.56   (geometric s[m-1]^2/s[m-2])
#     oracle   2.14   4.38   5.61   5.86   (per-cell ideal 6.09)
#   log-demand autocorrelation med 0.998 / min 0.82; prev-vs-oracle penalty at
#   K=16 med 1.063, and the worst windows (up to 4.8x) are the sunrise sweep
#   t=42300-52200 -- exactly the terminator band the pre-dawn sample missed.
#   BONUS: the global RMS norm UNDER-resolves the stiffest cells ~30% (max-cell
#   demand / accepted = 1.30 median), so bucketed controllers are stricter where
#   it matters and equivalence tests must expect small diffs at stiff cells.
#   w001 bit-reproduced cellstiff-10105947 (75 accepted + 4 rejected).
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
get!(ENV, "RESEACT_ADJ_NMACRO", "288")
ENV["RESEACT_ADJ_UJITTER"] = "0"
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "stiffdiurnal"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX; RTI = D.RTI
say(s) = (println(s); flush(stdout))

# The same step the driver's SUBCYCLE path compiles; compiled here because
# SUBCYCLE stays OFF (this probe must not perturb the global-controller path).
ros_step_cw(u, th, t, dt) = RTI.ros23_step((uu, tt) -> D.gC(uu, th, tt), u, t, dt,
                                           D.NS, D.NC, D.MASKS, D.ATOL_C, D.RTOL;
                                           unrolled = true, jac = D.JACMODE,
                                           symjac = D.SYMJAC ? ((uu, tt) -> D.gJ(uu, th, tt)) : nothing,
                                           cellwise = true)

UD = RX.ConcreteRArray(copy(D.UBASE))
TD = RX.ConcreteRNumber(D.T0); DD = RX.ConcreteRNumber(D.DT0C)
let t0 = time()
    global CCW = RX.@compile compile_options=D.COPTS ros_step_cw(UD, D.THC, TD, DD)
    say(@sprintf("  @compile ros_step(cellwise) %.1f s", time() - t0))
end

# host_adaptive! transcribed from tools/adjoint_gradient.jl (section 5) with one
# addition: every ACCEPTED step's per-cell error contributes to `percell`.
# Controller arithmetic is kept line-for-line so the accepted sequence is the
# real global controller's.
function adaptive_cw!(uh::Vector{Float64}, t0::Float64, t1::Float64,
                      dt0::Float64, ctrl, TH, percell::Vector{Float64};
                      clamp_nonneg::Bool = true, maxiters::Int = 20000)
    beta1 = ctrl.beta1; beta2 = ctrl.beta2
    invqmax = 1.0 / ctrl.qmax; invqmin = 1.0 / ctrl.qmin
    gamma = ctrl.gamma
    qsmin = ctrl.qsteady_min; qsmax = ctrl.qsteady_max
    qoldinit = ctrl.qoldinit
    wlen = t1 - t0
    u = copy(uh); t = t0; dt = dt0; qold = qoldinit
    nacc = 0; nrej = 0; iters = 0
    tlim = t1 - 1.0e-9
    while (t < tlim) && (iters < maxiters)
        dtc = min(dt, t1 - t)
        r = CCW(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t), RX.ConcreteRNumber(dtc))
        raw = Array(r[1]); ee = Float64(r[2])
        EEst = isnan(ee) ? 1.0e10 : ee
        q11 = max(EEst, 1.0e-35)^beta1
        q = q11 / qold^beta2
        q = max(invqmax, min(invqmin, q / gamma))
        accept = EEst <= 1.0
        insteady = (qsmin <= q) & (q <= qsmax)
        qa = insteady ? one(q) : q
        if accept
            ce = Array(r[3])
            @. percell += dtc / clamp(dtc * (1.0 / max(ce, 1e-300))^(1 / 3), 1e-6, wlen)
            u = clamp_nonneg ? max.(raw, 0.0) : raw
            t = t + dtc
            dt = dtc / qa
            qold = max(EEst, qoldinit)
            nacc += 1
        else
            dt = dtc / min(invqmin, q11 / gamma)
            nrej += 1
        end
        iters += 1
    end
    iters >= maxiters && error("adaptive_cw! hit maxiters at t=$t (t1=$t1)")
    return u, t, dt, nacc, nrej
end

const OUTDIR = get(ENV, "RESEACT_STIFF_OUT", joinpath(REPO, "logs", "stiffdiurnal"))
mkpath(OUTDIR)

function main()
    open(joinpath(OUTDIR, "meta.txt"), "w") do io
        println(io, "grid=$(ENV["RESEACT_NLON"])x$(ENV["RESEACT_NLAT"])x$(ENV["RESEACT_NLEV"])")
        println(io, "NC=$(D.NC)")
        println(io, "NS=$(D.NS)")
        println(io, "T0=$(D.T0)")
        println(io, "MACRO_DT=$(D.MACRO_DT)")
        println(io, "NMACRO=$(D.NMACRO)")
        println(io, "rtol=$(D.RTOL) atol_chem=$(D.ATOL_C)")
        println(io, "record=6xFloat64 header (tcur,tnext,naT,nrT,naC,nrC) + NC x Float64 percell_steps")
    end
    say("\n" * "="^78)
    say(@sprintf("DIURNAL STIFFNESS  grid=%sx%sx%s  NC=%d  %d windows of %.0f s from T0=%.0f",
                 ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"],
                 D.NC, D.NMACRO, D.MACRO_DT, D.T0))
    say("="^78)
    bin = open(joinpath(OUTDIR, "percell_steps.bin"), "w")
    D.refresh_forcing(D.T0)
    u = copy(D.UBASE); tcur = D.T0; dtT = D.DT0T; dtC = D.DT0C
    m = 0; tstart = time()
    for tnext in D.STOPS
        tnext <= tcur + 1e-9 && continue
        m += 1
        uT, _, dtTe, naT, nrT = D.host_adaptive!(D.CSSP, u, tcur, tnext, dtT,
                                                 RTI.pictrl_ssprk43(), D.THT;
                                                 clamp_nonneg = D.CLAMP[])
        percell = zeros(Float64, D.NC)
        uC, _, dtCe, naC, nrC = adaptive_cw!(uT, tcur, tnext, dtC,
                                             RTI.pictrl_ros23(), D.THC, percell;
                                             clamp_nonneg = D.CLAMP[])
        write(bin, Float64(tcur), Float64(tnext),
              Float64(naT), Float64(nrT), Float64(naC), Float64(nrC))
        write(bin, percell); flush(bin)
        q = quantile(percell, [0.5, 0.9, 0.99, 1.0])
        say(@sprintf("  w%03d [%6.0f,%6.0f] chem %3d+%dr | need/cell med %5.1f p90 %6.1f p99 %6.1f max %7.1f | %5.0f s elapsed",
                     m, tcur, tnext, naC, nrC, q[1], q[2], q[3], q[4], time() - tstart))
        u = uC; tcur = tnext; dtT = dtTe; dtC = dtCe
        if round(tnext; digits = 6) in D.FSTOPS
            D.refresh_forcing(tnext)
        end
    end
    close(bin)
    say(@sprintf("\nDIURNAL_DONE  windows=%d  wall=%.0f s", m, time() - tstart))
end
main()
