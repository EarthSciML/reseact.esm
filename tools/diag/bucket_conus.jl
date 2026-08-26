#!/usr/bin/env julia
# ===========================================================================
# bucket_conus.jl -- THE SPEED CLAIM: bucketed chemistry at CONUS 13x7x72.
# ===========================================================================
# bucket_verify.jl is the accuracy gate (at 6x6x8 the stiffness spread is small
# and the ratio proves nothing). This measures the thing the design banks on:
# chemistry-half WALL per window and total CELL-STEPS across one diurnal
# segment that CONTAINS THE SUNRISE SWEEP t=42300-52200 -- the trend
# predictor's worst case -- for three arms in ONE process against ONE build:
#
#   lockstep        the global controller (the baseline)
#   bucketed K=16   offline replay priced ~2.9x cell-steps
#   bucketed K=64   offline replay priced ~4x cell-steps
#
# The offline-replay figures are cell-step ratios along the RECORDED lockstep
# trajectory; the arms here integrate their own trajectories, so the measured
# ratio is the real, deliverable one (gather + launch overheads included in
# the wall figure). A few windows of warmup precede the sunrise band so the
# predictor has the history the trend formula needs.
#
#   RESEACT_T0=41400, RESEACT_ADJ_NMACRO=40  ->  [41400, 53400], 40 windows,
#   covering 42300-52200 with 3 windows of warmup.
#
# Output: one WINDOW line per (arm, window) with the chemistry wall and
# cell-steps, an ARM summary each, and RESULT lines with the ratios.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
get!(ENV, "RESEACT_T0", "41400")
get!(ENV, "RESEACT_ADJ_NMACRO", "40")
get!(ENV, "RESEACT_BUCKET", "16")
get!(ENV, "RESEACT_BUCKET_KS", "16,64")
ENV["RESEACT_ADJ_UJITTER"] = "0"
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_SUBCYCLE"] = "0"
ENV["RESEACT_LABEL"] = "bucketconus"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX; RTI = D.RTI
say(s) = (println(s); flush(stdout))

const NC = D.NC
const SUNRISE = (42300.0, 52200.0)
inband(t0, t1) = t1 > SUNRISE[1] && t0 < SUNRISE[2]

say("\n" * "="^78)
say(@sprintf("BUCKET CONUS  grid=%sx%sx%s  NC=%d  T0=%.0f  %d windows of %.0f s  rungs %s",
             ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"],
             NC, D.T0, D.NMACRO, D.MACRO_DT, string(D.BLADDER.caps)))
say(@sprintf("  sunrise band %.0f-%.0f is windows %d..%d of the segment",
             SUNRISE[1], SUNRISE[2],
             max(1, ceil(Int, (SUNRISE[1] - D.T0) / D.MACRO_DT)),
             ceil(Int, (SUNRISE[2] - D.T0) / D.MACRO_DT)))
say("="^78)

# One arm over the driver's macro-stop lattice. K = 0 means lockstep.
function run_arm(label::String, K::Int)
    say("\n---- ARM $label ----")
    D.refresh_forcing(D.T0)
    D.bucket_reset!()
    u = copy(D.UBASE); t = D.T0; dtT = D.DT0T; dtC = D.DT0C
    m = 0
    tchem = 0.0; ttrans = 0.0
    cellsteps = 0.0; acc_cellsteps = 0.0; calls = 0; rejects = 0
    st = D.BucketStats()
    tstart = time()
    for tnext in D.STOPS
        tnext <= t + 1e-9 && continue
        m += 1
        tt = time()
        uT, _, dtT, naT, _ = D.host_adaptive!(D.CSSP, u, t, tnext, dtT,
                                              RTI.pictrl_ssprk43(), D.THT; clamp_nonneg = D.CLAMP[])
        ttrans += time() - tt
        tc = time()
        local wsteps, wrej, wcell
        if K == 0
            u, _, dtC, naC, nrC = D.host_adaptive!(D.CROS, uT, t, tnext, dtC,
                                                   RTI.pictrl_ros23(), D.THC; clamp_nonneg = D.CLAMP[])
            wsteps = naC; wrej = nrC
            wcell = Float64(naC + nrC) * NC
            cellsteps += wcell; acc_cellsteps += Float64(naC) * NC
            calls += naC + nrC; rejects += nrC
        else
            u, s1 = D.bucket_window!(D.BLADDER, uT, t, tnext; K = K,
                                     state = D.BUCKET_STATE, verbose = false)
            st = st + s1
            wsteps = s1.accepts; wrej = s1.rejects
            wcell = s1.cellsteps
            cellsteps += s1.cellsteps; acc_cellsteps += s1.acc_cellsteps
            calls += s1.calls; rejects += s1.rejects
        end
        wchem = time() - tc
        tchem += wchem
        say(@sprintf("  WINDOW %s w%03d [%6.0f,%6.0f]%s chemwall %7.2f s  acc %4d rej %3d  cellsteps %.4g",
                     label, m, t, tnext, inband(t, tnext) ? " *sunrise*" : "          ",
                     wchem, wsteps, wrej, wcell))
        flush(stdout)
        u_ok = all(isfinite, u)
        u_ok || error("ARM $label: non-finite state after window $m")
        t = tnext
        if round(tnext; digits = 6) in D.FSTOPS
            D.refresh_forcing(tnext)
        end
    end
    say(@sprintf("  ARM %s TOTAL: chem wall %.2f s, transport %.2f s, %d calls (%d rejected), cell-steps %.6g (accepted %.6g), %.0f s elapsed",
                 label, tchem, ttrans, calls, rejects, cellsteps, acc_cellsteps, time() - tstart))
    K > 0 && D.bucket_report(st)
    return (; label, K, tchem, ttrans, cellsteps, acc_cellsteps, calls, rejects)
end

A0  = run_arm("lockstep", 0)
A16 = run_arm("K16", 16)
A64 = run_arm("K64", 64)

say("\n" * "="^78)
for A in (A16, A64)
    @printf("RESULT %s cellstep_speedup %.3fx  (lockstep %.6g attempted / bucketed %.6g attempted)\n",
            A.label, A0.cellsteps / max(A.cellsteps, 1), A0.cellsteps, A.cellsteps)
    @printf("RESULT %s WALL_speedup     %.3fx  (chemistry %.2f s lockstep vs %.2f s bucketed)\n",
            A.label, A0.tchem / max(A.tchem, 1e-9), A0.tchem, A.tchem)
end
@printf("RESULT lockstep chem %.2f s, %.6g cell-steps (%.6g accepted) over %d windows\n",
        A0.tchem, A0.cellsteps, A0.acc_cellsteps, D.NMACRO)
say("="^78)
say("BUCKETCONUS_DONE")
