#!/usr/bin/env julia
# ===========================================================================
# steptime_shard.jl -- ROS23 step throughput for one grid slice (P1 probe)
# ===========================================================================
# Times the compiled chemistry step for whatever slice RESEACT_LON0/NLON/...
# select. Run once for full CONUS and once per half-domain shard; with
# RESEACT_STEPTIME_BARRIER set, the process times solo, signals, waits for the
# GO file, then times again while its peer runs concurrently. The comparison
#   throughput(full, solo)  vs  sum of throughput(shard_i, concurrent)
# is the decisive number for sharding chemistry across processes/devices:
# it banks BOTH the smaller working set (a half-CONUS extended observed
# buffer approaches L3) and the second execution stream.
#
#   RESEACT_STEPTIME_REPS      timing reps (default 60)
#   RESEACT_STEPTIME_BARRIER   GO-file path; empty = solo only
#
# MEASURED 2026-08-24/25 on ccc0232 (shared, load 17-26), 60 reps:
#   full CONUS solo  NC=6552  375.64 ms med / 212.72 min   57.3 us/cell
#   west  half solo  NC=3528  133.56 / 115.84              37.9 us/cell
#   east  half solo  NC=3024  111.58 /  93.10              36.9 us/cell
#   west  half CONC  NC=3528  141.06 / 127.52              (+5.6% vs solo)
#   east  half CONC  NC=3024  120.61 / 104.59              (+8.1% vs solo)
#   => domain step max(conc) 141.1 ms = 2.66x med (1.67x min) over full solo.
#   ~1.5x/cell working-set effect x a near-free second stream. Plain separate
#   PROCESSES -- no Reactant multi-device machinery needed.
#   Level-quarters 13x7x18 (NC=1638; nlon<6 fails the transport stencil, so
#   4-way geographic splits are impossible at CONUS): solo 30.3-31.7 us/cell
#   -- the working-set effect keeps deepening. The 4-process CONCURRENT phase
#   OOMs a 40GB memcg twice (3 holds ~3.5GB + 1 build peak + foreign juliae):
#   run N>=4 under a larger allocation, or with ESS_OOP_SSA=1 (module -40%).
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
ENV["RESEACT_ADJ_UJITTER"] = "0"
ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "steptime")

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX
say(s) = (println(s); flush(stdout))

const REPS = parse(Int, get(ENV, "RESEACT_STEPTIME_REPS", "60"))
const BAR = get(ENV, "RESEACT_STEPTIME_BARRIER", "")
const LBL = ENV["RESEACT_LABEL"]

function time_steps(tag, reps)
    u = RX.ConcreteRArray(copy(D.UBASE))
    tt = RX.ConcreteRNumber(D.T0); dd = RX.ConcreteRNumber(D.DT0C)
    for _ in 1:3
        D.CROS(u, D.THC, tt, dd)
    end
    ts = Float64[]
    tstart = time()
    for _ in 1:reps
        t0 = time()
        D.CROS(u, D.THC, tt, dd)
        push!(ts, time() - t0)
    end
    tend = time()
    say(@sprintf("STEPTIME %s %s NC=%d median %.2f ms min %.2f ms reps=%d span=[%.3f,%.3f]",
                 LBL, tag, D.NC, 1000 * median(ts), 1000 * minimum(ts), reps, tstart, tend))
end

time_steps("solo", REPS)
if BAR != ""
    touch(string(BAR, ".", LBL, ".solo.done"))
    say("  waiting on GO file: $BAR")
    while !isfile(BAR)
        sleep(1.0)
    end
    time_steps("concurrent", REPS)
end
say("STEPTIME_DONE $LBL")
