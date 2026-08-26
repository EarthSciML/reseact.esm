#!/usr/bin/env julia
# ===========================================================================
# level_schedule.jl -- WHAT DOES THE DYADIC-LEVEL SCHEME ACTUALLY REALISE?
# ===========================================================================
# `cell_stiffness.jl` prices the IDEAL: give every cell its own dt and the
# chemistry cell-step count falls 7.53x at CONUS. That is an upper bound and it
# is not achievable, for two reasons the scheme itself introduces:
#
#   1. DYADIC BUCKETING. A cell is run at dt = W/2^j for 2^j steps, so a cell
#      that needs 9 steps lands in the 16 bucket. Worst case 2x, and the loss is
#      exactly `2^ceil(log2 n) / n` averaged over the cells.
#   2. CHUNK PADDING. A level's cells are run in chunks of the build's lane
#      capacity C, so level j costs `2^j * C * ceil(n_j / C)` LANE-steps, not
#      `2^j * n_j`. Small levels with few cells pay a whole chunk.
#
# This measures both, on the same accepted step sequence the global controller
# takes, and sweeps C. It is the number to hold the implementation to.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default CONUS)
#   RESEACT_STIFF_NMACRO     macro steps to sample (default 2)
#   RESEACT_LS_JMAX          highest dyadic level (default 10 => 1024 substeps)
#   RESEACT_LS_CAPS          comma-separated capacities to sweep
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "levsched"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX; RTI = D.RTI
say(s) = (println(s); flush(stdout))

const NMAC = parse(Int, get(ENV, "RESEACT_STIFF_NMACRO", "2"))
const JMAX = parse(Int, get(ENV, "RESEACT_LS_JMAX", "10"))
const CAPS = parse.(Int, split(get(ENV, "RESEACT_LS_CAPS", "1024,2048,4096,6144,8192,16384"), ','))

ros_step_cw(u, th, t, dt) = RTI.ros23_step((uu, tt) -> D.gC(uu, th, tt), u, t, dt,
                                           D.NS, D.NC, D.MASKS, D.ATOL_C, D.RTOL;
                                           unrolled = true, jac = D.JACMODE,
                                           symjac = D.SYMJAC ? ((uu, tt) -> D.gJ(uu, th, tt)) : nothing,
                                           cellwise = true)
UD = RX.ConcreteRArray(copy(D.UBASE)); TD = RX.ConcreteRNumber(D.T0); DD = RX.ConcreteRNumber(D.DT0C)
t0 = time()
CCW = RX.@compile compile_options=D.COPTS ros_step_cw(UD, D.THC, TD, DD)
say(@sprintf("  @compile ros_step(cellwise) %.1f s", time() - t0))

say("\n" * "="^78)
say(@sprintf("LEVEL SCHEDULE  grid=%sx%sx%s  NC=%d cells, NS=%d species, jmax=%d",
             ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"], D.NC, D.NS, JMAX))
say("="^78)

"dyadic level of a cell needing `n` steps in the window"
level_of(n) = clamp(n <= 1 ? 0 : ceil(Int, log2(n)), 0, JMAX)

function sample(NMAC)
    D.refresh_forcing(D.T0)
    u = copy(D.UBASE); tcur = D.T0
    windows = NamedTuple[]
    for m in 1:NMAC
        t1 = tcur + D.MACRO_DT
        uT, _, _, naT, _ = D.host_adaptive!(D.CSSP, u, tcur, t1, D.DT0T,
                                            RTI.pictrl_ssprk43(), D.THT; clamp_nonneg = D.CLAMP[])
        seq = D.StepSeq()
        uC, _, _, naC, nrC = D.host_adaptive!(D.CROS, uT, tcur, t1, D.DT0C,
                                              RTI.pictrl_ros23(), D.THC;
                                              seq = seq, clamp_nonneg = D.CLAMP[])
        need = zeros(Float64, D.NC)
        uu = copy(uT)
        for (tt, dd) in seq
            r = CCW(RX.ConcreteRArray(uu), D.THC, RX.ConcreteRNumber(tt), RX.ConcreteRNumber(dd))
            raw = Array(r[1]); ce = Array(r[3])
            dtc = dd .* (1.0 ./ max.(ce, 1e-300)) .^ (1 / 3)
            need .+= dd ./ min.(max.(dtc, 1e-6), D.MACRO_DT)
            uu = D.CLAMP[] ? max.(raw, 0.0) : raw
        end
        push!(windows, (; m, naC, nrC, need))
        say(@sprintf("  macro %d: transport %d, chemistry %d accepted / %d rejected", m, naT, naC, nrC))
        u = uC; tcur = t1
    end
    return windows
end

W = sample(NMAC)

glob = sum(w.naC for w in W) * D.NC
ideal = sum(sum(max.(w.need, 1.0)) for w in W)
lv = [level_of.(max.(w.need, 1.0)) for w in W]
dyad = sum(sum(2.0 .^ l) for l in lv)

say("\n---- level histogram, PER WINDOW (the schedule is rebuilt each window) ----")
for (w, l) in zip(W, lv)
    counts = [count(==(j), l) for j in 0:JMAX]
    say(@sprintf("   macro %d (%d global steps): %s", w.m, w.naC,
                 join([@sprintf("j%d=%d", j, counts[j+1]) for j in 0:JMAX if counts[j+1] > 0], "  ")))
end

say("\n---- cost ----")
@printf("  global dt       : %d chemistry steps x %d cells   = %.4g cell-steps\n",
        sum(w.naC for w in W), D.NC, glob)
@printf("  ideal per-cell  : %.4g cell-steps   (%.2fx vs global)\n", ideal, glob / ideal)
@printf("  dyadic levels   : %.4g cell-steps   (%.2fx vs global, %.2fx of ideal)\n",
        dyad, glob / dyad, dyad / ideal)

# ---- chunking ------------------------------------------------------------
# ONE capacity is a trap and the numbers say so loudly. A level's cost is
# `2^j * (lanes actually run)`, and the SPARSE HIGH levels are the expensive
# ones: at CONUS levels 6-8 hold 0.5% of the cells but, rounded up to a full
# chunk of C = 1024, they cost more than half the total. The capacity has to be
# matched to the level's population -- which is affordable precisely because a
# capacity build is seconds, not the 725 s the full-grid build costs, so a
# LADDER of capacities is a handful of extra builds and compiles.
# Minimum lanes a level of `n` cells costs on a ladder. Take the LARGEST rung
# that still fits while one does, then ONE rung for the remainder -- taking the
# smallest rung >= n up front is what makes a ladder look worse the more rungs
# it has (1,611 cells billed a 4,096 chunk instead of 1,024 + 1,024).
function ladder_lanes(n::Int, ladder::Vector{Int})
    n <= 0 && return 0
    L = sort(ladder); tot = 0; rem = n
    while rem >= L[1]
        c = L[findlast(<=(rem), L)]
        tot += c; rem -= c
    end
    rem > 0 && (tot += L[1])
    return tot
end

const LADDERS = [
    ("single C=1024",      [1024]),
    ("single C=4096",      [4096]),
    ("64/1024",            [64, 1024]),
    ("64/512/4096",        [64, 512, 4096]),
    ("16/64/256/1024",     [16, 64, 256, 1024]),
    ("16/64/256/1024/4096",[16, 64, 256, 1024, 4096]),
    ("8/32/128/512/2048",  [8, 32, 128, 512, 2048]),
    ("32/256/2048",        [32, 256, 2048]),
    ("128/2048",           [128, 2048]),
]
say("\n---- lane-steps under a CAPACITY LADDER (built once, reused every window) ----")
for (name, ladder) in LADDERS
    lanes = 0.0
    for l in lv, j in 0:JMAX
        nj = count(==(j), l); nj == 0 && continue
        lanes += 2.0^j * ladder_lanes(nj, ladder)
    end
    @printf("   %-22s %10.4g lane-steps => %5.2fx vs global   (padding waste %4.1f%%, %d builds)\n",
            name, lanes, glob / lanes, 100 * (lanes - dyad) / lanes, length(ladder))
end
say("\nLEVSCHED_DONE")
