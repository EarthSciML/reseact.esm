#!/usr/bin/env julia
# ===========================================================================
# subcycle_verify.jl -- IS THE LEVEL SUBCYCLE THE SAME INTEGRATION, AND IS IT
#                       ACTUALLY FASTER IN WALL TIME?
# ===========================================================================
# Both halves in ONE process, against ONE build, so the comparison cannot drift:
# the driver is included with RESEACT_SUBCYCLE=1, which compiles the global-dt
# chemistry step (`CROS`) *and* the capacity ladder. The global path is then
# `host_adaptive!(CROS, ...)` and the subcycled path is `subcycle_chem(...)`,
# started from the same state over the same window.
#
# THE CHECKS, in increasing order of how much they can hide.
#
#   1. PERMUTE ROUNDTRIP -- lanes_in! then lanes_out! must be BIT-IDENTICAL.
#      This is index algebra on the host, so bit-identity is the actual claim
#      and `isequal` is the actual test (`==` would call 0.0 == -0.0 equal and
#      NaN != NaN, and this is exactly the kind of check that is worth nothing
#      if it cannot see a NaN).
#   2. ONE STEP AT CAPACITY -- one ROS23 step on the largest rung against the
#      same step from the global program. Chemistry is pointwise, so a subset of
#      cells is a fair comparison. Expect the capacity build's own RHS floor
#      (2.888e-14 relative, capC_probe), amplified a little by the step.
#   3. ONE WINDOW -- the whole level schedule against the global controller from
#      the same state. These are DIFFERENT step sequences, so the bar is solver
#      tolerance (RTOL = 1e-4), not roundoff.
#   4. A TRAJECTORY -- NW independent macro steps each way, transport included.
#
# EVERY float check has a NEGATIVE CONTROL, because on this exact code a prior
# agent had three runs "pass" while every lane was NaN: `max` over relative
# errors silently reports 0 on NaN, since `NaN > worst` is false. So the
# comparison below scores a non-finite difference as Inf (never as "skip"), and
# each check is re-run against a deliberately mis-mapped state that it MUST
# report -- a check that cannot fail has not passed.
#
#   RESEACT_NLON/NLAT/NLEV       grid (default 6x6x8 -> NC = 288)
#   RESEACT_SUBCYCLE_LADDER      capacity ladder (default 8,32,128,512)
#   RESEACT_SUB_NW               macro steps in the trajectory check (default 2)
#   RESEACT_SUB_TIMING           1 = also run the wall-clock A/B (default 1)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_SUBCYCLE_LADDER", "8,32,128,512")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_SUBCYCLE"] = "1"
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "subverify"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics, Random, LinearAlgebra
RX = D.RX; RTI = D.RTI
say(s) = (println(s); flush(stdout))

const NW      = parse(Int, get(ENV, "RESEACT_SUB_NW", "2"))
const TIMING  = get(ENV, "RESEACT_SUB_TIMING", "1") == "1"
const L       = D.LADDER
const NC      = D.NC
const NS      = D.NS
const RNG     = MersenneTwister(20260824)

say("\n" * "="^78)
say(@sprintf("SUBCYCLE VERIFY  grid=%sx%sx%s  NC=%d cells, NS=%d species  ladder=%s  jmax=%d",
             ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"],
             NC, NS, string(D.SUB_LADDER), D.SUB_JMAX))
say("="^78)

# --------------------------------------------------------------------------- #
# The comparison. A NON-FINITE difference is the WORST outcome, not a skipped
# one -- see the header.
# --------------------------------------------------------------------------- #
"""
    worst_rel(a, b, cells) -> (worst, (species_block, cell))

Worst |a-b| over `cells`, divided by `max(|a|, 1e-9 * that species' own max)`.

The FLOOR is not cosmetic. `clamp_nonneg` writes EXACT zeros, so a state that
one path clamps to 0.0 and the other leaves at 1e-30 has an infinite relative
difference and nothing else to say about it. Flooring the denominator at 1e-9 of
the species scale makes the measure a mixed absolute/relative one below that
level -- still five orders BELOW the solver's own RTOL of 1e-4, so nothing a
real disagreement could hide behind.
"""
function worst_rel(a::Vector{Float64}, b::Vector{Float64}, cells)
    worst = 0.0; at = (0, 0)
    for s in 1:NS
        base = (s - 1) * NC
        scale = 0.0
        for c in cells
            v = a[base + c]
            isfinite(v) && (scale = max(scale, abs(v)))
        end
        for c in cells
            x = a[base + c]; y = b[base + c]
            den = max(abs(x), scale * 1e-9, 1e-300)
            r = (isfinite(x) && isfinite(y)) ? abs(x - y) / den : Inf
            if r > worst || (r == Inf && worst < Inf)
                worst = r; at = (s, c)
            end
        end
    end
    return worst, at
end

# The two cells a transposition negative control swaps. Adjacent cells can hold
# nearly the same value, and a control that swaps two near-equal numbers is a
# control that cannot fire; take the EXTREMES of species block 1 instead.
function control_pair(u::Vector{Float64})
    lo, hi = 1, 1
    for c in 1:NC
        u[c] < u[lo] && (lo = c)
        u[c] > u[hi] && (hi = c)
    end
    return lo, hi
end
function transpose_cells(u::Vector{Float64}, c1::Int, c2::Int)
    v = copy(u)
    for s in 1:NS
        b = (s - 1) * NC
        v[b + c1], v[b + c2] = v[b + c2], v[b + c1]
    end
    return v
end

nfail = Ref(0)
function check(name::AbstractString, got::Float64, bound::Float64)
    ok = isfinite(got) && got <= bound
    ok || (nfail[] += 1)
    say(@sprintf("  %-46s %.3e  (bound %.1e)  %s", name, got, bound, ok ? "PASS" : "FAIL"))
    return ok
end
function control(name::AbstractString, got::Float64, floor_::Float64)
    ok = !(got <= floor_)          # NaN-safe: a NaN control still counts as fired
    ok || (nfail[] += 1)
    say(@sprintf("  %-46s %.3e  (must exceed %.1e)  %s", name, got, floor_,
                 ok ? "FIRED" : "DID NOT FIRE -- the check above proves nothing"))
    return ok
end

# --------------------------------------------------------------------------- #
# Setup: a real post-transport state at T0.
# --------------------------------------------------------------------------- #
D.refresh_forcing(D.T0)
const T0 = D.T0
const T1 = T0 + D.MACRO_DT
const W  = T1 - T0
UT = let u = copy(D.UBASE)
    r, _, _, na, _ = D.host_adaptive!(D.CSSP, u, T0, T1, D.DT0T,
                                      RTI.pictrl_ssprk43(), D.THT; clamp_nonneg = D.CLAMP[])
    say(@sprintf("  transport half over [%.0f, %.0f]: %d accepted steps", T0, T1, na))
    r
end
all(isfinite, UT) || error("subcycle_verify: the transport half returned non-finite states")

# --------------------------------------------------------------------------- #
# CHECK 1 -- the permute roundtrip is bit-identical.
# --------------------------------------------------------------------------- #
say("\n---- 1. permute / unpermute (host index algebra: BIT-IDENTICAL is the claim) ----")
function roundtrip(rung, cellpos::Vector{Int}; outpos::Vector{Int} = cellpos)
    C = rung.C; n = length(cellpos)
    lane  = [cellpos[i <= n ? i : mod1(i - n, n)] for i in 1:C]
    olane = [outpos[i <= n ? i : mod1(i - n, n)] for i in 1:C]
    uc = D.lanes_in!(fill(NaN, length(rung.u0h)), UT, rung, lane)
    uo = fill(NaN, length(UT))
    D.lanes_out!(uo, uc, rung, olane, n)
    return uo, lane
end
let rung = L.rungs[maximum(L.caps)], n = min(rung.C, NC)
    for (label, cellpos) in (("identity", collect(1:n)),
                             ("shuffled", shuffle(RNG, collect(1:NC))[1:n]))
        uo, _ = roundtrip(rung, cellpos)
        nbad = count(c -> any(s -> !isequal(uo[(s - 1) * NC + c], UT[(s - 1) * NC + c]), 1:NS),
                     cellpos)
        nfail[] += (nbad == 0 ? 0 : 1)
        say(@sprintf("  %-46s %d of %d cells differ  %s",
                     "roundtrip $label (isequal)", nbad, n, nbad == 0 ? "PASS" : "FAIL"))
    end
    # NEGATIVE CONTROL: write two cells back to the wrong place.
    cellpos = collect(1:n)
    outpos = copy(cellpos); outpos[1], outpos[2] = outpos[2], outpos[1]
    uo, _ = roundtrip(rung, cellpos; outpos = outpos)
    nbad = count(c -> any(s -> !isequal(uo[(s - 1) * NC + c], UT[(s - 1) * NC + c]), 1:NS),
                 cellpos)
    nfail[] += (nbad > 0 ? 0 : 1)
    say(@sprintf("  %-46s %d of %d cells differ  %s",
                 "NEG CONTROL two cells written back swapped", nbad, n,
                 nbad > 0 ? "FIRED" : "DID NOT FIRE"))
end

# --------------------------------------------------------------------------- #
# CHECK 2 -- one capacity step against the global program's same step.
# --------------------------------------------------------------------------- #
# `gcellpos` is what the LANE INPUTS are gathered from; `cellpos` is what the
# state is mapped by and what the answer is compared against. They differ only
# in the negative control -- otherwise the check proves nothing (capC_probe.jl).
say("\n---- 2. ONE step at capacity vs the global program (pointwise: a subset is fair) ----")
function cap_step(rung, cellpos::Vector{Int}, u::Vector{Float64}, t::Float64, dt::Float64;
                  gcellpos::Vector{Int} = cellpos)
    C = rung.C; n = length(cellpos)
    lane  = [cellpos[i <= n ? i : mod1(i - n, n)] for i in 1:C]
    glane = [gcellpos[i <= n ? i : mod1(i - n, n)] for i in 1:C]
    D.batch_gather!(L, rung, [L.cells[c] for c in glane])
    uc = D.lanes_in!(copy(rung.u0h), u, rung, lane)
    res = rung.cstep(RX.ConcreteRArray(uc), rung.th, RX.ConcreteRNumber(t),
                     RX.ConcreteRNumber(dt))
    raw = Array(res[1])
    uo = copy(u)
    D.lanes_out!(uo, D.CLAMP[] ? max.(raw, 0.0) : raw, rung, lane, n)
    return uo, Float64(res[2]), Array(res[3])
end

const DT1 = min(D.DT0C, W)
let rung = L.rungs[maximum(L.caps)], n = min(rung.C, NC)
    cellpos = shuffle(RNG, collect(1:NC))[1:n]      # a RANDOM subset AND a random order
    rg = D.CROS(RX.ConcreteRArray(UT), D.THC, RX.ConcreteRNumber(T0), RX.ConcreteRNumber(DT1))
    ugl = Array(rg[1]); ugl = D.CLAMP[] ? max.(ugl, 0.0) : ugl
    say(@sprintf("  global step at dt=%.4g: EEst=%.4e, %d non-finite states",
                 DT1, Float64(rg[2]), count(!isfinite, ugl)))
    ucap, ee, ce = cap_step(rung, cellpos, UT, T0, DT1)
    say(@sprintf("  capacity step C=%d over %d cells: EEst=%.4e, max cell err %.4e",
                 rung.C, n, ee, maximum(ce[1:n])))
    w, at = worst_rel(ugl, ucap, cellpos)
    check("one step, capacity C=$(rung.C) vs global", w, 1e-10)
    say(@sprintf("      (worst at species block %d, cell position %d)", at[1], at[2]))
    # NEGATIVE CONTROL: two lanes' INPUTS gathered from the wrong cells.
    lo, hi = control_pair(UT)
    il = something(findfirst(==(lo), cellpos), 1)
    ih = something(findfirst(==(hi), cellpos), 2)
    il == ih && (ih = il == 1 ? 2 : 1)
    g = copy(cellpos); g[il], g[ih] = g[ih], g[il]
    ubad, _, _ = cap_step(rung, cellpos, UT, T0, DT1; gcellpos = g)
    wb, _ = worst_rel(ugl, ubad, cellpos)
    control("NEG CONTROL two lanes gathered from wrong cells", wb, 1e-8)
    # restore the correct gather so nothing downstream inherits it
    cap_step(rung, cellpos, UT, T0, DT1)
end

# --------------------------------------------------------------------------- #
# CHECK 3 -- one whole window: the level schedule vs the global controller.
# --------------------------------------------------------------------------- #
say("\n---- 3. ONE macro window: level schedule vs global controller ----")
D.SUB_NEED[] = nothing; D.SUB_DTPROBE[] = D.DT0C
UG1, _, _, nag, nrg = D.host_adaptive!(D.CROS, UT, T0, T1, D.DT0C,
                                       RTI.pictrl_ros23(), D.THC; clamp_nonneg = D.CLAMP[])
say(@sprintf("  global controller: %d accepted / %d rejected  =>  %d cell-steps",
             nag, nrg, nag * NC))
# ---- 3b. WHY does every cell land at the ceiling? -------------------------
# A DYADIC LEVEL IS 2^j EQUAL STEPS ACROSS THE WHOLE WINDOW, so it has to
# resolve the cell's WORST INSTANT in that window. The 7.53x/5.14x that
# tools/diag/cell_stiffness.jl and level_schedule.jl priced is a different
# quantity: they accumulate `need += dd / dtc(t)` over the accepted step
# sequence, which is the count for a per-cell VARIABLE dt -- it banks the
# temporal adaptivity as well as the spatial. A level banks only the spatial.
#
# This measures both on the SAME window, per cell, off the global controller's
# own accepted sequence, so the gap between them is a number rather than an
# argument.
say("\n---- 3b. integral need (variable per-cell dt) vs worst-instant need (a LEVEL) ----")
need_int, need_worst = let
    seq = D.StepSeq()
    D.host_adaptive!(D.CROS, UT, T0, T1, D.DT0C, RTI.pictrl_ros23(), D.THC;
                     seq = seq, clamp_nonneg = D.CLAMP[])
    ni = zeros(NC); nw = ones(NC); strict = Float64[]
    uu = copy(UT)
    for (tt, dd) in seq
        r = D.CROSCW(RX.ConcreteRArray(uu), D.THC, RX.ConcreteRNumber(tt),
                     RX.ConcreteRNumber(dd))
        raw = Array(r[1]); ce = Array(r[3])
        # How much STRICTER is a per-cell test than the global one? The global
        # controller accepts on the RMS over the whole domain, so an individual
        # cell is allowed to exceed 1; the subcycle requires EVERY cell to be
        # under 1. That ratio is work the subcycle does which the global run
        # simply does not do, and it belongs in the accounting rather than in
        # the speedup.
        let g = Float64(r[2])
            m = 0.0
            for c in 1:NC; isfinite(ce[c]) && (m = max(m, ce[c])); end
            g > 0 && push!(strict, m / g)
        end
        for c in 1:NC
            e = ce[c]
            dtc = (isfinite(e) && e > 0) ? dd * (1.0 / e)^(1 / 3) : dd * 1e-6
            ni[c] += dd / min(max(dtc, 1e-6), W)      # the level_schedule.jl quantity
            nw[c] = max(nw[c], W / max(dtc, 1e-12))   # what 2^j EQUAL steps must cover
        end
        uu = D.CLAMP[] ? max.(raw, 0.0) : raw
    end
    say(@sprintf("  per-cell test vs the global RMS test: max_c(ce)/EEst median %.1fx, max %.1fx over %d accepted steps",
                 median(strict), maximum(strict), length(strict)))
    ni, nw
end
lv_int = D.LevelSubcycle.assign_levels(max.(need_int, 1.0), D.SUB_JMAX)
lv_wst = D.LevelSubcycle.assign_levels(need_worst, D.SUB_JMAX)
hist(l) = join([@sprintf("j%d=%d", j, count(==(j), l)) for j in 0:D.SUB_JMAX
                if count(==(j), l) > 0], "  ")
say(@sprintf("  integral      need: median %.4g  max %.4g   levels %s",
             median(need_int), maximum(need_int), hist(lv_int)))
say(@sprintf("  worst-instant need: median %.4g  max %.4g   levels %s",
             median(need_worst), maximum(need_worst), hist(lv_wst)))
say(@sprintf("  ratio worst/integral: median %.1fx  max %.1fx",
             median(need_worst ./ max.(need_int, 1.0)),
             maximum(need_worst ./ max.(need_int, 1.0))))
say(@sprintf("  cell-steps: global %.4g | dyadic on INTEGRAL need %.4g (%.2fx) | dyadic on a LEVEL %.4g (%.2fx)",
             Float64(nag) * NC, sum(2.0 .^ lv_int), Float64(nag) * NC / sum(2.0 .^ lv_int),
             sum(2.0 .^ lv_wst), Float64(nag) * NC / sum(2.0 .^ lv_wst)))

US1, NEED1, ST1 = D.subcycle_chem(L, UT, T0, T1)
D.subcycle_report(ST1; nglobal_steps = nag)
let (w, at) = worst_rel(UG1, US1, 1:NC)
    check("one window, subcycled vs global", w, 5e-3)
    say(@sprintf("      (worst at species block %d, cell position %d; RTOL is %.0e)",
                 at[1], at[2], D.RTOL))
    # NEG CONTROL: the same metric against a state with two cells transposed.
    c1, c2 = control_pair(US1)
    wb, _ = worst_rel(UG1, transpose_cells(US1, c1, c2), 1:NC)
    control("NEG CONTROL two cells transposed in the answer", wb, 5e-3)
end
say(@sprintf("  level histogram: %s",
             join([@sprintf("j%d=%d", j, count(==(j), D.LevelSubcycle.assign_levels(NEED1, D.SUB_JMAX)))
                   for j in 0:D.SUB_JMAX
                   if count(==(j), D.LevelSubcycle.assign_levels(NEED1, D.SUB_JMAX)) > 0], "  ")))

# --------------------------------------------------------------------------- #
# CHECK 4 -- a trajectory, transport included, NW macro steps each way.
#            This is also the WALL-CLOCK A/B: same process, same build, same
#            forcing, so the only difference between the two numbers is the
#            chemistry half's scheme.
# --------------------------------------------------------------------------- #
say("\n---- 4. a $NW-macro-step trajectory, and the wall-clock A/B ----")
function run_traj(mode::Symbol, nw::Int)
    D.refresh_forcing(T0)
    D.SUB_NEED[] = nothing; D.SUB_DTPROBE[] = D.DT0C
    u = copy(D.UBASE); t = T0; dtT = D.DT0T; dtC = D.DT0C
    tchem = 0.0; ttrans = 0.0; nacc = 0; nrej = 0; ncalls = 0
    st = D.SubStats()
    t0w = time()
    for m in 1:nw
        t1 = t + D.MACRO_DT
        tt = time()
        uT, _, dtT, naT, _ = D.host_adaptive!(D.CSSP, u, t, t1, dtT,
                                              RTI.pictrl_ssprk43(), D.THT; clamp_nonneg = D.CLAMP[])
        ttrans += time() - tt
        tc = time()
        if mode === :global
            u, _, dtC, naC, nrC = D.host_adaptive!(D.CROS, uT, t, t1, dtC,
                                                   RTI.pictrl_ros23(), D.THC; clamp_nonneg = D.CLAMP[])
            nacc += naC; nrej += nrC; ncalls += naC + nrC
        else
            u, need, s1 = D.subcycle_chem(L, uT, t, t1;
                                          need_prev = D.SUB_NEED[], dt_probe = D.SUB_DTPROBE[])
            D.SUB_NEED[] = need
            D.SUB_DTPROBE[] = D.MACRO_DT /
                2.0^clamp(round(Int, median(log2.(max.(need, 1.0)))), 0, D.SUB_JMAX)
            st = st + s1; ncalls += s1.calls; nrej += s1.rejects
        end
        tchem += time() - tc
        t = t1
    end
    return (; u, wall = time() - t0w, tchem, ttrans, nacc, nrej, ncalls, st)
end

RG = run_traj(:global, NW)
say(@sprintf("  GLOBAL   : %.2f s total = %.2f s chemistry + %.2f s transport   (%d accepted / %d rejected, %d calls)",
             RG.wall, RG.tchem, RG.ttrans, RG.nacc, RG.nrej, RG.ncalls))
RS = run_traj(:sub, NW)
say(@sprintf("  SUBCYCLED: %.2f s total = %.2f s chemistry + %.2f s transport   (%d batches, %d calls, %d rejections)",
             RS.wall, RS.tchem, RS.ttrans, RS.st.batches, RS.ncalls, RS.st.rejects))
D.subcycle_report(RS.st; nglobal_steps = RG.nacc)

let (w, at) = worst_rel(RG.u, RS.u, 1:NC)
    check("trajectory over $NW macro steps", w, 2e-2)
    say(@sprintf("      (worst at species block %d, cell position %d)", at[1], at[2]))
    c1, c2 = control_pair(RS.u)
    wb, _ = worst_rel(RG.u, transpose_cells(RS.u, c1, c2), 1:NC)
    control("NEG CONTROL two cells transposed in the answer", wb, 2e-2)
end

# --------------------------------------------------------------------------- #
# CHECK 5 -- the PRODUCTION path. Checks 3 and 4 call `subcycle_chem` directly;
# this one goes through `macro_step`'s RESEACT_SUBCYCLE branch and the driver's
# real macro-stop lattice (GEOS-FP boundaries included, so it also exercises a
# `refresh_forcing` landing between two windows -- the case where a stale lane
# gather would show up).
# --------------------------------------------------------------------------- #
say("\n---- 5. the production path: forward_pass() through macro_step ----")
D.SUB_NEED[] = nothing; D.SUB_DTPROBE[] = D.DT0C; D.SUB_STATS[] = D.SubStats()
UE, CK, _, CN, TF = D.forward_pass(; record = false)
say(@sprintf("  forward_pass %.2f s over %d macro steps, J = %.15g",
             TF, length(CK), dot(D.WOBJ, UE)))
D.subcycle_report(D.SUB_STATS[])
let nbad = count(!isfinite, UE)
    nfail[] += (nbad == 0 ? 0 : 1)
    say(@sprintf("  %-46s %d of %d states  %s", "non-finite states at the end",
                 nbad, length(UE), nbad == 0 ? "PASS" : "FAIL"))
end

say("\n" * "="^78)
@printf("RESULT cellstep_speedup  %.3fx   (%.4g global cell-steps / %.4g subcycled)\n",
        Float64(RG.nacc) * NC / max(RS.st.cellsteps, 1), Float64(RG.nacc) * NC, RS.st.cellsteps)
@printf("RESULT lanestep_speedup  %.3fx   (padding waste %.1f%%)\n",
        Float64(RG.nacc) * NC / max(RS.st.lanesteps, 1),
        100 * (RS.st.lanesteps - RS.st.cellsteps) / max(RS.st.lanesteps, 1))
@printf("RESULT WALL_speedup      %.3fx   (chemistry %.2f s global vs %.2f s subcycled)\n",
        RG.tchem / max(RS.tchem, 1e-9), RG.tchem, RS.tchem)
@printf("RESULT calls             %d global vs %d subcycled (%.1fx more)\n",
        RG.ncalls, RS.ncalls, RS.ncalls / max(RG.ncalls, 1))
@printf("RESULT ms_per_call       %.3f global vs %.3f subcycled\n",
        1000 * RG.tchem / max(RG.ncalls, 1), 1000 * RS.st.t_dev / max(RS.st.calls, 1))
@printf("RESULT ladder_oneoff     %.1f s builds+jacobians+compiles (+%.1f s doc load)\n",
        sum(r.tbuild + r.tjac + r.tcomp for r in values(L.rungs)), L.tload)
say(nfail[] == 0 ? "RESULT verdict PASS -- every check passed and every control fired" :
                   "RESULT verdict FAIL -- $(nfail[]) checks/controls did not")
say("="^78)
say("SUBVERIFY_DONE")
nfail[] == 0 || exit(1)
