#!/usr/bin/env julia
# ===========================================================================
# bucket_verify.jl -- IS THE BUCKETED CHEMISTRY THE SAME INTEGRATION?
# ===========================================================================
# The merge gate for tools/bucket_chem.jl, at 6x6x8. One process, one build:
# the driver is included with RESEACT_BUCKET=4, which compiles the global-dt
# chemistry step (CROS) *and* the bucket rung(s); a TIGHT reference program
# (rtol/10) is compiled here on top of the same RHS.
#
# THE GATE (accuracy). Run N macro windows three ways from the same base
# point, transport included:
#   (a) LOCKSTEP  -- the global controller, host_adaptive!(CROS, ...)
#   (b) BUCKETED  -- K=4 through bucket_window!
#   (c) TIGHT     -- lockstep at rtol/10 (and atol/10), the reference
# and require the TOLERANCE CONTRACT: with the PRODUCTION controller's own
# atol/rtol (not the tight arm's),
#   err(arm) = max over cells of rms over species of
#              (u_arm - u_tight) / (atol + rtol * |u_tight|)
#   GATE: err(bucketed) <= 1.0
# i.e. the bucketed trajectory ends within one tolerance unit of the tight
# reference, in the norm the controllers actually control. err(lockstep) is
# printed for context and must ALSO be <= 1 -- if it is not, the reference
# setup itself is suspect and the run reports that rather than papering over
# it. The gate is TOLERANCE-based on purpose: bitwise against (a) would test
# a claim the design explicitly disowns (the bucketed max-controller is
# STRICTER than the global RMS controller, so trajectories differ).
#
# HISTORY, measured on this gate's first run (coordinator ruling): the
# original pointwise form |u_b - u_t| <= max(3 |u_a - u_t|, 10 atol) is
# DEFECTIVE -- the 3x LOCAL slack collapses at states where lockstep lands
# coincidentally close to the tight reference, and a bare-atol floor sits
# orders of magnitude below the rtol-dominated error unit for any state of
# ordinary magnitude, so a handful of states failed while the bucketed
# answer was at a few percent of ONE tolerance unit from tight. That
# pointwise comparison is kept below as a REPORTED DIAGNOSTIC, no longer
# gating.
#
# THE BITWISE CHECK (padding / lane-order leak). Run (b) twice with different
# random WITHIN-BUCKET lane permutations: membership identical, lane order and
# round-robin padding different. A lane's arithmetic is lane-local and a
# padding lane cannot move a real lane (capacity_chem.jl, measured exactly 0),
# so the scattered per-cell results must be BITWISE IDENTICAL (isequal).
# Membership-CHANGING permutations are deliberately not part of the bitwise
# claim: a bucket's controller couples its members through the max-error
# accept, so moving a tied cell between buckets legitimately changes dt
# sequences. The production path (deterministic stable sort, no shuffle) is
# ALSO compared bitwise against run (b) via forward_pass/macro_step, which
# exercises the driver dispatch on the way.
#
# Cell-step counts for (a) and (b) are printed; at 6x6x8 the stiffness spread
# is small so the ratio may be modest -- the GATE is accuracy, the CONUS job
# (bucket_conus.sbatch) is the speed claim.
#
# Every check fails LOUDLY: nonzero exit and a RESULT verdict FAIL line.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_ADJ_NMACRO", "6")
get!(ENV, "RESEACT_BUCKET", "4")
ENV["RESEACT_ADJ_UJITTER"] = "0"
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_SUBCYCLE"] = "0"
ENV["RESEACT_LABEL"] = "bucketverify"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics, Random, LinearAlgebra
RX = D.RX; RTI = D.RTI
say(s) = (println(s); flush(stdout))

const NW = D.NMACRO
const K  = D.BUCKETK
const NC = D.NC
const NS = D.NS

say("\n" * "="^78)
say(@sprintf("BUCKET VERIFY  grid=%sx%sx%s  NC=%d cells, NS=%d species  K=%d  ladder rungs %s  %d windows",
             ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"],
             NC, NS, K, string(D.BLADDER.caps), NW))
say("="^78)

# --------------------------------------------------------------------------- #
# The TIGHT reference program: the same RHS at rtol/10 (and atol/10).
# Tolerances are baked into the traced step, so a tighter controller is a
# separate compiled program, exactly like stiffness_diurnal.jl's cellwise one.
# --------------------------------------------------------------------------- #
ros_step_tight(u, th, t, dt) = RTI.ros23_step((uu, tt) -> D.gC(uu, th, tt), u, t, dt,
                                              NS, NC, D.MASKS, D.ATOL_C / 10, D.RTOL / 10;
                                              unrolled = true, jac = D.JACMODE,
                                              symjac = D.SYMJAC ? ((uu, tt) -> D.gJ(uu, th, tt)) : nothing)
const CROST = let t0 = time()
    UD = RX.ConcreteRArray(copy(D.UBASE))
    TD = RX.ConcreteRNumber(D.T0); DD = RX.ConcreteRNumber(D.DT0C)
    c = RX.@compile compile_options=D.COPTS ros_step_tight(UD, D.THC, TD, DD)
    say(@sprintf("  @compile ros_step_tight (rtol/10) %.1f s", time() - t0))
    c
end

# --------------------------------------------------------------------------- #
# The three arms. Each is the driver's own macro-stop lattice (STOPS covers
# exactly NW windows at RESEACT_ADJ_NMACRO=NW), transport included, from the
# same base point, so the only difference between arms is the chemistry half.
# --------------------------------------------------------------------------- #
function run_traj(mode::Symbol; lane_seed::Union{Nothing,Int} = nothing)
    D.refresh_forcing(D.T0)
    D.bucket_reset!()
    rng = lane_seed === nothing ? nothing : MersenneTwister(lane_seed)
    u = copy(D.UBASE); t = D.T0; dtT = D.DT0T; dtC = D.DT0C
    nacc = 0; nrej = 0; tchem = 0.0
    st = D.BucketStats()
    for tnext in D.STOPS
        tnext <= t + 1e-9 && continue
        uT, _, dtT, _, _ = D.host_adaptive!(D.CSSP, u, t, tnext, dtT,
                                            RTI.pictrl_ssprk43(), D.THT; clamp_nonneg = D.CLAMP[])
        tc = time()
        if mode === :lock
            u, _, dtC, na, nr = D.host_adaptive!(D.CROS, uT, t, tnext, dtC,
                                                 RTI.pictrl_ros23(), D.THC; clamp_nonneg = D.CLAMP[])
            nacc += na; nrej += nr
        elseif mode === :tight
            u, _, dtC, na, nr = D.host_adaptive!(CROST, uT, t, tnext, dtC,
                                                 RTI.pictrl_ros23(), D.THC; clamp_nonneg = D.CLAMP[])
            nacc += na; nrej += nr
        else
            u, s1 = D.bucket_window!(D.BLADDER, uT, t, tnext; K = K,
                                     state = D.BUCKET_STATE, lane_rng = rng)
            st = st + s1; nacc += s1.accepts; nrej += s1.rejects
        end
        tchem += time() - tc
        t = tnext
        if round(tnext; digits = 6) in D.FSTOPS
            D.refresh_forcing(tnext)
        end
    end
    return (; u, nacc, nrej, tchem, st)
end

nfail = Ref(0)

say("\n---- arm (a): LOCKSTEP (global controller) ----")
RA = run_traj(:lock)
say(@sprintf("  %d accepted / %d rejected chemistry steps, %.2f s chemistry wall  =>  %.4g cell-steps (accepted), %.4g attempted",
             RA.nacc, RA.nrej, RA.tchem, Float64(RA.nacc) * NC, Float64(RA.nacc + RA.nrej) * NC))

say("\n---- arm (c): TIGHT reference (rtol/10) ----")
RT = run_traj(:tight)
say(@sprintf("  %d accepted / %d rejected chemistry steps, %.2f s chemistry wall",
             RT.nacc, RT.nrej, RT.tchem))

say("\n---- arm (b): BUCKETED K=$K, lane permutation seed 1 ----")
RB = run_traj(:bucket; lane_seed = 1)
D.bucket_report(RB.st; nglobal_steps = RA.nacc)

say("\n---- arm (b'): BUCKETED K=$K, lane permutation seed 2 (bitwise control) ----")
RB2 = run_traj(:bucket; lane_seed = 2)

say("\n---- production path: forward_pass() through macro_step (no shuffle) ----")
D.refresh_forcing(D.T0); D.bucket_reset!()
UE, CK, _, CN, TF = D.forward_pass(; record = false)
say(@sprintf("  forward_pass %.2f s over %d macro steps, J = %.15g",
             TF, length(CK), dot(D.WOBJ, UE)))
let nbad = count(!isfinite, UE)
    nfail[] += (nbad == 0 ? 0 : 1)
    say(@sprintf("  non-finite states at the end: %d of %d  %s", nbad, length(UE),
                 nbad == 0 ? "PASS" : "FAIL"))
end

# --------------------------------------------------------------------------- #
# THE GATE: the tolerance contract, in the production controller's own norm.
# --------------------------------------------------------------------------- #
say("\n---- GATE: err(arm) = max_cells rms_species (u_arm - u_tight)/(atol + rtol|u_tight|) ----")
"""
    tol_err(ua, ut) -> (err, worst_cell)

Max over cells of the RMS over species of the deviation from the tight
reference, in PRODUCTION tolerance units (atol + rtol|u_tight| per state --
the same atol/rtol the production controller runs with, NOT the tight arm's).
A non-finite state scores Inf, never a skip.
"""
function tol_err(ua::Vector{Float64}, ut::Vector{Float64})
    worst = 0.0; wc = 0
    for c in 1:NC
        ss = 0.0
        for s in 1:NS
            i = (s - 1) * NC + c
            d = (ua[i] - ut[i]) / (D.ATOL_C + D.RTOL * abs(ut[i]))
            ss += d * d
        end
        r = sqrt(ss / NS)
        isfinite(r) || (r = Inf)
        if r > worst
            worst = r; wc = c
        end
    end
    return worst, wc
end
errA, cA = tol_err(RA.u, RT.u)
errB, cB = tol_err(RB.u, RT.u)
say(@sprintf("  err(lockstep) = %.4e  (worst at cell %d)   [context]", errA, cA))
say(@sprintf("  err(bucketed) = %.4e  (worst at cell %d)", errB, cB))
gate_ok = errB <= 1.0                       # NaN/Inf-safe: a non-finite err fails
gate_ok || (nfail[] += 1)
say("  GATE err(bucketed) <= 1.0: " * (gate_ok ? "PASS" : "FAIL"))
if !(errA <= 1.0)
    nfail[] += 1
    say("  SUSPECT REFERENCE: err(lockstep) > 1 -- the lockstep arm itself does not " *
        "meet the tolerance contract against the tight reference, so the gate " *
        "conclusion is unreliable; investigate the reference setup (reported, not papered over)")
end

# NEGATIVE CONTROL: the gate must be able to fail. Perturb ONE state of the
# bucketed answer by 100 tolerance units (RMS over NS species dilutes that to
# 100/sqrt(NS), still far over 1) and require the gate to see it.
let ubad = copy(RB.u)
    i = argmax(abs.(RT.u))
    ubad[i] += 100 * (D.ATOL_C + D.RTOL * abs(RT.u[i]))
    ebad, _ = tol_err(ubad, RT.u)
    fired = !(ebad <= 1.0)
    fired || (nfail[] += 1)
    say(@sprintf("  NEG CONTROL perturbed answer: err=%.3e  %s", ebad,
                 fired ? "FIRED" : "DID NOT FIRE -- the gate proves nothing"))
end

# --------------------------------------------------------------------------- #
# DIAGNOSTIC (not gating): the anatomy behind the err numbers. Three views:
#   * per-species breakdown at each arm's worst cell -- WHERE the units come
#     from (a near-zero species scores atol-units; a big one rtol-units);
#   * the per-cell err distribution -- is the worst cell a tail or typical;
#   * the tight arm's step-count ratio vs lockstep -- for an order-3 error
#     estimate, rtol/10 should cost ~10^(1/3) = 2.15x accepted steps; a ratio
#     far from that says the tight arm is not the reference it claims to be.
# --------------------------------------------------------------------------- #
say("\n---- DIAGNOSTIC (not gating): err anatomy ----")
function cell_anatomy(label::String, ua::Vector{Float64}, ut::Vector{Float64}, c::Int)
    say(@sprintf("  worst cell %d (%s): per species block, u_tight / u_arm / |dev| / dev in tolerance units", c, label))
    for s in 1:NS
        i = (s - 1) * NC + c
        unit = D.ATOL_C + D.RTOL * abs(ut[i])
        say(@sprintf("    s%02d  u_t=% .6e  u=% .6e  |d|=%.3e  %8.2f units%s",
                     s, ut[i], ua[i], abs(ua[i] - ut[i]), abs(ua[i] - ut[i]) / unit,
                     ut[i] == 0.0 ? "  (tight EXACTLY 0: clamped; unit = atol)" : ""))
    end
end
cell_anatomy("lockstep", RA.u, RT.u, cA)
cell_anatomy("bucketed", RB.u, RT.u, cB)
function tol_err_dist(label::String, ua::Vector{Float64}, ut::Vector{Float64})
    errs = Vector{Float64}(undef, NC)
    for c in 1:NC
        ss = 0.0
        for s in 1:NS
            i = (s - 1) * NC + c
            d = (ua[i] - ut[i]) / (D.ATOL_C + D.RTOL * abs(ut[i]))
            ss += d * d
        end
        e = sqrt(ss / NS)
        errs[c] = isfinite(e) ? e : Inf
    end
    q = quantile(errs, [0.5, 0.9, 0.99])
    say(@sprintf("  err/cell %-9s p50 %.3e  p90 %.3e  p99 %.3e  max %.3e  cells>1: %d of %d",
                 label, q[1], q[2], q[3], maximum(errs), count(>(1.0), errs), NC))
end
tol_err_dist("lockstep", RA.u, RT.u)
tol_err_dist("bucketed", RB.u, RT.u)
say(@sprintf("  tight arm integrity: %d accepted steps vs lockstep's %d = %.2fx (rtol/10 on an order-3 estimate predicts ~%.2fx)",
             RT.nacc, RA.nacc, RT.nacc / max(RA.nacc, 1), 10.0^(1 / 3)))

# --------------------------------------------------------------------------- #
# DIAGNOSTIC (not gating): the original pointwise-3x comparison, kept as a
# report line -- see the header for why it cannot gate.
# --------------------------------------------------------------------------- #
say("\n---- DIAGNOSTIC (not gating): pointwise |u_b-u_t| <= max(3 |u_a-u_t|, 10 (atol+rtol|u_t|)) ----")
let floor_(uti) = 10 * (D.ATOL_C + D.RTOL * abs(uti))
    nviol = 0; worstratio = 0.0; wat = 1
    for i in 1:length(RT.u)
        db = abs(RB.u[i] - RT.u[i]); da = abs(RA.u[i] - RT.u[i])
        bound = max(3 * da, floor_(RT.u[i]))
        (isfinite(db) && db <= bound) || (nviol += 1)
        r = db / bound
        if !isfinite(r) || r > worstratio
            worstratio = r; wat = i
        end
    end
    s = (wat - 1) ÷ NC + 1; c = (wat - 1) % NC + 1
    say(@sprintf("  %d of %d states over the pointwise bound; worst ratio %.3e at species block %d, cell %d",
                 nviol, length(RT.u), worstratio, s, c))
    say(@sprintf("  summary: max|u_lock-u_tight| = %.3e, max|u_bucket-u_tight| = %.3e",
                 maximum(abs.(RA.u .- RT.u)), maximum(abs.(RB.u .- RT.u))))
end

# --------------------------------------------------------------------------- #
# THE BITWISE CHECK: within-bucket lane permutation and padding must not move
# any cell by even one bit; nor may the production path's unshuffled order.
# --------------------------------------------------------------------------- #
say("\n---- BITWISE: within-bucket lane permutations (seed 1 vs seed 2 vs production) ----")
function bitdiff(a::Vector{Float64}, b::Vector{Float64})
    nbad = 0
    for c in 1:NC
        any(s -> !isequal(a[(s - 1) * NC + c], b[(s - 1) * NC + c]), 1:NS) && (nbad += 1)
    end
    return nbad
end
for (label, a, b) in (("seed1 vs seed2", RB.u, RB2.u),
                      ("seed1 vs production forward_pass", RB.u, UE))
    nbad = bitdiff(a, b)
    ok = nbad == 0
    ok || (nfail[] += 1)
    say(@sprintf("  %-38s %d of %d cells differ  %s", label, nbad, NC,
                 ok ? "PASS (bitwise identical)" : "FAIL"))
end
# NEGATIVE CONTROL: two cells transposed must be seen.
let v = copy(RB.u)
    for s in 1:NS
        b = (s - 1) * NC
        v[b + 1], v[b + 2] = v[b + 2], v[b + 1]
    end
    nbad = bitdiff(RB.u, v)
    fired = nbad > 0
    fired || (nfail[] += 1)
    say(@sprintf("  NEG CONTROL two cells transposed: %d cells differ  %s", nbad,
                 fired ? "FIRED" : "DID NOT FIRE"))
end

# --------------------------------------------------------------------------- #
# The scoreboard.
# --------------------------------------------------------------------------- #
say("\n" * "="^78)
@printf("RESULT err_lockstep        %.4e  (tolerance units vs tight; context)\n", errA)
@printf("RESULT err_bucketed        %.4e  (tolerance units vs tight; GATE <= 1.0)\n", errB)
@printf("RESULT cellsteps_lockstep  %.4g accepted (%.4g attempted; %d acc / %d rej)\n",
        Float64(RA.nacc) * NC, Float64(RA.nacc + RA.nrej) * NC, RA.nacc, RA.nrej)
@printf("RESULT cellsteps_bucketed  %.4g attempted (%.4g accepted; %d calls, %d rejected)\n",
        RB.st.cellsteps, RB.st.acc_cellsteps, RB.st.calls, RB.st.rejects)
@printf("RESULT cellstep_ratio      %.3fx (lockstep accepted / bucketed attempted)\n",
        Float64(RA.nacc) * NC / max(RB.st.cellsteps, 1))
@printf("RESULT chem_wall           lockstep %.2f s vs bucketed %.2f s\n", RA.tchem, RB.tchem)
say(nfail[] == 0 ? "RESULT verdict PASS -- gate met, bitwise identical, controls fired" :
                   "RESULT verdict FAIL -- $(nfail[]) checks/controls did not pass")
say("="^78)
say("BUCKETVERIFY_DONE")
nfail[] == 0 || exit(1)
