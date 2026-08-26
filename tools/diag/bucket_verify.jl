#!/usr/bin/env julia
# ===========================================================================
# bucket_verify.jl -- IS THE BUCKETED CHEMISTRY THE SAME INTEGRATION?
# ===========================================================================
# The merge gate for tools/bucket_chem.jl, at 6x6x8. One process, one build:
# the driver is included with RESEACT_BUCKET=4, which compiles the global-dt
# chemistry step (CROS) *and* the bucket rung(s); a TIGHT reference program
# (rtol/10) is compiled here on top of the same RHS.
#
# THE GATE (accuracy, per the design spec). Run N macro windows three ways
# from the same base point, transport included:
#   (a) LOCKSTEP  -- the global controller, host_adaptive!(CROS, ...)
#   (b) BUCKETED  -- K=4 through bucket_window!
#   (c) TIGHT     -- lockstep at rtol/10 (and atol/10), the reference
# and require, for EVERY state,
#   |u_b - u_t| <= max(3 * |u_a - u_t|, 10 * atol)
# The bucketed max-controller is STRICTER than the global RMS controller, so
# (b) may differ from (a) -- but it must not sit further from the tight
# reference than lockstep does (3x slack for controller-path divergence).
# The gate is TOLERANCE-based on purpose: bitwise against (a) would be testing
# a claim the design explicitly disowns.
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
# THE GATE.
# --------------------------------------------------------------------------- #
say("\n---- GATE: |u_bucket - u_tight| <= max(3 |u_lockstep - u_tight|, 10 atol) per state ----")
function gate(ub::Vector{Float64}, ua::Vector{Float64}, ut::Vector{Float64})
    floor_ = 10 * D.ATOL_C
    nviol = 0; worstratio = 0.0; wat = 0
    for i in 1:length(ut)
        db = abs(ub[i] - ut[i]); da = abs(ua[i] - ut[i])
        bound = max(3 * da, floor_)
        ok = isfinite(db) && db <= bound
        ok || (nviol += 1)
        r = db / bound
        if !isfinite(r) || r > worstratio
            worstratio = r; wat = i
        end
    end
    return nviol, worstratio, wat
end
nviol, wr, wat = gate(RB.u, RA.u, RT.u)
let s = (wat - 1) ÷ NC + 1, c = (wat - 1) % NC + 1
    say(@sprintf("  %d of %d states violate; worst |u_b-u_t|/bound = %.3e at species block %d, cell %d",
                 nviol, length(RT.u), wr, s, c))
    say(@sprintf("    there: u_t=%.6e  u_lock=%.6e  u_bucket=%.6e  (10*atol floor %.1e)",
                 RT.u[wat], RA.u[wat], RB.u[wat], 10 * D.ATOL_C))
end
say(@sprintf("  summary: max|u_lock-u_tight| = %.3e, max|u_bucket-u_tight| = %.3e",
             maximum(abs.(RA.u .- RT.u)), maximum(abs.(RB.u .- RT.u))))
gate_ok = nviol == 0
gate_ok || (nfail[] += 1)
say("  GATE " * (gate_ok ? "PASS" : "FAIL"))

# NEGATIVE CONTROL: the gate must be able to fail. Perturb the bucketed answer
# by 100x the bound at its largest state and require a violation.
let ubad = copy(RB.u)
    i = argmax(abs.(RT.u))
    ubad[i] += 100 * max(3 * abs(RA.u[i] - RT.u[i]), 10 * D.ATOL_C)
    nv, _, _ = gate(ubad, RA.u, RT.u)
    fired = nv > 0
    fired || (nfail[] += 1)
    say("  NEG CONTROL perturbed answer: " *
        (fired ? "FIRED" : "DID NOT FIRE -- the gate proves nothing"))
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
