# ===========================================================================
# bucket_chem.jl -- PER-BUCKET ADAPTIVE chemistry stepping (Phase 1, forward).
# ===========================================================================
# The successor to the DYADIC level scheme (tools/subcycle_chem.jl /
# tools/level_subcycle.jl), and the design verdicts that force its shape:
#
#   * EQUAL substeps are dead. A level of 2^j equal steps must resolve a cell's
#     WORST instant over the window; the projected saving priced the INTEGRAL
#     of demand. Those differ by a median ~5x (max ~655x) at CONUS, and the
#     level scheme measured 0.023x end to end. What survives from that
#     experiment is the MACHINERY -- the capacity ladder, gather, permute --
#     which measured sound (dispatch cheaper per call than the global program,
#     gather+permute <1% of window wall). This file reuses it unchanged.
#   * What replaces the level is a PI CONTROLLER PER BUCKET: K equal-count
#     buckets of cells sorted by predicted demand, each bucket advancing the
#     window with its own adaptive dt using the SAME controller arithmetic as
#     the global path (pictrl_ros23 / host_adaptive!), accepting on the
#     NaN-safe MAX over its real lanes of the CELLWISE error. Offline replay
#     of recorded CONUS windows (stiffness_policy_replay.py) prices this at
#     ~2.9x cell-steps at K=16 and ~4x at K=64 against the actual global
#     controller (oracle 4.38x at K=16, per-cell ideal 6.09x).
#   * The predictor is `trend`: S_pred = clamp(S[m-1]^2 / max(S[m-2], 1), 1, Inf)
#     on the DECAYED realized demand, falling back to `prev` with one window of
#     history and to 1 with none. The trend form recovers most of the sunrise
#     sweep (the prev predictor's worst case); the decay
#     `need = max(realized, need/2)` is mandatory -- an undecayed max ratchets
#     and never releases a cell.
#   * The accept test is the max over REAL lanes, written NaN-safely: padding
#     lanes DUPLICATE real cells (a zero lane NaNs through log(PS/Pc)), an RMS
#     over the padded batch is diluted by the duplicates, and `worst <= 1.0`
#     is false for NaN so the reject branch must be `!(worst <= 1.0)`.
#   * A bucketed max-controller is STRICTER than the global RMS controller
#     (which under-resolves the stiffest cell ~30%), so trajectories differ
#     within tolerance and equivalence gates must be tolerance-based
#     (tools/diag/bucket_verify.jl), never bitwise against the global path.
#   * FORWARD ONLY. The adjoint tape records one (t, dt) sequence; a bucketed
#     window is K interleaved sequences. Same refusal as RESEACT_SUBCYCLE.
#
# CONTROLLER STATE IS NOT CARRIED ACROSS WINDOWS (dt_b, qold_b die with the
# bucket -- membership changes every window); the carried state is the per-CELL
# need vector. The per-bucket dt SEED comes from the prediction:
# dt_b = clamp(W / max_c S_pred[c], 1e-6, W).
#
# CAPACITIES. K equal-count buckets of NC cells have at most two distinct
# sizes (floor/ceil of NC/K), so of the RESEACT_BUCKET_LADDER menu only the
# rung(s) covering those sizes are actually reachable -- and only those are
# built and compiled (prepare-once, at driver start, with the driver's COPTS).
# Building the whole menu would buy nothing but compile time.
#
# This is a PLAIN INCLUDE into the driver's scope, like subcycle_chem.jl, and
# it INCLUDES subcycle_chem.jl to reuse its verified pieces (capacity builds,
# runner layout, reference geometry, lanes_in!/lanes_out!, batch_gather!,
# round-robin padding). subcycle_chem.jl's own behavior is untouched -- it is
# a preserved negative-result experiment -- and the two schemes are mutually
# exclusive at the knob level, so the include cannot double.
#
#   RESEACT_BUCKET          K buckets (default 0 = off; the driver dispatches)
#   RESEACT_BUCKET_KS       comma list of K values to prepare rungs for
#                           (default = RESEACT_BUCKET; a measurement script
#                           that A/Bs several K in one process lists them all)
#   RESEACT_BUCKET_LADDER   capacity menu (default 32,128,512,2048,8192)
# ===========================================================================

include(joinpath(REPO, "tools", "subcycle_chem.jl"))

const BUCKET_LADDER = sort(parse.(Int, split(get(ENV, "RESEACT_BUCKET_LADDER",
                                                 "32,128,512,2048,8192"), ',')))
const BUCKET_KS = sort(unique(parse.(Int, split(get(ENV, "RESEACT_BUCKET_KS",
                                                    string(max(BUCKETK, 1))), ','))))

# --------------------------------------------------------------------------- #
# Bucket geometry: equal-count split of NC sorted cells into K buckets.
# --------------------------------------------------------------------------- #
"Bucket boundaries: bucket b holds sorted positions (bounds[b]+1):bounds[b+1]."
bucket_bounds(NCl::Int, K::Int) = round.(Int, LinRange(0, NCl, min(K, NCl) + 1))

"The distinct ladder capacities the equal-count buckets of the given Ks need."
function bucket_caps_needed(NCl::Int, Ks::Vector{Int}, ladder::Vector{Int})
    caps = Int[]
    for K in Ks
        K >= 1 || error("bucket_caps_needed: K=$K; RESEACT_BUCKET/RESEACT_BUCKET_KS must be >= 1")
        bd = bucket_bounds(NCl, K)
        for b in 1:(length(bd) - 1)
            n = bd[b + 1] - bd[b]
            n == 0 && continue
            i = findfirst(>=(n), ladder)
            i === nothing && error("bucket_caps_needed: no ladder rung >= bucket size $n " *
                                   "(NC=$NCl, K=$K, ladder=$ladder); extend RESEACT_BUCKET_LADDER")
            push!(caps, ladder[i])
        end
    end
    return sort(unique(caps))
end

"""
    build_bucket_ladder(insp) -> SubLadder

The prepare-once side: build + compile ONLY the rungs the configured K values
can reach (see header). Reuses build_subcycle_ladder wholesale -- rungs built
there are cellwise ROS23 steps compiled with the driver's COPTS.
"""
build_bucket_ladder(insp) =
    build_subcycle_ladder(insp; caps = bucket_caps_needed(NC, BUCKET_KS, BUCKET_LADDER))

# --------------------------------------------------------------------------- #
# Stats.
# --------------------------------------------------------------------------- #
mutable struct BucketStats
    calls::Int          # compiled-program invocations (accepted + rejected attempts)
    accepts::Int
    rejects::Int
    nbuckets::Int
    lanesteps::Float64  # capacity x attempts, padding included
    cellsteps::Float64  # real cells x attempts (the honest per-cell cost)
    acc_cellsteps::Float64  # real cells x ACCEPTED steps (what the replay priced)
    worststeps::Int     # attempts taken by the busiest single bucket-window
    t_gather::Float64
    t_perm::Float64
    t_dev::Float64
    t_total::Float64
end
BucketStats() = BucketStats(0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

function Base.:+(a::BucketStats, b::BucketStats)
    BucketStats(a.calls + b.calls, a.accepts + b.accepts, a.rejects + b.rejects,
                a.nbuckets + b.nbuckets, a.lanesteps + b.lanesteps,
                a.cellsteps + b.cellsteps, a.acc_cellsteps + b.acc_cellsteps,
                max(a.worststeps, b.worststeps),
                a.t_gather + b.t_gather, a.t_perm + b.t_perm,
                a.t_dev + b.t_dev, a.t_total + b.t_total)
end

# --------------------------------------------------------------------------- #
# Cross-window state: the per-cell need memory (two windows of it, for `trend`).
# --------------------------------------------------------------------------- #
mutable struct BucketState
    need1::Union{Nothing,Vector{Float64}}   # decayed need after window m-1
    need2::Union{Nothing,Vector{Float64}}   # decayed need after window m-2
end
BucketState() = BucketState(nothing, nothing)

"Predicted per-cell step demand for the coming window, clamped >= 1."
function bucket_predict(st::BucketState, NCl::Int)
    st.need1 === nothing && return ones(Float64, NCl)
    st.need2 === nothing && return max.(st.need1, 1.0)
    return clamp.(st.need1 .^ 2 ./ max.(st.need2, 1.0), 1.0, Inf)
end

"The decaying-max update; an undecayed max RATCHETS (see header)."
function bucket_update!(st::BucketState, realized::Vector{Float64})
    st.need2 = st.need1
    st.need1 = st.need1 === nothing ? copy(realized) : max.(realized, st.need1 ./ 2)
    return st
end

# --------------------------------------------------------------------------- #
# One bucket: PI-controlled adaptive stepping over the whole window.
# --------------------------------------------------------------------------- #
"""
    run_bucket!(uo, BL, members, t0, t1, S_pred, realized, st)

Advance `members` (runner cell positions) across `[t0, t1]` on the smallest
rung that fits, with the SAME PI arithmetic as `host_adaptive!`/pictrl_ros23
but the accept test = NaN-safe max over real lanes of the cellwise error.
Writes the result into `uo` in place and accumulates each cell's REALIZED
step demand (the stiffness_diurnal metric) into `realized`.
"""
function run_bucket!(uo::Vector{Float64}, BL::SubLadder, members::Vector{Int},
                     t0::Float64, t1::Float64, S_pred::Vector{Float64},
                     realized::Vector{Float64}, st::BucketStats;
                     maxiters::Int = 20000)
    W = t1 - t0
    nreal = length(members)
    ci = findfirst(>=(nreal), BL.caps)
    ci === nothing && error("run_bucket!: no rung >= $nreal lanes in ladder $(BL.caps)")
    r = BL.rungs[BL.caps[ci]]
    C = r.C
    # lane -> runner cell position; padding repeats real lanes ROUND-ROBIN
    # (subcycle_chem.jl): a zero lane NaNs through log(PS/Pc), so every lane
    # must carry a real cell's inputs BEFORE anything evaluates.
    lanepos = Vector{Int}(undef, C)
    @inbounds for l in 1:C
        lanepos[l] = members[l <= nreal ? l : mod1(l - nreal, nreal)]
    end
    lane_cells = [BL.cells[c] for c in lanepos]
    tg = time(); batch_gather!(BL, r, lane_cells); st.t_gather += time() - tg
    tp = time(); uc = lanes_in!(copy(r.u0h), uo, r, lanepos); st.t_perm += time() - tp

    ctrl = RTI.pictrl_ros23()
    beta1 = ctrl.beta1; beta2 = ctrl.beta2
    invqmax = 1.0 / ctrl.qmax; invqmin = 1.0 / ctrl.qmin
    gamma = ctrl.gamma
    qsmin = ctrl.qsteady_min; qsmax = ctrl.qsteady_max
    qoldinit = ctrl.qoldinit

    # dt SEED from the prediction; controller state dies with the bucket.
    smax = 1.0
    @inbounds for c in members; smax = max(smax, S_pred[c]); end
    dt = clamp(W / smax, 1.0e-6, W)
    t = t0; qold = qoldinit
    nacc = 0; nrej = 0; iters = 0
    tlim = t1 - 1.0e-9
    while (t < tlim) && (iters < maxiters)
        dtc = min(dt, t1 - t)
        td = time()
        res = r.cstep(RX.ConcreteRArray(uc), r.th, RX.ConcreteRNumber(t),
                      RX.ConcreteRNumber(dtc))
        raw = Array(res[1]); ce = Array(res[3])
        st.t_dev += time() - td
        st.calls += 1
        st.lanesteps += C
        st.cellsteps += nreal
        # MAX over REAL lanes, NaN-safe: a NaN lane scores Inf, never "skip".
        worst = 0.0
        @inbounds for l in 1:nreal
            e = ce[l]
            worst = isfinite(e) ? max(worst, e) : Inf
        end
        reject = !(worst <= 1.0)             # NaN/Inf-safe reject test
        EEst = isfinite(worst) ? worst : 1.0e10   # host_adaptive!'s NaN mapping
        q11 = max(EEst, 1.0e-35)^beta1
        q = q11 / qold^beta2
        q = max(invqmax, min(invqmin, q / gamma))
        insteady = (qsmin <= q) & (q <= qsmax)
        qa = insteady ? one(q) : q
        if !reject
            # realized demand, accepted steps only (stiffness_diurnal.jl:110)
            @inbounds for l in 1:nreal
                e = ce[l]
                realized[lanepos[l]] +=
                    dtc / clamp(dtc * (1.0 / max(e, 1e-300))^(1 / 3), 1.0e-6, W)
            end
            uc = CLAMP[] ? max.(raw, 0.0) : raw
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
    iters >= maxiters && error(@sprintf(
        "run_bucket!: maxiters (%d) at t=%.3f of [%.3f, %.3f] for a bucket of %d cells",
        maxiters, t, t0, t1, nreal))
    tp2 = time(); lanes_out!(uo, uc, r, lanepos, nreal); st.t_perm += time() - tp2
    st.accepts += nacc; st.rejects += nrej; st.nbuckets += 1
    st.acc_cellsteps += Float64(nacc) * nreal
    st.worststeps = max(st.worststeps, nacc + nrej)
    return nacc, nrej
end

# --------------------------------------------------------------------------- #
# One macro window of chemistry, bucketed.
# --------------------------------------------------------------------------- #
"""
    bucket_window!(BL, u, t0, t1; K, state, lane_rng, verbose) -> (u_new, stats)

The drop-in replacement for `host_adaptive!(CROS, ...)` over one macro window.
`u` is not modified; the returned vector is a copy with the chemistry half
advanced. `state` carries the per-cell need memory across windows and is
UPDATED in place (decaying max).

`lane_rng` shuffles the LANE ORDER within each bucket (membership untouched).
It exists for the verification harness: a lane's arithmetic is lane-local and
padding cannot move a real lane, so the scattered result must be bitwise
identical under any within-bucket permutation -- MEMBERSHIP-changing
permutations legitimately change trajectories (each bucket's controller couples
its members through the max-error accept), so bitwise identity is only ever
claimed at fixed membership.
"""
function bucket_window!(BL::SubLadder, u::Vector{Float64}, t0::Float64, t1::Float64;
                        K::Int = BUCKETK, state::BucketState = BUCKET_STATE,
                        lane_rng = nothing, verbose::Bool = true)
    K >= 1 || error("bucket_window!: K=$K")
    st = BucketStats()
    ttot = time()
    S_pred = bucket_predict(state, NC)
    order = sortperm(S_pred)                    # stable: ties break by cell index
    bd = bucket_bounds(NC, K)
    realized = zeros(Float64, NC)
    uo = copy(u)
    persteps = Int[]
    for b in 1:(length(bd) - 1)
        members = order[(bd[b] + 1):bd[b + 1]]
        isempty(members) && continue
        lane_rng !== nothing && shuffle!(lane_rng, members)
        na, nr = run_bucket!(uo, BL, members, t0, t1, S_pred, realized, st)
        push!(persteps, na + nr)
    end
    bucket_update!(state, realized)
    st.t_total = time() - ttot
    if verbose
        say(@sprintf("    bucket window [%.0f,%.0f] K=%d: %d calls (%d acc + %d rej), steps/bucket %s, worst %d",
                     t0, t1, K, st.calls, st.accepts, st.rejects,
                     join(persteps, "/"), st.worststeps))
        say(@sprintf("      cell-steps %.4g (accepted %.4g)  lane-steps %.4g (padding %.1f%%)  wall %.2f s = dev %.2f + gather %.2f + perm %.2f",
                     st.cellsteps, st.acc_cellsteps, st.lanesteps,
                     100 * (st.lanesteps - st.cellsteps) / max(st.lanesteps, 1),
                     st.t_total, st.t_dev, st.t_gather, st.t_perm))
    end
    return uo, st
end

# --------------------------------------------------------------------------- #
# Driver-facing cross-window state and the whole-pass report.
# --------------------------------------------------------------------------- #
const BUCKET_STATE = BucketState()
const BUCKET_STATS = Ref(BucketStats())
const BUCKET_LAST  = Ref(BucketStats())

"Reset the cross-window predictor memory and the running totals."
function bucket_reset!()
    BUCKET_STATE.need1 = nothing
    BUCKET_STATE.need2 = nothing
    BUCKET_STATS[] = BucketStats()
    BUCKET_LAST[] = BucketStats()
    return nothing
end

"""
    bucket_report(st; nglobal_steps = 0)

`nglobal_steps` is the ACCEPTED global-dt chemistry step count of the SAME
windows when known (an A/B run); 0 omits the ratio rather than inventing one.
"""
function bucket_report(st::BucketStats; nglobal_steps::Int = 0)
    say(@sprintf("    bucketed: %d bucket-windows, %d program calls (%d accepted / %d rejected), worst bucket %d attempts",
                 st.nbuckets, st.calls, st.accepts, st.rejects, st.worststeps))
    if nglobal_steps > 0
        say(@sprintf("    cell-steps %.4g vs global %.4g  =>  %.2fx",
                     st.cellsteps, Float64(nglobal_steps) * NC,
                     Float64(nglobal_steps) * NC / max(st.cellsteps, 1)))
    else
        say(@sprintf("    cell-steps %.4g (accepted %.4g)", st.cellsteps, st.acc_cellsteps))
    end
    say(@sprintf("    lane-steps %.4g (padding waste %.1f%%)",
                 st.lanesteps, 100 * (st.lanesteps - st.cellsteps) / max(st.lanesteps, 1)))
    say(@sprintf("    wall %.2f s = device %.2f s + gather %.2f s + permute %.2f s + %.2f s other   (%.2f ms/call on the device)",
                 st.t_total, st.t_dev, st.t_gather, st.t_perm,
                 st.t_total - st.t_dev - st.t_gather - st.t_perm,
                 1000 * st.t_dev / max(st.calls, 1)))
end
