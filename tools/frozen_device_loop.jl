# ===========================================================================
# frozen_device_loop.jl -- the DIFFERENTIATED map, on the device.
# ===========================================================================
# Included by tools/adjoint_gradient.jl when RESEACT_ADJ_DEVLOOP is not 0.
# It replaces, and ONLY replaces, the two HOST loops that walk a macro step's
# inner steps one XLA call at a time:
#
#   replay_fixed        n_inner step calls  ->  ONE frozen_run call per half
#   backward_stage!     n_inner VJP  calls  ->  ONE frozen_vjp call per half
#
# WHY, AND WHAT IT IS NOT FOR. The host round-trip costs ~0.2% of the loop on
# XLA:CPU, where the "device" is the same memory. On an A100/H100 each of the
# ~28,000 accepted inner steps of a 48 h window becomes a device sync plus two
# transfers, and that design dominates. This change is about PORTABILITY and
# program structure. Expect no CPU speedup -- expect a small regression, because
# a masked iteration still evaluates the body, so the loop costs `cap` steps
# where the host loop cost `nlive`. A flat or slightly worse wall time here is
# the expected outcome, not a failure.
#
# WHAT MAKES IT LEGITIMATE. The map the discrete adjoint differentiates is
# ALREADY a fixed-step composition with a known trip count: the forward pass
# records the accepted (t, dt) of every inner step and `replay_fixed` replays
# THAT sequence with the controller switched off (see its header for why
# re-deriving it does not reproduce the forward pass). So the trip count is not
# guessed and not overshot -- it is READ OFF THE TAPE, and `cap` only rounds it
# up to a bucket. The FORWARD pass keeps its adaptive `stablehlo.while`
# (`RxTracedIntegrator.adaptive_solve`); nothing differentiates it.
#
# THE CLAMP STAYS OUTSIDE THE STEP. `adaptive_solve(clamp_nonneg=true)` clamps
# every ACCEPTED state, after the step and after EEst was formed from the
# unclamped state. `frozen_*_body(..., clamp_nonneg=true)` wraps the step the
# same way and `RxTracedIntegrator._frozen_clamp0` is a select on `raw > 0`, so
# its derivative is the identical 0/1 diagonal the host driver applies as
# `lam .*= (raw .> 0)` -- placement and value both preserved.
#
# CAP BUCKETING. `cap` is the one compile-time constant, so a distinct `cap` is
# a distinct compiled program. Bucketing to the next power of two bounds the
# wasted (masked) iterations at 2x while keeping the number of compiles to the
# two or three buckets a window actually spans. RESEACT_ADJ_DEVCAP=N forces a
# single bucket (one compile per program, maximal waste); RESEACT_ADJ_DEVCAP_MIN
# floors it, because a transport half is often only 5-10 steps.
#
# Env:
#   RESEACT_ADJ_DEVLOOP     0 = off (host loop, the default), 1 = device loop,
#                           both = run BOTH sweeps in ONE process and compare
#                           (the gradient-agreement gate; identical base point,
#                           identical tape, so nothing but the loop differs)
#   RESEACT_ADJ_DEVCAP      force a single cap for every half (0 = bucket)
#   RESEACT_ADJ_DEVCAP_MIN  floor on the bucket (default 4)
#   RESEACT_ADJ_DEVDUMP     1 = dump the differentiated module at two caps and
#                           report its stablehlo.while count and size
# ===========================================================================

const DEVCAP_FIX = parse(Int, get(ENV, "RESEACT_ADJ_DEVCAP", "0"))
const DEVCAP_MIN = parse(Int, get(ENV, "RESEACT_ADJ_DEVCAP_MIN", "4"))
DEVCAP_FIX >= 0 || error("RESEACT_ADJ_DEVCAP must be >= 0")
devcap(n::Int) = DEVCAP_FIX > 0 ? DEVCAP_FIX : max(DEVCAP_MIN, nextpow(2, max(n, 1)))

# The step body of one half, as a HOST-only closure: theta is an explicit
# argument, never a capture, so no traced value hides inside a callee of the
# loop body (see `frozen_run`'s docstring). Rebuilt on demand rather than
# cached, because CLAMP[] can flip mid-run (the driver's no-clamp retry).
function dev_body(half::Symbol)
    if half === :T
        return RTI.frozen_ssprk43_body(gT, ATOL_T, RTOL; clamp_nonneg = CLAMP[])
    elseif half === :C
        return RTI.frozen_ros23_body(gC, NS, NC, MASKS, ATOL_C, RTOL;
                                     jac = JACMODE, gj = gJ, clamp_nonneg = CLAMP[])
    end
    error("dev_body: half must be :T or :C, got $half")
end

const DEVPROGS = Dict{Tuple{Symbol,Symbol,Int,Bool},Any}()
dev_th(half::Symbol) = half === :T ? THT : THC

"""
    dev_prog(kind, half, cap)

The compiled frozen-grid program for `kind in (:out, :vjp)` of `half in (:T, :C)`
at trip count `cap`, compiled on first use and cached. The clamp is part of the
key: it is baked into the body, and the driver's no-clamp retry flips it.
"""
function dev_prog(kind::Symbol, half::Symbol, cap::Int)
    key = (kind, half, cap, CLAMP[])
    haskey(DEVPROGS, key) && return DEVPROGS[key]
    bodyf = dev_body(half)
    TH = dev_th(half)
    # Shapes only -- every value here is a runtime operand.
    TSR = RX.ConcreteRArray(zeros(Float64, cap))
    DTSR = RX.ConcreteRArray(ones(Float64, cap))
    NLR = RX.ConcreteRNumber(cap)
    prog = if kind === :out
        fo_ = (u, th, ts, dts, nl) -> RTI.frozen_run(bodyf, u, th, ts, dts, nl, cap)
        timed_compile("dev.$(half).out[$cap]",
            () -> RX.@compile compile_options=COPTS fo_(U_R, TH, TSR, DTSR, NLR))
    elseif kind === :vjp
        fv_ = (u, th, lam, ts, dts, nl) ->
            RTI.frozen_vjp(bodyf, u, th, lam, ts, dts, nl, cap; active_bufs = BUFGRAD)
        timed_compile("dev.$(half).vjp[$cap]",
            () -> RX.@compile compile_options=COPTS fv_(U_R, TH, LAM_R, TSR, DTSR, NLR))
    else
        error("dev_prog: kind must be :out or :vjp, got $kind")
    end
    DEVPROGS[key] = prog
    return prog
end

# FRESH `ConcreteRArray`s ON EVERY CALL BELOW, AND THAT IS LOAD-BEARING.
# `RX.@compile` DONATES the input buffers a program does not return, so handing
# the same `ConcreteRArray` to two calls gives the second one a CORRUPTED array
# -- silently, and only from the second call on, so the first measurement in a
# process is right and everything after it drifts. `step_call` / `vjp_call` above
# upload fresh arrays for the same reason, which is why the host path never sees
# it. Do not "optimise" these uploads into a reused buffer: it costs 0.1 ms and
# it is the difference between a gradient and a plausible wrong number.
# (Measured the hard way: an early `tools/diag/frozen_loop_smoke.jl` reused one
# input array and reported a 1.7e-1 gradient error in `frozen_vjp` that was
# entirely its own. See that file's header.)
#
# The recorded (t, dt) sequence as the two [cap] runtime tensors the loop reads.
# PADDING IS THE LAST LIVE (t, dt), never zero: a Rosenbrock attempt forms
# W = I/(gamma*dt), so dt = 0 is a division by zero inside a masked iteration
# whose result is discarded -- but whose NaN would still be differentiated (the
# select routes an exactly zero cotangent into it, and 0 * Inf is NaN). The last
# live step from the frozen end state is an ordinary, finite step instead.
function dev_pack(seq::StepSeq, cap::Int)
    n = length(seq)
    n >= 1 || error("dev_pack: empty sequence")
    n <= cap || error("dev_pack: sequence of $n steps does not fit cap=$cap")
    ts = Vector{Float64}(undef, cap); dts = Vector{Float64}(undef, cap)
    for i in 1:cap
        j = min(i, n)
        ts[i] = seq[j][1]; dts[i] = seq[j][2]
    end
    return ts, dts
end

"""
    devloop_replay(half, u, seq; TH) -> u_end

The frozen forward composition of one half, in ONE device call. Same map, same
step algebra and same clamp placement as `replay_fixed`'s host loop -- it just
does not come back to Julia between steps, and it records no inner tape (the
device VJP below does not need one).
"""
function devloop_replay(half::Symbol, u::Vector{Float64}, seq::StepSeq;
                        TH = dev_th(half))
    isempty(seq) && return copy(u)
    cap = devcap(length(seq))
    prog = dev_prog(:out, half, cap)
    ts, dts = dev_pack(seq, cap)
    tu = time()
    UD = RX.ConcreteRArray(u); TSD = RX.ConcreteRArray(ts); DTD = RX.ConcreteRArray(dts)
    NLD = RX.ConcreteRNumber(length(seq))
    t0 = time(); _exec_add!("$(half).devreplay.upload", t0 - tu)
    r = prog(UD, TH, TSD, DTD, NLD)
    t1 = time(); out = Array(r)
    _exec_add!("$(half).devreplay.exec", t1 - t0)
    _exec_add!("$(half).devreplay.read", time() - t1)
    _exec_add!("$(half).devreplay", time() - tu)
    return out
end

"""
    devloop_vjp(half, u, lam, seq; TH) -> (lambda_in, dJ/dp aligned with PNAMES)

Reverse mode over the WHOLE half in ONE device call: the chain rule over the
half's inner steps, and the summation of the parameter cotangent over them,
both happen inside XLA. The host sees one upload and one readback per half
instead of one per inner step.
"""
function devloop_vjp(half::Symbol, u::Vector{Float64}, lam::Vector{Float64},
                     seq::StepSeq; TH = dev_th(half))
    isempty(seq) && return copy(lam), zeros(Float64, length(PNAMES))
    cap = devcap(length(seq))
    prog = dev_prog(:vjp, half, cap)
    ts, dts = dev_pack(seq, cap)
    tu = time()
    UD = RX.ConcreteRArray(u); LD = RX.ConcreteRArray(lam)
    TSD = RX.ConcreteRArray(ts); DTD = RX.ConcreteRArray(dts)
    NLD = RX.ConcreteRNumber(length(seq))
    t0 = time(); _exec_add!("$(half).devvjp.upload", t0 - tu)
    r = prog(UD, TH, LD, TSD, DTD, NLD)
    t1 = time()
    lin = Array(r[1])
    t2 = time()
    gp = r[2].p                          # r[2].bufs, if active, is dJ/d(GEOS-FP)
    g = Float64[Float64(getfield(gp, k)) for k in PNAMES]
    t3 = time()
    _exec_add!("$(half).devvjp.exec", t1 - t0)
    _exec_add!("$(half).devvjp.read.lam", t2 - t1)
    _exec_add!("$(half).devvjp.read.p", t3 - t2)
    _exec_add!("$(half).devvjp", t3 - tu)
    return lin, g
end

# ---- the whole-window frozen replay, device version --------------------------
# What `fdtape` differences. Deliberately routed through the SAME compiled
# device programs the sweep differentiates, so the acceptance test stays
# confounder-free: the discrete adjoint IS the derivative of this map.
function devloop_replay_window(THT_, THC_)
    u = copy(UBASE)
    for ck in CKPTS
        refresh_forcing_if_needed(ck.epoch)
        u = devloop_replay(:T, u, ck.seqT; TH = THT_)
        u = devloop_replay(:C, u, ck.seqC; TH = THC_)
    end
    return dot(WOBJ, u)
end

# ---- gate 4: is the differentiated module a while, and is it O(1) in cap? ----
# Dumps the UNOPTIMISED module of the reverse program at two caps and counts
# `stablehlo.while` regions and lines. An unrolled body would grow with cap; a
# while region does not.
function devloop_dump_structure(half::Symbol, caps = (32, 128))
    say("\n---- DEVLOOP structure: the differentiated module of half $(half) ----")
    bodyf = dev_body(half)
    TH = dev_th(half)
    for cap in caps
        TSR = RX.ConcreteRArray(zeros(Float64, cap))
        DTSR = RX.ConcreteRArray(ones(Float64, cap))
        NLR = RX.ConcreteRNumber(cap)
        fv_ = (u, th, lam, ts, dts, nl) ->
            RTI.frozen_vjp(bodyf, u, th, lam, ts, dts, nl, cap; active_bufs = BUFGRAD)
        try
            t0 = time()
            s = repr(RX.@code_hlo optimize=false fv_(U_R, TH, LAM_R, TSR, DTSR, NLR))
            nw = length(collect(eachmatch(r"stablehlo\.while\b", s)))
            say(@sprintf("  cap=%-5d %4d stablehlo.while  %9d lines  %8.2f MB of MLIR  (%.1f s)  %s",
                         cap, nw, count(==('\n'), s) + 1, sizeof(s) / 2^20, time() - t0,
                         nw > 0 ? "NOT unrolled" : "UNROLLED -- size O(cap)"))
        catch e
            say("  cap=$cap  code_hlo failed: " * first(split(sprint(showerror, e), '\n')))
        end
    end
end

# ---- the faithfulness test the host path gets from `tapes_for` ---------------
# The frozen replay must land on the state the forward pass actually reached.
# It is the same check, on the same quantity, at the same place -- the device
# loop just produces the end state in one call instead of n_inner.
function devloop_checkpoint_check(k::Int, uend::Vector{Float64})
    ref = k < length(CKPTS) ? CKPTS[k + 1].u : UEND
    isempty(ref) && return nothing
    tc = time()
    d = maximum(abs.(uend .- ref) ./ max.(abs.(ref), 1e-30))
    _exec_add!("ckpt-check", time() - tc)
    global REPLAY_MAXREL = max(REPLAY_MAXREL, d)
    d < 1e-6 || error("device frozen replay of macro step $k lands $(d) relative away " *
                      "from the checkpointed end state -- the loop is not replaying the " *
                      "trajectory the forward pass took")
    return nothing
end

# The host path attributes a non-finite lambda to the exact inner step it
# started on (`BADREC`). The device path cannot: the inner states never come
# back. What it CAN do is refuse to report a poisoned gradient silently, so the
# first macro step at which lambda or the parameter cotangent goes non-finite is
# named and counted here.
DEV_NONFINITE = 0
function devloop_finite_check(k::Int, lam::Vector{Float64}, gC_, gT_)
    nl = count(!isfinite, lam)
    ng = count(!isfinite, gC_) + count(!isfinite, gT_)
    (nl == 0 && ng == 0) && return nothing
    global DEV_NONFINITE += 1
    # ONE string literal: `@sprintf` validates its format at MACRO EXPANSION
    # time, so a concatenated `"..." * "..."` format is a LoadError raised when
    # this file is included -- not when this branch runs. (Same trap the fdtape
    # stage records in adjoint_gradient.jl.)
    DEV_NONFINITE == 1 && say(@sprintf("  FIRST NON-FINITE (device loop): macro step %d -- lambda %d/%d, grad_theta %d/%d. The device path cannot attribute this to an inner step; re-run that window with RESEACT_ADJ_DEVLOOP=0 to get the per-step PROBE.",
        k, nl, length(lam), ng, 2 * length(PNAMES)))
    return nothing
end

"""
    devloop_precompile()

Compile every frozen-grid program the backward sweep will need, BEFORE it is
timed. The forward pass has already recorded each macro step's accepted (t, dt)
sequence, so the exact set of `(kind, half, cap)` buckets is known here -- there
is nothing adaptive left to discover. Compiling them lazily inside the sweep is
the same total work but it lands in the sweep's wall time, where it is
indistinguishable from execution: the first run of this file put 491 s of
compile into a 508 s "backward sweep" and made the device arm look 28x slower
than it is.
"""
function devloop_precompile()
    need = Set{Tuple{Symbol,Symbol,Int}}()
    capsT = Int[]; capsC = Int[]
    for ck in CKPTS
        if !isempty(ck.seqT)
            c = devcap(length(ck.seqT)); push!(capsT, c)
            push!(need, (:out, :T, c)); push!(need, (:vjp, :T, c))
        end
        if !isempty(ck.seqC)
            c = devcap(length(ck.seqC)); push!(capsC, c)
            push!(need, (:out, :C, c)); push!(need, (:vjp, :C, c))
        end
    end
    isempty(need) && return nothing
    hist(v) = join((@sprintf("%d x%d", c, count(==(c), v)) for c in sort(unique(v))), ", ")
    say(@sprintf("  frozen-grid caps: transport {%s}, chemistry {%s} over %d macro steps",
                 hist(capsT), hist(capsC), length(CKPTS)))
    # ONE string literal -- `@sprintf` validates its format at macro expansion,
    # so a concatenated format is a LoadError when this file is INCLUDED.
    say(@sprintf("  wasted (masked) iterations: transport %.1f%%, chemistry %.1f%% -- the price of bucketing, and on CPU it is real work",
                 100 * (1 - sum(length(ck.seqT) for ck in CKPTS) / max(sum(capsT), 1)),
                 100 * (1 - sum(length(ck.seqC) for ck in CKPTS) / max(sum(capsC), 1))))
    say("  ---- compiling $(length(need)) frozen-grid programs (one stablehlo.while each) ----")
    for k in sort(collect(need); by = x -> (x[2], x[1], x[3]))
        dev_prog(k[1], k[2], k[3])
    end
    return nothing
end
