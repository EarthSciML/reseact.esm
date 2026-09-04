#!/usr/bin/env julia
# ===========================================================================
# p5_vjp_time.jl -- ros_vjp / ssp_vjp wall time: driver default vs default +
#                   an extra `excluded_passes` set, with a bit-for-bit gate on
#                   BOTH VJP outputs (lambda_in and the p-gradient).
# ===========================================================================
# One process, one build. Each program is compiled twice -- the driver's COPTS
# and COPTS plus P5_EXTRA -- and timed interleaved A/B/A/B on the same inputs
# with FRESH buffers per call (donation). The gate compares Array(r[1]) and
# every runtime-scalar component of r[2].p between the two compiles; "win"
# means faster AND identical, or faster with a difference at roundoff that is
# reported as a number rather than hidden.
#
# Every CompileOptions keeps sync=true and xla_cpu_prefer_vector_width=128.
#
#   RESEACT_NLON/NLAT/NLEV   grid (6x6x8 in a session; CONUS via sbatch only)
#   P5_EXTRA                 comma-separated EXTRA pattern base-names
#   P5_PROGRAMS              default "ros_vjp,ssp_vjp"; may add ros_step,ssp_step
#   P5_XLAOPT                extra XLA debug options for arm B, "k=v;k=v"
#   P5_ARMB=bufgrad          arm A = forcing buffers ACTIVE (old VJP), arm B = p only
#   P5_REPS                  timing reps per arm per pass (default 40)
#
# RESULTS (Reactant v0.2.280, driver default excluded_passes, 2026-09-04):
#
#   program    6x6x8 (288 cells)          CONUS 13x7x72 (slurm 10363325, ccc0234)
#   ros_step    3.5 ms  cpu/wall 3.3       38.2 ms  cpu/wall 5.5
#   ros_vjp    13.8 ms  cpu/wall 4.9       87.5 ms  cpu/wall 6.4   (2.3x its step)
#   ssp_step    2.0 ms                     14.0 ms  cpu/wall 6.8
#   ssp_vjp    17.2 ms  cpu/wall 5.0      305.1 ms  cpu/wall 7.9   (21.8x its step)
#
#   * The XLA:CPU concurrency-optimized scheduler flag is a no-op on every
#     program (1.04x / 0.98x at 6x6x8), and the VJPs keep MORE cores busy than
#     the primal: the reverse passes are not starved of cores, they do more
#     work. The pass-exclusion knob is exhausted for them (p5_vjp_census.jl).
#   * Per 300 s window at CONUS (45.3 chemistry + 3.2 transport steps): chem
#     VJPs 3.96 s, transport VJPs 0.98 s. The chemistry VJP at 2.3x its step
#     is ordinary reverse-mode overhead; the TRANSPORT VJP at 21.8x is the
#     anomaly. Its module carries 628 scatters (the reverse of the stencil's
#     619 gathers); a model-free micro-benchmark of those shapes at CONUS
#     scale (280k-element buffers, 9.8k / 59k unique-index updates) prices
#     them at 0.077 / 0.46 ms each, ~116 ms of the 305 ms. The scatter
#     patterns Enzyme-JAX has (scatter_indices_are_unique,
#     scatter_to_dynamic_update_slice) already run by default. The lever is
#     upstream of the pipeline: emit stencil shifts as affine slices (whose
#     reverse is a pad, fusable) instead of dense-index gathers (whose reverse
#     is a serial scatter-add) -- an EarthSciAST emitter change, not a pass.
#   * THE LEVER (P5_ARMB=bufgrad; the finding is in p5_vjp_dump.jl): making
#     the forcing buffers Const in the VJP -- `active_bufs=false`, the driver
#     default via RESEACT_ADJ_BUFGRAD=0. At 6x6x8 (logs/p5-time-local-bufgrad.out):
#       ros_vjp  12.25 -> 8.54 ms  (1.44x)  lambda AND p-gradient bit-for-bit
#       ssp_vjp  17.16 -> 14.81 ms (1.16x)  lambda bit-for-bit, p-gradient
#                                            max rel 3.0e-16 (one reassociated ulp)
#     CONUS arm (slurm 10364305, ccc0232, busier than 10363325's ccc0234):
#       ros_vjp  98.8 -> 93.0 ms  (1.06x)  lambda AND p-gradient bit-for-bit
#       ssp_vjp 393.1 -> 397.2 ms (0.99x)  p-gradient max rel 1.7e-16
#     So at CONUS the forcing-buffer gradient is ~6% of the chemistry VJP and
#     nothing of the transport VJP: the 6x6x8 copy census does not scale with
#     the buffers the way the byte count suggested (a CONUS dump would say
#     why; not run). Kept as the driver default because the gradient is
#     identical to every printed digit (tools/diag/p5_bufgrad_pair.sbatch,
#     slurm 10364306, P5PAIR_IDENTICAL) and the work removed is real.
#   * WHAT IS LEFT, priced: at CONUS the chemistry VJP is 2.3x its step --
#     ordinary reverse-mode overhead, so its remaining lever is the primal's
#     (per-element codegen, fewer steps). The transport VJP at ~22x its step
#     is the anomaly: ~116 ms of its 305 ms are the 628 scatter-adds that
#     reverse the stencil's dense-index gathers (micro-benchmarked), and the
#     6x6x8 dump shows 311 copy-insertion copies; the fix is upstream of the
#     pipeline (emit stencil shifts as affine slices, whose reverse is a pad).
#   * The 48 h run (slurm 10359755, same node) averaged 0.1847 s/VJP over
#     26,114 chemistry + 1,859 transport VJPs; the quiet-node figures above
#     predict 0.102 s. The run shared ccc0234 with a 20-core job, so its VJP
#     figure carries ~1.8x contention (the forward pass ~1.3x): on a quiet
#     node the window costs ~8.5 s, not 13.3 s.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p5time")
include(joinpath(@__DIR__, "_env.jl"))
using Printf, Statistics
say(s) = (println(s); flush(stdout))
const EXTRA = String.(filter(!isempty, strip.(split(get(ENV, "P5_EXTRA", ""), ','))))
const PROGS = String.(split(get(ENV, "P5_PROGRAMS", "ros_vjp,ssp_vjp"), ','))
const REPS = parse(Int, get(ENV, "P5_REPS", "40"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
const BASEX = collect(String, D.EXCLP)
const FULLX = unique(vcat(BASEX, EXTRA))
# Arm B may also carry extra XLA debug options: P5_XLAOPT="name=value;name=value"
# (Bool/Int/String parsed by shape). The race workaround stays in every arm.
function parse_xlaopt(s)
    kv = Pair{Symbol,Any}[]
    for item in filter(!isempty, strip.(split(s, ';')))
        k, v = strip.(split(item, '='; limit = 2))
        val = v in ("true", "false") ? parse(Bool, v) :
              occursin(r"^-?\d+$", v) ? parse(Int, v) : String(v)
        push!(kv, Symbol(k) => val)
    end
    return kv
end
const XLAOPT = parse_xlaopt(get(ENV, "P5_XLAOPT", ""))
say("P5_VJP_TIME grid=$(D.NC) cells   default excl=[" * join(BASEX, ",") * "]   +extra=[" * join(EXTRA, ",") * "]   +xlaopt=" * repr(XLAOPT))
copts_excl(excl; xlaopt = Pair{Symbol,Any}[]) = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128, xlaopt...),
    (isempty(excl) ? (;) : (; excluded_passes = collect(String, excl)))...)

fresh_u() = RX.ConcreteRArray(copy(D.UBASE))
fresh_l() = RX.ConcreteRArray(copy(D.WOBJ))
tR() = RX.ConcreteRNumber(D.T0)
# P5_ARMB=bufgrad: arm A is the VJP with the forcing buffers ACTIVE (the
# pre-2026-09-04 form), arm B the driver's current form (`p` active only).
const ARMB = get(ENV, "P5_ARMB", "")
D_ssp_vjp_full(u, th, lam, t, dt) = D.RTI.ssprk43_step_vjp(D.gT, u, th, t, dt, lam, D.ATOL_T, D.RTOL; active_bufs = true)
D_ros_vjp_full(u, th, lam, t, dt) = D.RTI.ros23_step_vjp(D.gC, u, th, t, dt, lam, D.NS, D.NC, D.MASKS, D.ATOL_C, D.RTOL;
                                                          jac = D.JACMODE, gj = D.gJ, active_bufs = true)
D_ssp_vjp_p(u, th, lam, t, dt) = D.RTI.ssprk43_step_vjp(D.gT, u, th, t, dt, lam, D.ATOL_T, D.RTOL; active_bufs = false)
D_ros_vjp_p(u, th, lam, t, dt) = D.RTI.ros23_step_vjp(D.gC, u, th, t, dt, lam, D.NS, D.NC, D.MASKS, D.ATOL_C, D.RTOL;
                                                       jac = D.JACMODE, gj = D.gJ, active_bufs = false)
function compile_prog_variant(prog, copts, variant)
    if variant == :full
        prog == "ros_vjp" && return RX.@compile compile_options = copts D_ros_vjp_full(D.U_R, D.THC, D.LAM_R, D.T_R, D.DTC_R)
        prog == "ssp_vjp" && return RX.@compile compile_options = copts D_ssp_vjp_full(D.U_R, D.THT, D.LAM_R, D.T_R, D.DTT_R)
    else
        prog == "ros_vjp" && return RX.@compile compile_options = copts D_ros_vjp_p(D.U_R, D.THC, D.LAM_R, D.T_R, D.DTC_R)
        prog == "ssp_vjp" && return RX.@compile compile_options = copts D_ssp_vjp_p(D.U_R, D.THT, D.LAM_R, D.T_R, D.DTT_R)
    end
    return compile_prog(prog, copts)
end
function compile_prog(prog, copts)
    prog == "ros_step" && return RX.@compile compile_options = copts D.ros_step(D.U_R, D.THC, D.T_R, D.DTC_R)
    prog == "ssp_step" && return RX.@compile compile_options = copts D.ssp_step(D.U_R, D.THT, D.T_R, D.DTT_R)
    prog == "ros_vjp"  && return RX.@compile compile_options = copts D.ros_vjp(D.U_R, D.THC, D.LAM_R, D.T_R, D.DTC_R)
    prog == "ssp_vjp"  && return RX.@compile compile_options = copts D.ssp_vjp(D.U_R, D.THT, D.LAM_R, D.T_R, D.DTT_R)
    error("unknown program $prog")
end
function call_prog(prog, c)
    if prog == "ros_step"; return c(fresh_u(), D.THC, tR(), RX.ConcreteRNumber(D.DT0C))
    elseif prog == "ssp_step"; return c(fresh_u(), D.THT, tR(), RX.ConcreteRNumber(D.DT0T))
    elseif prog == "ros_vjp"; return c(fresh_u(), D.THC, fresh_l(), tR(), RX.ConcreteRNumber(D.DT0C))
    else; return c(fresh_u(), D.THT, fresh_l(), tR(), RX.ConcreteRNumber(D.DT0T))
    end
end
# Outputs as host numbers: (state-like vector, p-gradient vector or empty).
function outputs(prog, r)
    if endswith(prog, "_vjp")
        lam = Array(r[1])
        gp = r[2].p
        pv = [Float64(getfield(gp, k)) for k in D.PNAMES]
        return lam, pv
    else
        return Array(r[1]), Float64[]
    end
end
# Process CPU time (all threads). CLOCK_PROCESS_CPUTIME_ID == 2 on Linux. The
# cpu/wall ratio over the timed region says how many cores XLA:CPU actually
# keeps busy for this program (~3 for the primal step, xla_cpu_sweep.jl).
function cpu_ns()
    ts = Ref((Clong(0), Clong(0)))
    ccall(:clock_gettime, Cint, (Cint, Ptr{Cvoid}), 2, ts)
    t = ts[]
    return Int64(t[1]) * 1_000_000_000 + Int64(t[2])
end
const CPUWALL = Dict{String,Float64}()
function time_calls(prog, c, reps)
    for _ in 1:3; call_prog(prog, c); end
    ts = Float64[]
    c0 = cpu_ns(); w0 = time_ns()
    for _ in 1:reps
        t0 = time(); call_prog(prog, c); push!(ts, time() - t0)
    end
    CPUWALL[prog] = (cpu_ns() - c0) / max(time_ns() - w0, 1)
    return median(ts), minimum(ts)
end

for prog in PROGS
    say("\n---- $prog ----")
    t0 = time()
    CA = ARMB == "bufgrad" ? compile_prog_variant(prog, copts_excl(BASEX), :full) : compile_prog(prog, copts_excl(BASEX))
    say(@sprintf("  @compile default %.1f s", time() - t0))
    CB = if ARMB == "bufgrad"
        t0 = time(); c = compile_prog_variant(prog, copts_excl(BASEX), :p); say(@sprintf("  @compile p-only  %.1f s", time() - t0)); c
    elseif isempty(EXTRA) && isempty(XLAOPT)
        CA      # no extra set: time the driver default alone (both arms identical)
    else
        t0 = time(); c = compile_prog(prog, copts_excl(FULLX; xlaopt = XLAOPT)); say(@sprintf("  @compile +extra  %.1f s", time() - t0)); c
    end
    a1 = time_calls(prog, CA, REPS); b1 = time_calls(prog, CB, REPS)
    a2 = time_calls(prog, CA, REPS); b2 = time_calls(prog, CB, REPS)
    ma = min(a1[1], a2[1]); mb = min(b1[1], b2[1])
    say(@sprintf("P5_TIME %-8s NC=%d  default %.3f ms  +extra %.3f ms   ratio %.3f   (mins %.3f / %.3f)   cpu/wall %.2f",
                 prog, D.NC, 1e3 * ma, 1e3 * mb, ma / mb, 1e3 * min(a1[2], a2[2]), 1e3 * min(b1[2], b2[2]), CPUWALL[prog]))
    la, pa = outputs(prog, call_prog(prog, CA)); lb, pb = outputs(prog, call_prog(prog, CB))
    same = la == lb && pa == pb
    rel(x, y) = isempty(x) ? 0.0 : maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1e-300))
    say(@sprintf("P5_GATE %-8s bit-for-bit %s   max rel lambda %.3e   max rel p-grad %.3e   nonfinite %d",
                 prog, same, rel(la, lb), rel(pa, pb), count(!isfinite, lb) + count(!isfinite, pb)))
end
say("P5_TIME_DONE")
