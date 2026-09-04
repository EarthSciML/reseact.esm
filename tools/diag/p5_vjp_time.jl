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
#   P5_REPS                  timing reps per arm per pass (default 40)
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
say("P5_VJP_TIME grid=$(D.NC) cells   default excl=[" * join(BASEX, ",") * "]   +extra=[" * join(EXTRA, ",") * "]")
copts_excl(excl) = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
    (isempty(excl) ? (;) : (; excluded_passes = collect(String, excl)))...)

fresh_u() = RX.ConcreteRArray(copy(D.UBASE))
fresh_l() = RX.ConcreteRArray(copy(D.WOBJ))
tR() = RX.ConcreteRNumber(D.T0)
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
function time_calls(prog, c, reps)
    for _ in 1:3; call_prog(prog, c); end
    ts = Float64[]
    for _ in 1:reps
        t0 = time(); call_prog(prog, c); push!(ts, time() - t0)
    end
    return median(ts), minimum(ts)
end

for prog in PROGS
    say("\n---- $prog ----")
    t0 = time(); CA = compile_prog(prog, copts_excl(BASEX)); say(@sprintf("  @compile default %.1f s", time() - t0))
    CB = if isempty(EXTRA)
        CA      # no extra set: time the driver default alone (both arms identical)
    else
        t0 = time(); c = compile_prog(prog, copts_excl(FULLX)); say(@sprintf("  @compile +extra  %.1f s", time() - t0)); c
    end
    a1 = time_calls(prog, CA, REPS); b1 = time_calls(prog, CB, REPS)
    a2 = time_calls(prog, CA, REPS); b2 = time_calls(prog, CB, REPS)
    ma = min(a1[1], a2[1]); mb = min(b1[1], b2[1])
    say(@sprintf("P5_TIME %-8s NC=%d  default %.3f ms  +extra %.3f ms   ratio %.3f   (mins %.3f / %.3f)",
                 prog, D.NC, 1e3 * ma, 1e3 * mb, ma / mb, 1e3 * min(a1[2], a2[2]), 1e3 * min(b1[2], b2[2])))
    la, pa = outputs(prog, call_prog(prog, CA)); lb, pb = outputs(prog, call_prog(prog, CB))
    same = la == lb && pa == pb
    rel(x, y) = isempty(x) ? 0.0 : maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1e-300))
    say(@sprintf("P5_GATE %-8s bit-for-bit %s   max rel lambda %.3e   max rel p-grad %.3e   nonfinite %d",
                 prog, same, rel(la, lb), rel(pa, pb), count(!isfinite, lb) + count(!isfinite, pb)))
end
say("P5_TIME_DONE")
