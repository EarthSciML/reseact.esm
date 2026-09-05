#!/usr/bin/env julia
# ===========================================================================
# p6_shift_slice_time.jl -- ESS_OOP_SHIFT_SLICE (stencil-shift gathers emitted
#                           as slices of the reshaped state): A/B timing, a
#                           bit-for-bit gate, and an op census, per program.
# ===========================================================================
# Sibling of p5_vjp_time.jl. ONE build; each program is compiled twice from
# it -- arm A with ESS_OOP_SHIFT_SLICE=0, arm B with =1 (the flag is read at
# TRACE time, once per RHS call, in the Reactant extension's `_oop_new_memo`,
# so no rebuild is needed) -- and timed interleaved A/B/A/B on the same inputs
# with fresh buffers. The gate compares every output bit for bit (primal:
# unew, EEst; VJP: lambda AND the p-gradient) and reports the max relative
# difference otherwise. The census counts gather / scatter / slice / pad /
# transpose / dynamic_update_slice in the OPTIMIZED StableHLO of each arm, and
# the extension's tally says how many gathers the box detector converted.
# Requires the EarthSciAST branch perf/stencil-shift-as-slice (worktree
# code/EarthSciAST-stencilslice) through RESEACT_RXENV=run-model-jl-stencil.
#
#   RESEACT_NLON/NLAT/NLEV   grid (6x6x8 in a session; CONUS via p5.sbatch with
#                            P5_SCRIPT=p6_shift_slice_time.jl and RESEACT_RXENV)
#   P5_PROGRAMS              default "ssp_step,ssp_vjp,ros_step,ros_vjp"
#   P5_REPS                  timing reps per arm per pass (default 40)
#
# RESULTS: see the P6_* lines of the run log; recorded in the commit message.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p6slice")
include(joinpath(@__DIR__, "_env.jl"))
using Printf, Statistics
say(s) = (println(s); flush(stdout))
const EXTRA = String.(filter(!isempty, strip.(split(get(ENV, "P5_EXTRA", ""), ','))))
const PROGS = String.(split(get(ENV, "P5_PROGRAMS", "ssp_step,ssp_vjp,ros_step,ros_vjp"), ','))
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


const EXT = Base.get_extension(D.EA, :EarthSciASTReactantExt)
EXT === nothing && error("EarthSciASTReactantExt not loaded")
isdefined(EXT, :_rx_shift_slice_stats) || error("this EarthSciAST has no ESS_OOP_SHIFT_SLICE: point RESEACT_RXENV at run-model-jl-stencil")
say("P6_VERSION " * (isdefined(EXT, :_rx_shift_slice_version) ? "?" : (isdefined(EXT, :_RX_SHIFT_SLICE_VERSION) ? EXT._RX_SHIFT_SLICE_VERSION : "v1/v2 (no marker)")))
function census(txt)
    c(op) = length(collect(eachmatch(Regex("stablehlo\\." * op * "\\b"), txt)))
    return (gather = c("gather"), scatter = c("scatter"), slice = c("slice"), dslice = c("dynamic_slice"),
            pad = c("pad"), transpose = c("transpose"), dus = c("dynamic_update_slice"))
end
function hlo_prog(prog)
    prog == "ros_step" && return repr(RX.@code_hlo optimize = true D.ros_step(D.U_R, D.THC, D.T_R, D.DTC_R))
    prog == "ssp_step" && return repr(RX.@code_hlo optimize = true D.ssp_step(D.U_R, D.THT, D.T_R, D.DTT_R))
    prog == "ros_vjp"  && return repr(RX.@code_hlo optimize = true D.ros_vjp(D.U_R, D.THC, D.LAM_R, D.T_R, D.DTC_R))
    prog == "ssp_vjp"  && return repr(RX.@code_hlo optimize = true D.ssp_vjp(D.U_R, D.THT, D.LAM_R, D.T_R, D.DTT_R))
    error("unknown program $prog")
end
# The bare transport RHS, both arms, bit for bit -- the most direct check of the
# emitter change, independent of the integrator.
rhsT(u, th, t) = D.gT(u, th, t)
let
    ENV["ESS_OOP_SHIFT_SLICE"] = "0"; EXT._rx_shift_slice_reset!()
    ca = RX.@compile compile_options = copts_excl(BASEX) rhsT(D.U_R, D.THT, D.T_R)
    ENV["ESS_OOP_SHIFT_SLICE"] = "1"; EXT._rx_shift_slice_reset!()
    cb = RX.@compile compile_options = copts_excl(BASEX) rhsT(D.U_R, D.THT, D.T_R)
    st = EXT._rx_shift_slice_stats()
    ra = Array(ca(fresh_u(), D.THT, tR())); rb = Array(cb(fresh_u(), D.THT, tR()))
    rr = abs.(ra .- rb) ./ max.(abs.(ra), abs.(rb), 1e-300); iw = argmax(rr)
    say(@sprintf("P6_RHS transport RHS bit-for-bit %s   max rel %.3e   detector sliced %d / gathered %d",
                 ra == rb, maximum(rr), st[1], st[2]))
    say(@sprintf("P6_RHS worst entry %d  off %.15e  on %.15e   max |diff|/max|rhs| %.3e   entries with rel>1e-12: %d of %d",
                 iw, ra[iw], rb[iw], maximum(abs.(ra .- rb)) / max(maximum(abs.(ra)), 1e-300), count(>(1e-12), rr), length(rr)))
    isdefined(EXT, :_rx_shift_slice_reasons) && say("P6_WHY rhsT " * repr(EXT._rx_shift_slice_reasons()))
end
for prog in PROGS
    say("\n---- $prog ----")
    ENV["ESS_OOP_SHIFT_SLICE"] = "0"; EXT._rx_shift_slice_reset!()
    t0 = time(); CA = compile_prog(prog, copts_excl(BASEX)); say(@sprintf("  @compile flag off %.1f s", time() - t0))
    ha = census(hlo_prog(prog))
    ENV["ESS_OOP_SHIFT_SLICE"] = "1"; EXT._rx_shift_slice_reset!()
    t0 = time(); CB = compile_prog(prog, copts_excl(BASEX)); say(@sprintf("  @compile flag on  %.1f s", time() - t0))
    st = EXT._rx_shift_slice_stats()
    hb = census(hlo_prog(prog))
    say("P6_CENSUS $prog off " * repr(ha))
    say("P6_CENSUS $prog on  " * repr(hb) * "   detector sliced $(st[1]) / gathered $(st[2]) (compile+code_hlo)")
    isdefined(EXT, :_rx_shift_slice_reasons) && say("P6_WHY $prog " * repr(EXT._rx_shift_slice_reasons()))
    a1 = time_calls(prog, CA, REPS); b1 = time_calls(prog, CB, REPS)
    a2 = time_calls(prog, CA, REPS); b2 = time_calls(prog, CB, REPS)
    ma = min(a1[1], a2[1]); mb = min(b1[1], b2[1])
    say(@sprintf("P6_TIME %-8s NC=%d  off %.3f ms  on %.3f ms   speedup %.3f   (mins %.3f / %.3f)   cpu/wall %.2f",
                 prog, D.NC, 1e3 * ma, 1e3 * mb, ma / mb, 1e3 * min(a1[2], a2[2]), 1e3 * min(b1[2], b2[2]), CPUWALL[prog]))
    la, pa = outputs(prog, call_prog(prog, CA)); lb, pb = outputs(prog, call_prog(prog, CB))
    same = la == lb && pa == pb
    rel(x, y) = isempty(x) ? 0.0 : maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1e-300))
    say(@sprintf("P6_GATE %-8s bit-for-bit %s   max rel out %.3e   max rel p-grad %.3e   nonfinite %d",
                 prog, same, rel(la, lb), rel(pa, pb), count(!isfinite, lb) + count(!isfinite, pb)))
    let r = abs.(la .- lb) ./ max.(abs.(la), abs.(lb), 1e-300), i = argmax(r), big = maximum(abs.(la))
        say(@sprintf("P6_GATE %-8s worst out entry %d  off %.15e  on %.15e   |diff|/max|out| %.3e   entries with rel>1e-12: %d of %d",
                     prog, i, la[i], lb[i], abs(la[i] - lb[i]) / max(big, 1e-300), count(>(1e-12), r), length(r)))
        say(@sprintf("P6_GATE %-8s max |diff|/max|out| over all entries %.3e", prog, maximum(abs.(la .- lb)) / max(big, 1e-300)))
    end
    if !isempty(pa)
        # Name the worst p-gradient component with both values, and the difference
        # scaled by the LARGEST component: a near-zero component's relative error
        # is roundoff in disguise, and only the scaled figure can say so.
        r = abs.(pa .- pb) ./ max.(abs.(pa), abs.(pb), 1e-300)
        i = argmax(r); big = maximum(abs.(pa))
        say(@sprintf("P6_GATE %-8s worst p-grad %s  off %.15e  on %.15e   |diff|/max|grad| %.3e",
                     prog, D.PNAMES[i], pa[i], pb[i], abs(pa[i] - pb[i]) / max(big, 1e-300)))
    end
end
say("P6_TIME_DONE")
