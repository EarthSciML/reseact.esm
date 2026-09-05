#!/usr/bin/env julia
# ===========================================================================
# p6_ascalled_time.jl -- the transport / chemistry step and VJP timed EXACTLY
#                        as tools/adjoint_gradient.jl calls them, vs standalone.
# ===========================================================================
# The sharding agent's CONUS driver runs imply ~2.5-3 s per transport VJP
# INSIDE backward_stage! against 305 ms measured standalone, and the transport
# step ~10x slower in host_adaptive! than standalone. This probe replicates the
# two call sites line for line -- fresh ConcreteRArray uploads of u and lambda,
# ConcreteRNumber t/dt, the driver's own theta object, Array(r[1]) readback,
# the EEst scalar (step) or the TWO 160-component Float64(getfield(gp, k))
# extractions plus the non-finite scans (VJP) -- and times every piece, for
# ESS_OOP_SHIFT_SLICE=0 and =1, next to the standalone device-resident call.
#
#   RESEACT_NLON/NLAT/NLEV   grid (CONUS via sbatch only)
#   P6_PROGRAMS              default "ssp_step,ssp_vjp,ros_step,ros_vjp"
#   P6_REPS                  reps per arm (default 20)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p6ascalled")
include(joinpath(@__DIR__, "_env.jl"))
using Printf, Statistics
say(s) = (println(s); flush(stdout))
const PROGS = String.(split(get(ENV, "P6_PROGRAMS", "ssp_step,ssp_vjp,ros_step,ros_vjp"), ','))
const REPS = parse(Int, get(ENV, "P6_REPS", "20"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
const EXT = Base.get_extension(D.EA, :EarthSciASTReactantExt)
say("P6_VERSION " * (isdefined(EXT, :_RX_SHIFT_SLICE_VERSION) ? EXT._RX_SHIFT_SLICE_VERSION : "no marker") *
    "   excl=[" * join(D.EXCLP, ",") * "]   NC=$(D.NC) N=$(D.N)")

function compile_prog(prog)
    prog == "ros_step" && return RX.@compile compile_options = D.COPTS D.ros_step(D.U_R, D.THC, D.T_R, D.DTC_R)
    prog == "ssp_step" && return RX.@compile compile_options = D.COPTS D.ssp_step(D.U_R, D.THT, D.T_R, D.DTT_R)
    prog == "ros_vjp"  && return RX.@compile compile_options = D.COPTS D.ros_vjp(D.U_R, D.THC, D.LAM_R, D.T_R, D.DTC_R)
    prog == "ssp_vjp"  && return RX.@compile compile_options = D.COPTS D.ssp_vjp(D.U_R, D.THT, D.LAM_R, D.T_R, D.DTT_R)
    error("unknown program $prog")
end
th(prog) = startswith(prog, "ssp") ? D.THT : D.THC
dt0(prog) = startswith(prog, "ssp") ? D.DT0T : D.DT0C
med(v) = 1e3 * median(v)

# standalone: device-resident arguments, no readback (what p5/p6 timed)
function standalone(prog, c)
    UD = RX.ConcreteRArray(copy(D.UBASE)); LD = RX.ConcreteRArray(copy(D.WOBJ))
    TD = RX.ConcreteRNumber(D.T0); DD = RX.ConcreteRNumber(dt0(prog))
    ts = Float64[]
    for _ in 1:REPS
        t0 = time()
        endswith(prog, "_vjp") ? c(UD, th(prog), LD, TD, DD) : c(UD, th(prog), TD, DD)
        push!(ts, time() - t0)
    end
    return med(ts)
end

# as host_adaptive! calls a step
function ascalled_step(prog, c)
    u = copy(D.UBASE); t = D.T0; dtc = dt0(prog)
    tot = Float64[]; tup = Float64[]; tex = Float64[]; trd = Float64[]; tpost = Float64[]
    for _ in 1:REPS
        a = time_ns()
        UD = RX.ConcreteRArray(u); TD = RX.ConcreteRNumber(t); DD = RX.ConcreteRNumber(dtc)
        b = time_ns()
        r = c(UD, th(prog), TD, DD)
        cc = time_ns()
        raw = Array(r[1]); ee = Float64(r[2])
        d = time_ns()
        EEst = isnan(ee) ? 1.0e10 : ee
        unew = max.(raw, 0.0)
        rec = (copy(u), t, dtc, BitVector(raw .> 0.0))     # the tape record
        e = time_ns()
        push!(tot, (e - a) / 1e9); push!(tup, (b - a) / 1e9); push!(tex, (cc - b) / 1e9)
        push!(trd, (d - cc) / 1e9); push!(tpost, (e - d) / 1e9)
    end
    return (total = med(tot), upload = med(tup), exec = med(tex), readback = med(trd), post = med(tpost))
end

# as backward_stage! calls a VJP
function ascalled_vjp(prog, c)
    u = copy(D.UBASE); lam = copy(D.WOBJ); mask = trues(length(u)); t = D.T0; dtc = dt0(prog)
    gacc = Dict{Symbol,Float64}(k => 0.0 for k in D.PNAMES)
    tot = Float64[]; tup = Float64[]; tex = Float64[]; trd = Float64[]; tgb = Float64[]; tacc = Float64[]
    for _ in 1:REPS
        a = time_ns()
        lam2 = lam .* mask
        nin = count(!isfinite, lam2)
        UD = RX.ConcreteRArray(u); LD = RX.ConcreteRArray(lam2)
        TD = RX.ConcreteRNumber(t); DD = RX.ConcreteRNumber(dtc)
        b = time_ns()
        r = c(UD, th(prog), LD, TD, DD)
        cc = time_ns()
        lout = Array(r[1])
        d = time_ns()
        gp = r[2].p
        gbad = count(k -> !isfinite(Float64(getfield(gp, k))), D.PNAMES)
        nout = count(!isfinite, lout)
        e = time_ns()
        for k in D.PNAMES; gacc[k] += Float64(getfield(gp, k)); end
        f = time_ns()
        push!(tot, (f - a) / 1e9); push!(tup, (b - a) / 1e9); push!(tex, (cc - b) / 1e9)
        push!(trd, (d - cc) / 1e9); push!(tgb, (e - d) / 1e9); push!(tacc, (f - e) / 1e9)
        (gbad > 0 || nout > nin) && say("  (non-finite seen: gbad=$gbad nout=$nout)")
    end
    return (total = med(tot), upload = med(tup), exec = med(tex), readback = med(trd),
            p_scan = med(tgb), p_accum = med(tacc))
end

for prog in PROGS
    say("\n---- $prog ----")
    for flag in ("0", "1")
        ENV["ESS_OOP_SHIFT_SLICE"] = flag
        t0 = time(); c = compile_prog(prog); say(@sprintf("  @compile flag=%s %.1f s", flag, time() - t0))
        endswith(prog, "_vjp") ? c(RX.ConcreteRArray(copy(D.UBASE)), th(prog), RX.ConcreteRArray(copy(D.WOBJ)), RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(dt0(prog))) :
                                 c(RX.ConcreteRArray(copy(D.UBASE)), th(prog), RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(dt0(prog)))
        sa = standalone(prog, c)
        ac = endswith(prog, "_vjp") ? ascalled_vjp(prog, c) : ascalled_step(prog, c)
        say(@sprintf("P6_ASCALLED %-8s flag=%s  standalone %.1f ms   as-called %.1f ms   breakdown %s",
                     prog, flag, sa, ac.total,
                     join(["$(k)=$(round(v; digits = 2))" for (k, v) in pairs(ac) if k != :total], " ")))
    end
end
say("P6_ASCALLED_DONE")
