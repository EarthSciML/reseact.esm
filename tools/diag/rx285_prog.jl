#!/usr/bin/env julia
# ===========================================================================
# rx285_prog.jl -- PER-PROGRAM timings and OUTPUT DUMP for ONE Reactant
#                  version.  The cross-version comparison is done by running
#                  this twice (two environments cannot share a process) and
#                  diffing the dumps; tools/diag/rx285_prog.sbatch drives it.
# ===========================================================================
# Same programs, same compile options, same base point and the same fresh
# device buffer per call as tools/diag/p13_ssa_ab.jl, so the numbers are
# comparable to the campaign's `P13_TIME` rows:
#
#   ros_step  38.895 / 38.619 ms      ssp_step  13.410 / 13.204 ms
#   ros_vjp   81.111 / 81.504 ms      ssp_vjp  306.556 / 307.441 ms
#   rhs        3.275 ms (min 3.181)   rhs_vjp   78.318 ms (min 75.612)
#
# `@compile` DONATES any input buffer the program does not return, so every
# call uploads fresh -- reusing one `ConcreteRArray` across calls silently
# corrupts the result, and is what made three earlier probe verdicts in this
# campaign wrong (see tools/diag/rx285_probe.jl).
#
#   RX285_PROGRAMS  default "ssp_step,ssp_vjp,ros_step,ros_vjp"
#   RX285_REPS      timing reps per pass (default 40); two passes, best of two
#   RX285_OUT       where to serialize (label, version, timings, outputs)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = get(ENV, "RESEACT_ADJ_UJITTER", "1e-1")
ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "rx285prog")
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Statistics, Serialization
say(s) = (println(s); flush(stdout))
const PROGS = String.(split(get(ENV, "RX285_PROGRAMS", "ssp_step,ssp_vjp,ros_step,ros_vjp"), ','))
const REPS  = parse(Int, get(ENV, "RX285_REPS", "40"))
const OUTF  = get(ENV, "RX285_OUT", joinpath(REPO, "logs", "rx285prog-" * get(ENV, "RESEACT_LABEL", "x") * ".jls"))

const D = Main
t0 = time()
include(joinpath(REPO, "tools", "adjoint_gradient.jl"))
say(@sprintf("RX285_BUILD %.1f s", time() - t0))
const RXV = string(pkgversion(RX))
say("RX285_PROG Reactant=$RXV grid=$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$NLEV_EFF NC=$NC excl=$(repr(EXCLP)) xlafix=$XLAFIX progs=$(join(PROGS, ","))")

const EZ2 = RTI.EZ
fresh_u() = RX.ConcreteRArray(copy(UBASE))
fresh_l() = RX.ConcreteRArray(copy(WOBJ))
tR() = RX.ConcreteRNumber(T0)

rhs_f(u, th, t) = gT(u, th, t)
ldot(u, p, rest, lam, t) = sum(lam .* gT(u, merge((p = p,), rest), t))
rhs_vjp_f(u, th, lam, t) = begin
    r = EZ2.gradient(EZ2.Reverse, ldot, u, th.p,
                     EZ2.Const(RTI._theta_rest(th)), EZ2.Const(lam), EZ2.Const(t))
    (r[1], r[2])
end

function compile_prog(prog)
    prog == "rhs"      && return RX.@compile compile_options=COPTS rhs_f(U_R, THT, T_R)
    prog == "rhs_vjp"  && return RX.@compile compile_options=COPTS rhs_vjp_f(U_R, THT, LAM_R, T_R)
    prog == "ssp_step" && return RX.@compile compile_options=COPTS ssp_step(U_R, THT, T_R, DTT_R)
    prog == "ssp_vjp"  && return RX.@compile compile_options=COPTS ssp_vjp(U_R, THT, LAM_R, T_R, DTT_R)
    prog == "ros_step" && return RX.@compile compile_options=COPTS ros_step(U_R, THC, T_R, DTC_R)
    prog == "ros_vjp"  && return RX.@compile compile_options=COPTS ros_vjp(U_R, THC, LAM_R, T_R, DTC_R)
    error("unknown program $prog")
end
function call_prog(prog, c)
    prog == "rhs"      && return c(fresh_u(), THT, tR())
    prog == "rhs_vjp"  && return c(fresh_u(), THT, fresh_l(), tR())
    prog == "ssp_step" && return c(fresh_u(), THT, tR(), RX.ConcreteRNumber(DT0T))
    prog == "ssp_vjp"  && return c(fresh_u(), THT, fresh_l(), tR(), RX.ConcreteRNumber(DT0T))
    prog == "ros_step" && return c(fresh_u(), THC, tR(), RX.ConcreteRNumber(DT0C))
    prog == "ros_vjp"  && return c(fresh_u(), THC, fresh_l(), tR(), RX.ConcreteRNumber(DT0C))
    error("unknown program $prog")
end
# `rhs` returns bare du; the steps return (u_next, err); the VJPs return
# (lambda, dtheta) with the p-gradient under dtheta.p, except the bare rhs_vjp
# built here which differentiates w.r.t. th.p so its second result IS p.
function outputs(prog, r)
    endswith(prog, "_vjp") || return (Array(r isa Tuple ? r[1] : r), Float64[])
    g = r[2]; gp = hasproperty(g, :p) ? g.p : g
    return (Array(r[1]), [Float64(getfield(gp, k)) for k in PNAMES])
end
function time_calls(prog, c)
    for _ in 1:3; call_prog(prog, c); end
    ts = Float64[]
    for _ in 1:REPS
        t0 = time(); call_prog(prog, c); push!(ts, time() - t0)
    end
    return median(ts), minimum(ts)
end

RESULTS = Dict{String,Any}("version" => RXV, "grid" => (GRID_MP["NLON"], GRID_MP["NLAT"], NLEV_EFF), "NC" => NC,
                       "excl" => copy(EXCLP), "xlafix" => XLAFIX,
                       "label" => get(ENV, "RESEACT_LABEL", ""), "progs" => PROGS)
for prog in PROGS
    tc = time(); c = compile_prog(prog); ct = time() - tc
    say(@sprintf("RX285_COMPILE %-8s %.1f s", prog, ct))
    r1 = time_calls(prog, c); r2 = time_calls(prog, c)
    med = min(r1[1], r2[1]); mn = min(r1[2], r2[2])
    o = outputs(prog, call_prog(prog, c))
    nnf = count(!isfinite, o[1]) + count(!isfinite, o[2])
    say(@sprintf("RX285_TIME %-8s med %.3f ms  min %.3f ms  compile %.1f s  nonfinite %d",
                 prog, 1e3 * med, 1e3 * mn, ct, nnf))
    RESULTS[prog] = Dict("med" => med, "min" => mn, "compile" => ct,
                     "state" => o[1], "pgrad" => o[2])
end
mkpath(dirname(OUTF)); open(OUTF, "w") do io; serialize(io, RESULTS); end
say("RX285_PROG_DONE wrote $OUTF")
