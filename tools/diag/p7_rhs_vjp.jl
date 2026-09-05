#!/usr/bin/env julia
# ===========================================================================
# p7_rhs_vjp.jl -- the transport RHS alone versus its reverse: is the 22x
#                  transport-VJP anomaly in Enzyme's reverse of the emitted PPM
#                  RHS, or in the SSPRK43 composition around it?
# ===========================================================================
# Compiles gT (one RHS evaluation of the transport half) and the reverse of
# <lambda, gT(u)> w.r.t. (u, p) with the forcing buffers Const -- the same
# shape ssprk43_step_vjp uses, minus the four-stage step. Times both (median,
# min, cpu/wall) and dumps the optimized HLO for tools/diag/p5_fusion_census.py.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 6 6 8; CONUS through sbatch)
#   P7_REPS                  timing reps (default 40)
#
# RESULTS (2026-09-05, driver default passes, quiet nodes):
#   CONUS 13x7x72 (slurm 10375981, ccc0235): rhs 5.397 ms (cpu/wall 5.9),
#     rhs_vjp 105.567 ms (cpu/wall 6.6) => 19.6x. The 4-stage step is 14 / 305 ms
#     (22x), so the whole anomaly is inside Enzyme's reverse of the emitted PPM
#     RHS. Refined census (p5_fusion_census.py logic with effective reads):
#     reverse arithmetic 23.5x the primal's, bytes read 45x; 501 divide-root
#     fusions on [1,76440] interior slabs do 64% of it; 179 full-buffer
#     zeroing DUS + 98 copies of the 2 MB extended buffer per RHS reverse.
#   6x6x8 (slurm 10376416): rhs 0.532 ms, rhs_vjp 2.960 ms (5.6x; floors).
#   Mechanism isolated in tools/diag/p8_ppm_reverse.jl.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p7rhs")
include(joinpath(@__DIR__, "_env.jl"))
using Printf, Statistics
say(s) = (println(s); flush(stdout))
const REPS = parse(Int, get(ENV, "P7_REPS", "40"))
const GRID = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT = joinpath(REPO, "logs", "p7-rhs-$GRID-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
const RX = D.RX
const EZ = D.RTI.EZ
say("P7_RHS_VJP grid=$GRID NC=$(D.NC) out=$OUT")

rhs(u, th, t) = D.gT(u, th, t)
ldot(u, p, rest, lam, t) = sum(lam .* D.gT(u, merge((p = p,), rest), t))
function rhs_vjp(u, th, lam, t)
    r = EZ.gradient(EZ.Reverse, ldot, u, th.p, EZ.Const(D.RTI._theta_rest(th)), EZ.Const(lam), EZ.Const(t))
    return r[1], r[2]
end

function copts_dump(dir)
    RX.CompileOptions(; sync = true,
        xla_debug_options = (; xla_cpu_prefer_vector_width = 128, xla_dump_to = dir, xla_dump_hlo_as_text = true),
        (isempty(D.EXCLP) ? (;) : (; excluded_passes = collect(String, D.EXCLP)))...)
end
function cpu_ns()
    ts = Ref((Clong(0), Clong(0)))
    ccall(:clock_gettime, Cint, (Cint, Ptr{Cvoid}), 2, ts)
    t = ts[]; return Int64(t[1]) * 1_000_000_000 + Int64(t[2])
end
fresh_u() = RX.ConcreteRArray(copy(D.UBASE)); fresh_l() = RX.ConcreteRArray(copy(D.WOBJ)); tR() = RX.ConcreteRNumber(D.T0)
function timeit(name, c, call)
    for _ in 1:3; call(c); end
    ts = Float64[]; c0 = cpu_ns(); w0 = time_ns()
    for _ in 1:REPS; t0 = time(); call(c); push!(ts, time() - t0); end
    cw = (cpu_ns() - c0) / max(time_ns() - w0, 1)
    say(@sprintf("P7_TIME %-8s NC=%d  median %.3f ms  min %.3f ms  cpu/wall %.2f", name, D.NC, 1e3 * median(ts), 1e3 * minimum(ts), cw))
end
let dir = joinpath(OUT, "rhs"); mkpath(dir); t0 = time()
    c = RX.@compile compile_options = copts_dump(dir) rhs(D.U_R, D.THT, D.T_R)
    say(@sprintf("  @compile rhs %.1f s", time() - t0))
    timeit("rhs", c, c -> c(fresh_u(), D.THT, tR()))
end
let dir = joinpath(OUT, "rhs_vjp"); mkpath(dir); t0 = time()
    c = RX.@compile compile_options = copts_dump(dir) rhs_vjp(D.U_R, D.THT, D.LAM_R, D.T_R)
    say(@sprintf("  @compile rhs_vjp %.1f s", time() - t0))
    timeit("rhs_vjp", c, c -> c(fresh_u(), D.THT, fresh_l(), tR()))
end
say("P7_DONE")
