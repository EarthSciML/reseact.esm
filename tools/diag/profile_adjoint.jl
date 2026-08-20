#!/usr/bin/env julia
# ===========================================================================
# profile_adjoint.jl -- WHERE DOES THE ADJOINT'S TIME ACTUALLY GO?
# ===========================================================================
# The 5.1x backward/forward ratio and the ~70 h week projection are both
# extrapolations from 3,744 states. Before believing either, decompose one
# chemistry step into:
#
#   * host->device upload            (ConcreteRArray(u))
#   * XLA execution                  (device-resident args, no readback)
#   * device->host readback          (Array(r[1]))
#   * residual Julia dispatch        (the leftover)
#   * the RHS alone, so we can say whether a step IS its 14 FD-Jacobian RHS
#     evaluations or is dominated by something else
#
# and separately count the inner chemistry steps of ONE 300 s macro step with
# their dt distribution -- because 240 steps for 300 s of chemistry means the
# accuracy controller, not the arithmetic, is what sets the bill.
#
# Reuses adjoint_gradient.jl's build verbatim by including it with STAGES unset,
# which builds + compiles ssp_step/ros_step and runs no stage.
#
#   RESEACT_NLON/NLAT/NLEV  grid (default 6/6/8)
#   RESEACT_PROF_REPS       timed repetitions per measurement (default 20)
#   RESEACT_PROF_VJP        also compile+profile the VJP (default 1)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_ADJ_UJITTER", "1e-1")
ENV["RESEACT_ADJ_STAGES"] = "none"           # build + compile, run nothing
ENV["RESEACT_LABEL"] = "profile"

Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics

const REPS = parse(Int, get(ENV, "RESEACT_PROF_REPS", "20"))
const DOVJP = get(ENV, "RESEACT_PROF_VJP", "1") == "1"
RX = D.RX; RTI = D.RTI
say(s) = (println(s); flush(stdout))

# median of REPS timings, in ms
function tmed(f, reps = REPS)
    f(); ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e6)
    end
    return median(ts), minimum(ts)
end

say("\n" * "="^75)
say(@sprintf("PROFILE  grid=%s  NS=%d NC=%d N=%d  reps=%d",
             "$(ENV["RESEACT_NLON"])x$(ENV["RESEACT_NLAT"])x$(ENV["RESEACT_NLEV"])",
             D.NS, D.NC, D.N, REPS))
say("="^75)

u = copy(D.UBASE)
UD  = RX.ConcreteRArray(u)
TD  = RX.ConcreteRNumber(D.T0)
DD  = RX.ConcreteRNumber(D.DT0C)
LAMD = RX.ConcreteRArray(copy(D.WOBJ))
r0 = D.CROS(UD, D.THC, TD, DD)      # warm + a device array to read back

say("\n---- A. one CHEMISTRY step (ros_step), decomposed ----")
a_full, a_full_min = tmed(() -> begin
    r = D.CROS(RX.ConcreteRArray(u), D.THC, RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(D.DT0C))
    (Array(r[1]), Float64(r[2]))
end)
a_dev,  _ = tmed(() -> D.CROS(UD, D.THC, TD, DD))
a_read, _ = tmed(() -> Array(r0[1]))
a_up,   _ = tmed(() -> RX.ConcreteRArray(u))
a_scal, _ = tmed(() -> (RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(D.DT0C)))
@printf("  A1 full host round-trip (what host_adaptive! does) %9.3f ms  (min %.3f)\n", a_full, a_full_min)
@printf("  A2 XLA exec, device-resident args, no readback     %9.3f ms   %5.1f%% of A1\n", a_dev, 100a_dev/a_full)
@printf("  A3 readback  Array(r[1])                           %9.3f ms   %5.1f%%\n", a_read, 100a_read/a_full)
@printf("  A4 upload    ConcreteRArray(u)                     %9.3f ms   %5.1f%%\n", a_up, 100a_up/a_full)
@printf("  A5 two ConcreteRNumber scalars                     %9.3f ms   %5.1f%%\n", a_scal, 100a_scal/a_full)
@printf("  A6 residual (A1 - A2 - A3 - A4 - A5)               %9.3f ms   %5.1f%%\n",
        a_full - a_dev - a_read - a_up - a_scal, 100*(a_full-a_dev-a_read-a_up-a_scal)/a_full)

say("\n---- B. is a step just its RHS evaluations? ----")
# ros23_step(jac=:fd, unrolled=true) does NS+1 FD-Jacobian RHS evals + stage evals.
rhs_only(uu, th, tt) = D.gC(uu, th, tt)
CRHS = nothing
try
    t0 = time()
    global CRHS = RX.@compile sync=true rhs_only(UD, D.THC, TD)
    @printf("  compiled bare chemistry RHS in %.1f s\n", time() - t0)
catch e
    say("  RHS compile FAILED: " * first(split(sprint(showerror, e), '\n')))
end
if CRHS !== nothing
    b_rhs, _ = tmed(() -> CRHS(UD, D.THC, TD))
    @printf("  B1 one chemistry RHS eval (device-resident)        %9.3f ms\n", b_rhs)
    @printf("  B2 step / RHS  = %.1f x   (NS+1 = %d FD-Jacobian evals + ~3 stage evals expected)\n",
            a_dev / b_rhs, D.NS + 1)
end

if DOVJP
    say("\n---- C. the VJP ----")
    t0 = time()
    CV = RX.@compile sync=true D.ros_vjp(UD, D.THC, LAMD, TD, DD)
    @printf("  compiled ros_vjp in %.1f s\n", time() - t0)
    c_full, _ = tmed(() -> begin
        r = CV(RX.ConcreteRArray(u), D.THC, RX.ConcreteRArray(D.WOBJ),
               RX.ConcreteRNumber(D.T0), RX.ConcreteRNumber(D.DT0C))
        (Array(r[1]), r[2].p)
    end)
    c_dev, _ = tmed(() -> CV(UD, D.THC, LAMD, TD, DD))
    @printf("  C1 full host round-trip                           %9.3f ms\n", c_full)
    @printf("  C2 XLA exec, device-resident                      %9.3f ms   %5.1f%% of C1\n", c_dev, 100c_dev/c_full)
    @printf("  C3 VJP / step (XLA exec only)  = %.2f x\n", c_dev / a_dev)
    @printf("  C4 VJP / step (as the driver calls them) = %.2f x\n", c_full / a_full)
end

say("\n---- D. inner chemistry steps of ONE 300 s macro step ----")
# What the controller actually does: how many attempts, and what dt it settles on.
seq = D.StepSeq()
uh, tend, dtend, na, nr = D.host_adaptive!(D.CROS, copy(D.UBASE), D.T0, D.T0 + D.MACRO_DT,
                                           D.DT0C, RTI.pictrl_ros23(), D.THC;
                                           seq = seq, clamp_nonneg = false)
dts = [d for (_, d) in seq]
@printf("  accepted %d, rejected %d over %.0f s  =>  mean dt %.3f s\n", na, nr, D.MACRO_DT, D.MACRO_DT / max(na,1))
if !isempty(dts)
    @printf("  dt: min %.4g  p25 %.4g  median %.4g  p75 %.4g  max %.4g\n",
            minimum(dts), quantile(dts, .25), median(dts), quantile(dts, .75), maximum(dts))
end
@printf("  => one macro step of chemistry costs %d x %.3f ms = %.2f s (as driven)\n",
        na + nr, a_full, (na + nr) * a_full / 1000)
@printf("     of which XLA exec %.2f s (%.0f%%), host overhead %.2f s (%.0f%%)\n",
        (na+nr)*a_dev/1000, 100a_dev/a_full,
        (na+nr)*(a_full-a_dev)/1000, 100*(a_full-a_dev)/a_full)
say("\nPROFILE_DONE")
