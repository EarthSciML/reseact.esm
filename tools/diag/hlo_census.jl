#!/usr/bin/env julia
# ===========================================================================
# hlo_census.jl -- HOW MANY OPS DOES ONE CHEMISTRY STEP ACTUALLY EXECUTE?
# ===========================================================================
# Every module-size number recorded in this repo (chem RHS 18,040 lines, ROS23
# primal jac=:sym 109,035) comes from `@code_hlo optimize=false`, i.e. BEFORE
# XLA's own CSE/fusion. That is the wrong number for a cost model: the measured
# per-step time divided by the UNOPTIMIZED op count gives ~1.3 cycles per
# element-op, which is only meaningful if the optimizer is not collapsing the
# graph underneath it.
#
# This prints, for the chem RHS, the symbolic block Jacobian and the whole ROS23
# step: unoptimized op count, OPTIMIZED op count, and the optimized op histogram
# (so `exp`/`power`/`divide` -- the expensive transcendentals -- are separated
# from cheap add/multiply, and fusion count is visible on the GPU backend).
#
# The question it answers: is the step expensive because the emitter produces
# ~25x more arithmetic than the mechanism needs (=> CSE in the emitter is the
# lever), or because each surviving op runs at ~1 element per 1.3 cycles
# (=> fusion / a parallel backend is the lever)?
#
#   RESEACT_NLON/NLAT/NLEV  grid (default 6/6/8)
#   RESEACT_BACKEND         cpu (default) | gpu
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "hlocensus"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf
RX = D.RX
say(s) = (println(s); flush(stdout))

# One op per line of the form `%name = dialect.op(...)`; count by `dialect.op`.
function census(txt::AbstractString)
    h = Dict{String,Int}()
    for m in eachmatch(r"=\s+\"?([a-zA-Z_][\w]*\.[\w.]+)\"?", txt)
        h[m.captures[1]] = get(h, m.captures[1], 0) + 1
    end
    return h
end
total(h) = sum(values(h))

# The expensive ones: a transcendental is 10-50x an add on any backend, so a
# histogram that is 90% add/multiply says something very different from one that
# is 20% exp/power.
const PRICEY = Set(["stablehlo.exponential", "stablehlo.power", "stablehlo.divide",
                    "stablehlo.log", "stablehlo.sqrt", "stablehlo.rsqrt",
                    "stablehlo.tanh", "stablehlo.exponential_minus_one"])

function report(nm, unopt_thunk, opt_thunk)
    say("\n" * "-"^70)
    say("  $nm")
    u = try sprint(show, unopt_thunk()) catch e; say("    unopt FAILED: " * first(split(sprint(showerror,e),'\n'))); "" end
    o = try sprint(show, opt_thunk())   catch e; say("    opt   FAILED: " * first(split(sprint(showerror,e),'\n'))); "" end
    hu, ho = census(u), census(o)
    @printf("    unoptimized ops %8d   (%d lines)\n", total(hu), count(==('\n'), u))
    @printf("    OPTIMIZED   ops %8d   (%d lines)   ratio %.2fx\n",
            total(ho), count(==('\n'), o), total(hu) / max(total(ho), 1))
    npricey = sum(get(ho, k, 0) for k in PRICEY)
    @printf("    of the optimized ops, %d are transcendental/divide (%.1f%%)\n",
            npricey, 100 * npricey / max(total(ho), 1))
    say("    top optimized ops:")
    for (k, v) in first(sort(collect(ho); by = last, rev = true), 12)
        @printf("      %-40s %7d\n", k, v)
    end
    return total(ho)
end

say("="^70)
say(@sprintf("HLO CENSUS  backend=%s grid=%sx%sx%s  NS=%d NC=%d N=%d",
             "cpu", ENV["RESEACT_NLON"], ENV["RESEACT_NLAT"], ENV["RESEACT_NLEV"],
             D.NS, D.NC, D.N))
say("="^70)

UR = RX.ConcreteRArray(copy(D.UBASE))
TR = RX.ConcreteRNumber(D.T0)
DR = RX.ConcreteRNumber(D.DT0C)

nrhs = report("chemistry RHS  gC(u, theta, t)",
    () -> RX.@code_hlo(optimize = false, D.gC(UR, D.THC, TR)),
    () -> RX.@code_hlo(D.gC(UR, D.THC, TR)))

njac = D.SYMJAC ? report("symbolic block Jacobian  gJ(u, theta, t)",
    () -> RX.@code_hlo(optimize = false, D.gJ(UR, D.THC, TR)),
    () -> RX.@code_hlo(D.gJ(UR, D.THC, TR))) : 0

nstep = report("ONE ROS23 chemistry step  ros_step(u, theta, t, dt)",
    () -> RX.@code_hlo(optimize = false, D.ros_step(UR, D.THC, TR, DR)),
    () -> RX.@code_hlo(D.ros_step(UR, D.THC, TR, DR)))

say("\n" * "="^70)
@printf("ONE ROS23 STEP = %d optimized ops over NC=%d cells = %.3g element-ops\n",
        nstep, D.NC, float(nstep) * D.NC)
say("Divide the measured s/step by that to get seconds per element-op; at 3 GHz,")
say("anything above ~0.3 cycles/element-op means the backend is not vectorising.")
say("CENSUS_DONE")
