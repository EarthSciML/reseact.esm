#!/usr/bin/env julia
# Fast smoke test for rx_traced_integrator.jl on toy ODEs (no EarthSciAST):
# exercises @trace while, ros23_step (FD block Jacobian + blocksolve), and
# ssprk43_step end-to-end under @compile in seconds. Checks against exact
# solutions of u' = -lambda*u (decay, species-major NS=3 x NC=4).
import Pkg
const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io=devnull)
using Reactant, Printf
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end
include(joinpath(HERE, "rx_native_patch.jl"))
include(joinpath(HERE, "rx_traced_integrator.jl"))
say(s) = (println(s); flush(stdout))

const NS = 3; const NC = 4; const N = NS * NC
# species-major decay rates: species s decays at rate lam[s] (same for all cells)
lam = repeat([0.3, 1.0, 2.5]; inner=NC)
u0 = collect(range(1.0, 2.0; length=N))
masks = [Float64[div(i - 1, NC) + 1 == s ? 1.0 : 0.0 for i in 1:N] for s in 1:NS]
const T0s = 0.0; const T1s = 2.0
uexact = u0 .* exp.(-lam .* (T1s - T0s))

# lam rides as a TRACED array through aux -- regression test for the rule that
# the loop body must not close over traced values (they must be loop-carried).
mkf(aux) = (u, t) -> begin
    du = aux.lam .* u
    0.0 .- du
end

# --- ROS23 (adaptive) ---
adv_ros = (uR, t0R, t1R, dtR, lamR) -> begin
    fstep = (u, t, dtc, aux) -> RxTracedIntegrator.ros23_step(
        mkf(aux), u, t, dtc, NS, NC, masks, 1e-8, 1e-6)
    RxTracedIntegrator.adaptive_solve(fstep, uR, t0R, t1R, dtR,
        RxTracedIntegrator.pictrl_ros23(), (lam=lamR,))
end
args = (RX.ConcreteRArray(u0), RX.ConcreteRNumber(T0s), RX.ConcreteRNumber(T1s),
        RX.ConcreteRNumber(0.05), RX.ConcreteRArray(lam))
t = time()
xr = RX.@compile sync=true adv_ros(args...)
res = xr(args...)
uT = Array(res[1])
err = maximum(abs.(uT .- uexact) ./ abs.(uexact))
@printf("ROS23  : compile+run %.1f s  nacc=%d nrej=%d t=%.3f relerr=%.2e\n",
    time() - t, Int(round(Float64(res[4]))), Int(round(Float64(res[5]))),
    Float64(res[2]), err)
err < 1e-3 || error("ROS23 smoke FAILED (relerr=$err)")

# --- SSPRK43 (adaptive, with nonneg clamp active to exercise that path) ---
adv_ssp = (uR, t0R, t1R, dtR, lamR) -> begin
    fstep = (u, t, dtc, aux) -> RxTracedIntegrator.ssprk43_step(
        mkf(aux), u, t, dtc, 1e-8, 1e-6)
    RxTracedIntegrator.adaptive_solve(fstep, uR, t0R, t1R, dtR,
        RxTracedIntegrator.pictrl_ssprk43(), (lam=lamR,); clamp_nonneg=true)
end
t = time()
xs = RX.@compile sync=true adv_ssp(args...)
res = xs(args...)
uT = Array(res[1])
err = maximum(abs.(uT .- uexact) ./ abs.(uexact))
@printf("SSPRK43: compile+run %.1f s  nacc=%d nrej=%d t=%.3f relerr=%.2e\n",
    time() - t, Int(round(Float64(res[4]))), Int(round(Float64(res[5]))),
    Float64(res[2]), err)
err < 1e-3 || error("SSPRK43 smoke FAILED (relerr=$err)")
say("SMOKE OK")
