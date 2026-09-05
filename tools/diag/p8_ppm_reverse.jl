#!/usr/bin/env julia
# ===========================================================================
# p8_ppm_reverse.jl -- how much does Enzyme's reverse of the PPM flux template
#                      cost, standalone, and does the ALGEBRAIC FORM matter?
# ===========================================================================
# tools/diag/p7_rhs_vjp.jl found that ONE transport RHS evaluation at CONUS
# costs 5.4 ms and its reverse 105.6 ms (19.6x), with the reverse doing 23.5x
# the elementwise arithmetic of the primal (refined fusion census). This probe
# takes the model out: a 1-D PPM flux divergence written exactly as
# EarthSciDiscretizations' templates spell it (ppm_slope_mono / ppm_face_value_
# mono / ppm_limit_right / ppm_flux, ifelse form) on a Reactant vector of N
# cells, its Enzyme reverse, and the same with the monotone slope limiter
# rewritten in sign/min form (no ifelse). Prints forward/reverse times and
# dumps the optimized HLO for tools/diag/p5_fusion_census.py.
#
#   P8_N   cells (default 76440 = CONUS interior-level slab)
#
# RESULTS (slurm 10376679 / 10376831, ccc0234, N=76440, 8 threads):
#   rhs_ifelse  0.139-0.147 ms   grad_ifelse 2.50-2.62 ms   => 17x
#   rhs_sign    0.167-0.170 ms   grad_sign   2.39-2.69 ms   => same (values ==)
#   Optimized HLO: ONE kLoop fusion each; body ops 116 (forward) vs 241
#   (reverse) -- 2.1x the instructions for 17x the time. XLA:CPU's loop fusion
#   evaluates each output element from scratch, and the reverse's six shifted
#   (pad/slice) contributions each re-derive the whole face chain at a
#   different face: ~6 x 2.1 ~ 13-17x. Not Enzyme's algebra, not the limiter's
#   form. The barrier variants: XLA:CPU drops stablehlo.optimization_barrier
#   (forward still one fusion) and Enzyme has no reverse rule for it (both
#   grad_bar* fail to compile). See p8_mof_probe.jl for what XLA:CPU will and
#   will not share across consumers.
# ===========================================================================
using Reactant, Printf, Statistics
const EZ = Reactant.Enzyme
const N = parse(Int, get(ENV, "P8_N", "76440"))
const OUT = joinpath(@__DIR__, "..", "..", "logs", "p8-ppm-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)
say(s) = (println(s); flush(stdout))

# --- the templates, verbatim (ifelse form) ---------------------------------
ppm_slope(am, ap) = 0.5 * (ap - am)
function slope_mono_ifelse(am, a0, ap)
    m = min(abs(a0 - am), abs(ap - a0))
    ifelse((ap - a0) * (a0 - am) > 0, max(-2m, min(2m, ppm_slope(am, ap))), zero(a0))
end
# sign/min form: identical values (the clamp of a quantity whose sign equals
# both differences' sign is sign * min(2m, |ppm_slope|))
function slope_mono_sign(am, a0, ap)
    d1 = a0 - am; d2 = ap - a0
    s = 0.5 * (sign(d1) + sign(d2))           # +-1 when the signs agree, else 0
    s * min(2 * min(abs(d1), abs(d2)), abs(ppm_slope(am, ap)))
end
face_value_mono(sl, a, b, c, d) = b + 0.5 * (c - b) + (1 / 6) * (sl(a, b, c) - sl(b, c, d))
function limit_right(ql, qr, qi)
    ifelse((qr - qi) * (qi - ql) <= 0, qi,
           ifelse(-(qr - ql)^2 > (qr - ql) * (6 * (qi - 0.5 * (ql + qr))), 3qi - 2ql, qr))
end
ppm_flux(sl, um2, um1, u0, up1, up2) =
    limit_right(face_value_mono(sl, um2, um1, u0, up1), face_value_mono(sl, um1, u0, up1, up2), u0)

# The same flux with stablehlo.optimization_barrier on the shared per-face
# intermediates. XLA:CPU's loop fusion computes each output element from
# scratch, so the reverse's six shifted contributions each re-derive the whole
# face chain at a different face; a barrier makes XLA materialise the chain
# ONCE per face and the reverse reads it at shifted indices instead.
bar(xs...) = Reactant.Ops.optimization_barrier(xs...)

# 1-D flux-form divergence, constant positive wind, 3-cell halo each side.
function rhs(flux, sl, u)
    n = length(u)
    F(i0) = flux(sl, (@view u[i0-2:i0-2+n-7]), (@view u[i0-1:i0-1+n-7]), (@view u[i0:i0+n-7]),
                 (@view u[i0+1:i0+1+n-7]), (@view u[i0+2:i0+2+n-7]))
    Fr = F(4); Fl = F(3)                      # right/left face of cells 4..n-3
    return Fl .- Fr
end
# broadcast the scalar templates over slabs; the barrier variants take slabs
flux_bc(sl, a...) = ppm_flux.(sl, a...)
rhs_ifelse(u) = rhs(flux_bc, slope_mono_ifelse, u)
rhs_sign(u)   = rhs(flux_bc, slope_mono_sign, u)
sl_bc(sl) = (a, b, c) -> sl.(a, b, c)
fvm_bc(sl, a, b, c, d) = face_value_mono.(sl, a, b, c, d)
function flux_bar_faces(sl, um2, um1, u0, up1, up2)
    ql = fvm_bc(sl, um2, um1, u0, up1); qr = fvm_bc(sl, um1, u0, up1, up2)
    ql, qr = bar(ql, qr)
    limit_right.(ql, qr, u0)
end
function flux_bar_all(sl, um2, um1, u0, up1, up2)
    s1 = sl.(um2, um1, u0); s2 = sl.(um1, u0, up1); s3 = sl.(u0, up1, up2)
    s1, s2, s3 = bar(s1, s2, s3)
    ql = um1 .+ 0.5 .* (u0 .- um1) .+ (1 / 6) .* (s1 .- s2)
    qr = u0 .+ 0.5 .* (up1 .- u0) .+ (1 / 6) .* (s2 .- s3)
    ql, qr = bar(ql, qr)
    limit_right.(ql, qr, u0)
end
rhs_barf(u) = rhs(flux_bar_faces, slope_mono_ifelse, u)
rhs_bara(u) = rhs(flux_bar_all, slope_mono_ifelse, u)
ldot(f, u, lam) = sum(lam .* f(u))
grad_ifelse(u, lam) = EZ.gradient(EZ.Reverse, ldot, EZ.Const(rhs_ifelse), u, EZ.Const(lam))[2]
grad_sign(u, lam)   = EZ.gradient(EZ.Reverse, ldot, EZ.Const(rhs_sign), u, EZ.Const(lam))[2]
grad_barf(u, lam)   = EZ.gradient(EZ.Reverse, ldot, EZ.Const(rhs_barf), u, EZ.Const(lam))[2]
grad_bara(u, lam)   = EZ.gradient(EZ.Reverse, ldot, EZ.Const(rhs_bara), u, EZ.Const(lam))[2]

copts(dir) = Reactant.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128, xla_dump_to = dir, xla_dump_hlo_as_text = true),
    excluded_passes = ["dynamic_update_to_concat", "sub_const_prop"])

u0 = 1.0 .+ 0.3 .* sin.(range(0, 40, length = N)) .+ 0.05 .* rand(N)
lam0 = rand(N - 6)
uR = Reactant.ConcreteRArray(u0); lR = Reactant.ConcreteRArray(lam0)
function timeit(name, c, args...; reps = 40)
    for _ in 1:3; c(args...); end
    ts = [(t0 = time(); c(args...); time() - t0) for _ in 1:reps]
    say(@sprintf("P8_TIME %-12s N=%d  median %.3f ms  min %.3f ms", name, N, 1e3 * median(ts), 1e3 * minimum(ts)))
end
# value gate: the two slope forms agree
let a = Array(Reactant.@jit(rhs_ifelse(uR))), b = Array(Reactant.@jit(rhs_sign(uR)))
    say(@sprintf("P8_GATE rhs ifelse vs sign: max abs diff %.3e (max |rhs| %.3e)", maximum(abs.(a .- b)), maximum(abs.(a))))
end
const REF = Dict{String,Vector{Float64}}()
for (name, f, args) in (("rhs_ifelse", rhs_ifelse, (uR,)), ("grad_ifelse", grad_ifelse, (uR, lR)),
                        ("rhs_sign", rhs_sign, (uR,)), ("grad_sign", grad_sign, (uR, lR)),
                        ("rhs_barf", rhs_barf, (uR,)), ("grad_barf", grad_barf, (uR, lR)),
                        ("rhs_bara", rhs_bara, (uR,)), ("grad_bara", grad_bara, (uR, lR)))
    dir = joinpath(OUT, name); mkpath(dir); t0 = time()
    c = try
        Reactant.compile(f, args; compile_options = copts(dir))
    catch e
        say("  $name compile FAILED: " * first(split(sprint(showerror, e), '\n'))[1:min(end, 300)]); continue
    end
    say(@sprintf("  @compile %s %.1f s", name, time() - t0))
    timeit(name, c, args...)
    v = Array(c(args...)); k = startswith(name, "grad") ? "grad" : "rhs"
    if haskey(REF, k)
        say(@sprintf("P8_GATE %-12s vs %s_ifelse: max abs diff %.3e (max |ref| %.3e)", name, k, maximum(abs.(v .- REF[k])), maximum(abs.(REF[k]))))
    else
        REF[k] = v
    end
end
say("P8_DONE")
