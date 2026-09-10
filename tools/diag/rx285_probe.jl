#!/usr/bin/env julia
# ===========================================================================
# rx285_probe.jl -- the version-sensitive WORKAROUNDS, re-tested per Reactant
#                   version.  Cheap, toy-sized, no ReSEACT build.
# ===========================================================================
# Everything here was established on 0.2.274/0.2.280 and is version-specific
# evidence, so it is re-run rather than inherited:
#
#   L   the four ways to read a PER-ITERATION dt inside a differentiated
#       `@trace for` (Reactant v0.2.281 claims to fix "wrong gradient with
#       dynamic indexing in a traced loop", #3197):
#         L-a  dts[i]                     -- "Scalar indexing is disallowed"
#         L-b  dts[i:i]                   -- iota MethodError
#         L-c  Ops.dynamic_slice(dts,[i],[1])  -- the one that worked, 1.19e-16
#         L-d  one-hot mask select        -- COMPILED AND SILENTLY WRONG, 2.7e-1
#       A varying dt sequence, so reading the wrong element cannot accidentally
#       agree with the reference.
#
#   F   flag assignability under this jll: xla_cpu_use_fusion_emitters was NOT
#       assignable on 0.2.280 (a wanted lever we could not pull), and
#       raise_first is the subject of the v0.2.284 "silent wrong result" fix.
#       Assignability is tested by actually compiling and running, and the
#       ANSWER is gated -- a flag that compiles and changes the number is worse
#       than one that errors.
#
# Companion: tools/diag/frozen_grid_probe.jl carries K4 (the checkpointing
# modes) and K3a (`dts[i]` inside the loop); run it alongside this.
# ===========================================================================
using Reactant, ForwardDiff, Printf
const RX = Reactant
const EZ = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end
say(s) = (println(s); flush(stdout))
function err1(e)
    s = sprint(showerror, e)
    m = match(r"error: ([^\n]+)", s)
    return m === nothing ? first(split(s, '\n')) : "MLIR: " * m.captures[1]
end

say("="^78)
say("rx285_probe   Reactant v$(pkgversion(Reactant))   jll v$(pkgversion(RX.Reactant_jll))   julia $(VERSION)")
say("="^78)

@inline vf(u, k) = k .* u .* (1 .+ 0.1 .* u)
const NS_L  = 6
const U0_L  = [0.5 + 0.1i for i in 1:NS_L]
const K0    = 0.7
const CAP_L = 16
const NLIVE = 10
const DTS_L = Float64[i <= NLIVE ? 0.005 * i : 1.0 for i in 1:CAP_L]
host_L(k) = (u = copy(U0_L); for i in 1:NLIVE; u = u .+ DTS_L[i] .* vf(u, k); end; sum(u))
const JREF_L = host_L(K0)
const GREF_L = ForwardDiff.derivative(host_L, K0)
say(@sprintf("\nL reference (varying dt, %d live of CAP %d): J=%.16g  dJ/dk = %.16g",
             NLIVE, CAP_L, JREF_L, GREF_L))

L_scalar(u, k, dts, nlive) = begin
    RX.@trace for i in 1:CAP_L
        h = dts[i]
        u = ifelse.(i <= nlive, u .+ h .* vf(u, k), u)
    end
    sum(u)
end
L_slice(u, k, dts, nlive) = begin
    RX.@trace for i in 1:CAP_L
        h = sum(dts[i:i])
        u = ifelse.(i <= nlive, u .+ h .* vf(u, k), u)
    end
    sum(u)
end
L_dynslice(u, k, dts, nlive) = begin
    RX.@trace for i in 1:CAP_L
        h = sum(RX.Ops.dynamic_slice(dts, [i], [1]))
        u = ifelse.(i <= nlive, u .+ h .* vf(u, k), u)
    end
    sum(u)
end
L_onehot(u, k, dts, nlive) = begin
    idx = RX.Ops.constant(collect(1:CAP_L))
    RX.@trace for i in 1:CAP_L
        h = sum(ifelse.(idx .== i, dts, zero(dts)))
        u = ifelse.(i <= nlive, u .+ h .* vf(u, k), u)
    end
    sum(u)
end

gradk(f) = (u, k, rest...) -> EZ.gradient(EZ.Reverse, f, EZ.Const(u), k,
                                          map(EZ.Const, rest)...)[2]

function check(tag, thunk, gref; rtol = 1e-10)
    try
        t0 = time(); got = Float64(thunk()); el = time() - t0
        rel = abs(got - gref) / max(abs(gref), 1e-300)
        say(@sprintf("  [%s] %-34s got %.16g  rel %.3e  (%.1f s)%s",
                     rel <= rtol ? "PASS" : "FAIL", tag, got, rel, el,
                     rel <= rtol ? "" : "   <-- WRONG, not an error"))
        return rel <= rtol
    catch e
        say(@sprintf("  [ERR ] %-34s %s", tag, err1(e)))
        return false
    end
end

# `@compile` DONATES every input buffer the program does not return, so a
# `ConcreteRArray` reused across two calls is silently corrupted -- which is
# how a correct formulation reads WRONG on its second call.  Every call below
# therefore gets FRESH device buffers, and the sequence is run twice with the
# variant order SWAPPED so that "which variant went first" cannot be mistaken
# for "which variant is wrong".
fresh() = (RX.ConcreteRArray(copy(U0_L)), RX.ConcreteRNumber(K0),
           RX.ConcreteRArray(copy(DTS_L)), RX.ConcreteRNumber(NLIVE))
const LVAR = [("L-a dts[i] scalar read", L_scalar),
              ("L-b dts[i:i] size-1 slice", L_slice),
              ("L-c Ops.dynamic_slice", L_dynslice),
              ("L-d one-hot mask select", L_onehot)]
for (pass, order) in (("fresh buffers, order a-b-c-d", LVAR),
                      ("fresh buffers, order d-c-b-a", reverse(LVAR)))
    say("\n  --- $pass ---")
    for (tag, f) in order
        # primal first: a formulation whose PRIMAL is already wrong is a
        # different failure from one whose primal is right and whose gradient
        # is wrong.
        check(tag * " PRIMAL", () -> begin
            a = fresh(); c = RX.@compile sync=true f(a...); c(fresh()...)
        end, JREF_L)
        check(tag * " GRAD", () -> begin
            a = fresh(); g = gradk(f)
            c = RX.@compile sync=true g(a...); c(fresh()...)
        end, GREF_L)
    end
end
# And the ANTI-test: the same L-c call sequence with buffers REUSED, which is
# what tools/diag/frozen_grid_probe2.jl did.  If this is the only way L-c or
# L-d reads wrong, the historical "silently wrong by 2.7e-1" was donation, not
# a Reactant bug.
say("\n  --- REUSED buffers (the probe2 shape): is the failure donation? ---")
let a = fresh()
    for (tag, f) in LVAR
        check(tag * " GRAD reused", () -> begin
            g = gradk(f); c = RX.@compile sync=true g(a...); c(a...)
        end, GREF_L)
    end
end

# ---------------------------------------------------------------- K ---------
# CHECKPOINTING, re-tested with FRESH buffers per call.  Every mode was
# recorded silently wrong on 0.2.274 and 0.2.280 (rel 2.10 for plain `true`,
# 26.4 for Periodic(5)) -- but the harness that measured that
# (tools/diag/frozen_grid_probe.jl K4) reused one `ConcreteRArray` across the
# whole arm sequence, and the FIRST arm ("none") is the one that passed.  That
# is the donation signature, so the verdict has to be re-earned.
say("\nK checkpointing on a STATIC-trip `@trace for`, fresh buffers per call")
const NSTEP_K = 50
const H_K     = 0.02
host_K(k) = (u = copy(U0_L); for _ in 1:NSTEP_K; u = u .+ H_K .* vf(u, k); end; sum(u))
const GREF_K = ForwardDiff.derivative(host_K, K0)
say(@sprintf("  reference: NSTEP=%d  J=%.12g  dJ/dk=%.12g", NSTEP_K, host_K(K0), GREF_K))
static_loop(u, k, ckpt) = begin
    if ckpt === nothing
        RX.@trace track_numbers=false for _ in 1:NSTEP_K
            u = u .+ H_K .* vf(u, k)
        end
    else
        RX.@trace track_numbers=false checkpointing=ckpt for _ in 1:NSTEP_K
            u = u .+ H_K .* vf(u, k)
        end
    end
    sum(u)
end
const CKPTS = [("none", nothing), ("true", true), ("Periodic(5)", RX.Periodic(5)),
               ("Binomial(2)", RX.Binomial(2)), ("Binomial(5)", RX.Binomial(5)),
               ("Binomial(10)", RX.Binomial(10))]
for (lbl, order) in (("order as listed", CKPTS), ("order reversed", reverse(CKPTS)))
    say("  --- $lbl ---")
    for (nm, ck) in order
        f = (u, k) -> static_loop(u, k, ck)
        check("K static for, ckpt=$nm", () -> begin
            g = gradk(f)
            c = RX.@compile sync=true g(RX.ConcreteRArray(copy(U0_L)), RX.ConcreteRNumber(K0))
            c(RX.ConcreteRArray(copy(U0_L)), RX.ConcreteRNumber(K0))
        end, GREF_K; rtol = 1e-8)
    end
end
# the same sequence with buffers REUSED -- the frozen_grid_probe K4 shape
say("  --- REUSED buffers (the K4 shape) ---")
let ur = RX.ConcreteRArray(copy(U0_L)), kr = RX.ConcreteRNumber(K0)
    for (nm, ck) in CKPTS
        f = (u, k) -> static_loop(u, k, ck)
        check("K reused, ckpt=$nm", () -> begin
            g = gradk(f); c = RX.@compile sync=true g(ur, kr); c(ur, kr)
        end, GREF_K; rtol = 1e-8)
    end
end
# and the DATA-DEPENDENT while, which is what Reactant's own error hint points
# `checkpointing=Binomial(budget)` at.
say("  --- data-dependent `@trace while` + Binomial, fresh buffers ---")
while_body(u, k, ckpt) = begin
    t = zero(k)
    if ckpt === nothing
        RX.@trace track_numbers=false while t < 1.0 - 1e-9
            u = u .+ H_K .* vf(u, k); t = t + H_K
        end
    else
        RX.@trace track_numbers=false checkpointing=ckpt while t < 1.0 - 1e-9
            u = u .+ H_K .* vf(u, k); t = t + H_K
        end
    end
    sum(u)
end
for (nm, ck) in (("none", nothing), ("Binomial(4)", RX.Binomial(4)),
                 ("Binomial(8)", RX.Binomial(8)), ("Binomial(16)", RX.Binomial(16)))
    f = (u, k) -> while_body(u, k, ck)
    check("K while, ckpt=$nm", () -> begin
        g = gradk(f)
        c = RX.@compile sync=true g(RX.ConcreteRArray(copy(U0_L)), RX.ConcreteRNumber(K0))
        c(RX.ConcreteRArray(copy(U0_L)), RX.ConcreteRNumber(K0))
    end, GREF_K; rtol = 1e-8)
end

# ---------------------------------------------------------------- F ---------
say("\nF flag assignability under this jll (compile + run + gate the answer)")
const FLAGS = [
    ("baseline (driver's COPTS shape)", (; sync = true,
        xla_debug_options = (; xla_cpu_prefer_vector_width = 128))),
    ("no xla_cpu_prefer_vector_width", (; sync = true)),
    ("xla_cpu_use_fusion_emitters=true", (; sync = true,
        xla_debug_options = (; xla_cpu_prefer_vector_width = 128,
                             xla_cpu_use_fusion_emitters = true))),
    ("raise_first=true", (; sync = true, raise_first = true,
        xla_debug_options = (; xla_cpu_prefer_vector_width = 128))),
]
for (nm, kw) in FLAGS
    try
        co = RX.CompileOptions(; kw...)
        f = L_dynslice
        ur = RX.ConcreteRArray(U0_L); kr = RX.ConcreteRNumber(K0)
        dr = RX.ConcreteRArray(DTS_L); nr = RX.ConcreteRNumber(NLIVE)
        g = gradk(f)
        c = RX.@compile compile_options = co g(ur, kr, dr, nr)
        got = Float64(c(ur, kr, dr, nr))
        rel = abs(got - GREF_L) / abs(GREF_L)
        say(@sprintf("  [%s] %-34s grad %.16g  rel %.3e", rel <= 1e-10 ? "OK  " : "WRONG", nm, got, rel))
    catch e
        say(@sprintf("  [ERR ] %-34s %s", nm, err1(e)))
    end
end

# The v0.2.284 fix is specifically `sum(dest .* weights)` fused under
# raise_first. Price it directly, since that IS the shape of every objective
# reduction in the driver (`sum(w .* u)`).
say("\nF2 the v0.2.284 shape: sum(dest .* weights), raise_first on/off")
let n = 257
    d = [1.0 + 1e-3 * i for i in 1:n]; w = [0.5 - 1e-4 * i for i in 1:n]
    ref = sum(d .* w)
    gref = copy(w)                     # d(sum(d.*w))/dd = w
    f(d, w) = sum(d .* w)
    gd(d, w) = EZ.gradient(EZ.Reverse, f, d, EZ.Const(w))[1]
    for rf in (false, true)
        try
            co = RX.CompileOptions(; sync = true, raise_first = rf,
                                   xla_debug_options = (; xla_cpu_prefer_vector_width = 128))
            dr = RX.ConcreteRArray(d); wr = RX.ConcreteRArray(w)
            cp = RX.@compile compile_options = co f(dr, wr)
            pv = Float64(cp(RX.ConcreteRArray(d), RX.ConcreteRArray(w)))
            cg = RX.@compile compile_options = co gd(dr, wr)
            gv = Array(cg(RX.ConcreteRArray(d), RX.ConcreteRArray(w)))
            say(@sprintf("  raise_first=%-5s primal rel %.3e   grad max abs dev %.3e",
                         rf, abs(pv - ref) / abs(ref), maximum(abs.(gv .- gref))))
        catch e
            say(@sprintf("  raise_first=%-5s ERR %s", rf, err1(e)))
        end
    end
end

say("\nRX285_PROBE_DONE")
