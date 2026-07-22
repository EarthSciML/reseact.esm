#!/usr/bin/env julia
# EXPERIMENT: can we get UNDER Reactant's 10,000 unique-name cap by lowering the
# :oop runner's primitive broadcasts to native stablehlo, instead of letting
# Reactant mint one `<op>_broadcast_scalar` helper per broadcast site?
#
# Key observation (Reactant src/TracedRArray.jl:394-404): `_copyto!` calls
# `Reactant.broadcast_to_size` on EVERY broadcast arg BEFORE `elem_apply`, so by
# the time `elem_apply(f, args...)` runs, all array args already share one shape.
# That makes a native fast path trivial: `elem_apply(::typeof(*), a, b)` can be
# exactly `Ops.multiply(a, b)` (one stablehlo.multiply, ZERO helper funcs).
#
# Non-destructive: adds more-specific `elem_apply` methods for the primitive ops
# at runtime via `@eval` (falls back to the stock generic method for anything
# not covered). Also keeps the raised name cap as a safety net + instrumentation.
#
# Env: RESEACT_MODEL (default 7x7x8 — helper count is grid-independent), RESEACT_RXENV.
import Pkg
const HERE  = @__DIR__
const REPO  = normpath(joinpath(HERE, "..", ".."))
const RXENV = get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl"))
const MODEL = get(ENV, "RESEACT_MODEL", "/tmp/reseact_7x7x8.esm")
Pkg.activate(RXENV; io=devnull)
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system, stencil_vs_pointwise
using Reactant
using Logging
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

# ---- Patch: native stablehlo lowering + instrumentation (shared) ------------
include(joinpath(HERE, "rx_native_patch.jl"))
using Profile   # makes SIGUSR1 a live-profile peek instead of process death

say(s) = (println(s); flush(stdout))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
const T0 = 64800.0
say("patched: native stablehlo fast paths for primitive broadcasts + raised cap (safety net)")
say("model: $MODEL")

# ---- Split + build the TRANSPORT half as oop --------------------------------
fo1 = u0 = p = nothing; ov = mc = mp = nothing
Logging.with_logger(Logging.NullLogger()) do
    global fo1, u0, p, ov, mc, mp
    file = EA.load(MODEL); flat = EA.flatten(file)
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
    parts = split_system(flat, stencil_vs_pointwise; nparts = 2)   # [transport, pointwise]
    doc   = EA.flattened_to_esm(parts[1])                          # transport only
    ff = reseact_forcing(CHEMDIR)
    mc = Dict{String,Any}(String(k)=>v for (k,v) in ff.const_arrays)
    mp = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        (EA.provider_is_const(prov) ? mc : mp)[k] = fld
    end
    ov = Dict{String,Float64}(String(k)=>Float64(v) for (k,v) in ff.parameters)
    t = time()
    fo1, u0, p, _, _ = EA.build_evaluator(doc; form=:oop,
        parameter_overrides=ov, const_arrays=mc, param_arrays=mp)
    say("  transport build_evaluator(:oop) OK in $(round(time()-t,digits=1)) s; nstates=$(length(u0))")
end
du_host = fo1(u0, p, T0)
say("  host transport RHS: finite=$(all(isfinite,du_host))")

# ---- Compile the transport half (the experiment) ----------------------------
_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
ur = RX.ConcreteRArray(u0); pr = _dev(p); tr = RX.ConcreteRNumber(T0)
host = EA.forcing_buffers(fo1); dev = map(RX.ConcreteRArray, host)
g = EA.rhs_with_buffers(fo1)
say("\n=== @compile transport (native broadcast lowering) ===")

xla = nothing
try
    t = time()
    global xla = RX.@compile sync=true g(ur, pr, tr, dev)
    say("RESULT: transport @compile SUCCEEDED in $(round(time()-t,digits=1)) s")
    report_patch_stats()
catch e
    say("RESULT: transport @compile FAILED ->")
    showerror(stdout, e); println()
    report_patch_stats()
    say("EXPERIMENT DONE (failed)"); exit()
end
got = Array(xla(ur, pr, tr, dev))
say("RESULT: transport run finite=$(all(isfinite,got)) maxabs(dev-host)=$(maximum(abs, got .- du_host)) approx=$(isapprox(got, du_host; rtol=1e-8, atol=1e-10))")
say("EXPERIMENT DONE")
