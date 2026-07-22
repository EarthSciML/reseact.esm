#!/usr/bin/env julia
# EXPERIMENT: does raising Reactant's 10,000 unique-name cap let the ReSEACT
# TRANSPORT half @compile?  (see RESULTS.md — the cap is hit by 25,945 identical
# `<op>_broadcast_scalar` helpers Reactant mints one-per-broadcast; known open
# Reactant issue #1616, no released fix / no flag.)
#
# Non-destructive: monkeypatches `Reactant.TracedUtils.__lookup_unique_name_in_module`
# at runtime (raises 0:10000 -> 0:10_000_000 and prints naming progress). Edits no
# package file; nothing to revert.
#
# Env: RESEACT_MODEL (default 7x7x8 for a cheaper XLA backend compile — the helper
#      count is grid-independent, so it's the same naming test), RESEACT_RXENV.
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

# ---- The patch: raise the cap + progress ------------------------------------
@eval Reactant.TracedUtils begin
    const _CAP_CALLS = Ref(0)
    function __lookup_unique_name_in_module(mod, name)
        _CAP_CALLS[] += 1
        (_CAP_CALLS[] % 2000 == 0) && (println("  [naming] $(_CAP_CALLS[]) helpers named…"); flush(stdout))
        new_name = name
        tab = MLIR.IR.SymbolTable(MLIR.IR.Operation(mod))
        for i in 0:10_000_000
            new_name = i == 0 ? name : name * "_" * string(i)
            MLIR.IR.mlirIsNull(MLIR.API.mlirSymbolTableLookup(tab, new_name)) && return new_name
        end
        return error("raised cap (10M) exceeded for $name")
    end
end
say(s) = (println(s); flush(stdout))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
const T0 = 64800.0
say("patched __lookup_unique_name_in_module cap -> 10,000,000")
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
say("\n=== @compile transport (cap raised) — naming progress follows ===")
xla = nothing
try
    t = time()
    global xla = RX.@compile sync=true g(ur, pr, tr, dev)
    say("RESULT: transport @compile SUCCEEDED in $(round(time()-t,digits=1)) s  (helpers named=$(Reactant.TracedUtils._CAP_CALLS[]))")
catch e
    say("RESULT: transport @compile FAILED after raising cap ->")
    showerror(stdout, e); println()
    say("EXPERIMENT DONE (failed)"); exit()
end
got = Array(xla(ur, pr, tr, dev))
say("RESULT: transport run finite=$(all(isfinite,got)) maxabs(dev-host)=$(maximum(abs, got .- du_host)) approx=$(isapprox(got, du_host; rtol=1e-8, atol=1e-10))")
say("EXPERIMENT DONE")
