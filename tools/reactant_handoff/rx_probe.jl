#!/usr/bin/env julia
# Does the full ReSEACT model actually run under Reactant.jl?  (see README.md)
#
# Builds the whole model as ONE out-of-place RHS -- the only form the Reactant
# extension (EarthSciAST/ext/EarthSciASTReactantExt.jl) can trace -- and tries
# the two documented routes:
#   Test A: 3-arg wrapper  fo(u,p,t)          -> EXPECTED to REFUSE (live forcing, audit J5)
#   Test B: rhs_with_buffers(fo)(u,p,t,dev)   -> the SUPPORTED route; the real test
#   Test C: a forcing refresh (sync_forcing!) is observed by the compiled program
#
# Env:
#   RESEACT_RXENV  Julia env with EarthSciAST + Reactant + a solver (see README).
#                  Default: <repo>/run-model-jl  (you must `Pkg.add("Reactant")` there
#                  first, or point this at a copy that has it).
#   RESEACT_MODEL  which .esm to compile. Default: <repo>/reseact.esm (7x7x72).
#                  Point at smaller/larger grids to test whether XLA compile time &
#                  MEMORY scale with cell count (the key open question -- see README).
import Pkg
const HERE  = @__DIR__
const REPO  = normpath(joinpath(HERE, "..", ".."))          # tools/reactant_handoff -> repo
const RXENV = get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl"))
const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
Pkg.activate(RXENV; io=devnull)
using EarthSciAST, EarthSciIO, JSON3
using Reactant
using Logging
const EA = EarthSciAST
const RX = Reactant
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))               # reseact_forcing(dir)
say(s) = (println(s); flush(stdout))
const T0 = 64800.0

# ---- Build the full-model out-of-place RHS (lift warnings silenced) ----------
# This is the piece NO tool in tools/ builds: the reseact tooling only produces
# the operator-SPLIT in-place f! pair (split_common.jl), and an in-place f!
# captures a host scratch buffer per _VecNode and therefore cannot be traced.
say("building oop RHS for: $MODEL")
fo = u0 = p = nothing
Logging.with_logger(Logging.NullLogger()) do
    global fo, u0, p
    file = EA.load(MODEL)
    flat = EA.flatten(file)
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
    doc  = EA.flattened_to_esm(flat)
    ff = reseact_forcing(CHEMDIR)
    mc = Dict{String,Any}(String(k)=>v for (k,v) in ff.const_arrays)
    mp = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        (EA.provider_is_const(prov) ? mc : mp)[k] = fld
    end
    ov = Dict{String,Float64}(String(k)=>Float64(v) for (k,v) in ff.parameters)
    t = time()
    fo, u0, p, _, _ = EA.build_evaluator(doc; form=:oop, parameter_overrides=ov,
        const_arrays=mc, param_arrays=mp)
    say("  build_evaluator(form=:oop) OK in $(round(time()-t,digits=1)) s; nstates=$(length(u0))")
end
du_host = fo(u0, p, T0)
say("  host f! reference: finite=$(all(isfinite,du_host)), n=$(length(du_host))")

_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
ur = RX.ConcreteRArray(u0); pr = _dev(p); tr = RX.ConcreteRNumber(T0)

# ---- Test A: the 3-arg wrapper must refuse a host-buffer trace ---------------
say("\n=== Test A: @compile fo(u,p,t)   [expect refusal: 6 live GEOS-FP buffers] ===")
try
    RX.@compile sync=true fo(ur, pr, tr)
    say("A: compiled WITHOUT refusal  (UNEXPECTED -- the J5 guard did not fire)")
catch e
    msg = sprint(showerror, e)
    say("A: refused. mentions param_arrays=$(occursin("param_arrays",msg)) const_arrays=$(occursin("const_arrays",msg))")
    say("A: head: " * first(split(msg, '\n')))
end

# ---- Test B: the supported route (buffers as traced arguments) ---------------
say("\n=== Test B: @compile rhs_with_buffers(fo)(u,p,t,dev)   [the real test] ===")
host = EA.forcing_buffers(fo)
dev  = map(RX.ConcreteRArray, host)
g    = EA.rhs_with_buffers(fo)
xla  = nothing
try
    t = time()
    global xla = RX.@compile sync=true g(ur, pr, tr, dev)
    say("B: @compile SUCCEEDED in $(round(time()-t,digits=1)) s")
catch e
    say("B: @compile FAILED ->"); showerror(stdout, e); println()
    say("RX_PROBE DONE (B failed)"); exit()
end
got = Array(xla(ur, pr, tr, dev))
say("B: run OK. finite=$(all(isfinite,got)) maxabs(dev-host)=$(maximum(abs, got .- du_host)) approx=$(isapprox(got,du_host;rtol=1e-8,atol=1e-10))")

# ---- Test C: forcing refresh observed by the compiled program ----------------
say("\n=== Test C: forcing refresh via sync_forcing! is observed ===")
try
    key = first(keys(host))
    before   = copy(Array(xla(ur, pr, tr, dev)))
    getfield(host, key) .+= 1.0            # perturb one live buffer host-side
    du_host2 = fo(u0, p, T0)               # host RHS sees it
    EA.sync_forcing!(dev, host)            # mirror host->device (NO recompile)
    after    = Array(xla(ur, pr, tr, dev))
    say("C: compiled saw refresh=$(!isapprox(after,before)) matches_host=$(isapprox(after,du_host2;rtol=1e-8,atol=1e-10))")
catch e
    say("C: FAILED ->"); showerror(stdout, e); println()
end
say("RX_PROBE DONE")
