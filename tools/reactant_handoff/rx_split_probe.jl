#!/usr/bin/env julia
# Split ReSEACT into [transport, pointwise] and trace BOTH halves under Reactant.
#
# This is the next step past rx_probe.jl (which traced the whole, UNSPLIT model as
# one :oop RHS). Here we:
#   1. split the flattened system with EarthSciASTSplitter (stencil_vs_pointwise),
#   2. build each part as an OUT-OF-PLACE RHS (form = :oop) over the SAME live
#      forcing buffers -- the splitter is representation-agnostic and forwards
#      form=:oop straight through build_evaluator, so each part yields a traceable
#      RHS (this is the ":oop split path" the handoff said did not exist yet),
#   3. verify on host that f_transport(u) + f_pointwise(u) == f_full(u),
#   4. @compile each half via rhs_with_buffers and check it matches its host half.
#
# Env (same as rx_probe.jl):
#   RESEACT_RXENV  Julia env with EarthSciAST + EarthSciASTSplitter + Reactant.
#   RESEACT_MODEL  which .esm to split+compile. Default: <repo>/reseact.esm.
import Pkg
const HERE  = @__DIR__
const REPO  = normpath(joinpath(HERE, "..", ".."))
const RXENV = get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl"))
const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
Pkg.activate(RXENV; io=devnull)
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system, stencil_vs_pointwise
using Reactant
using Logging
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end   # no GPU on this box; silence the fallback
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))               # reseact_forcing(dir)
say(s) = (println(s); flush(stdout))
const T0 = 64800.0
const PARTNAME = ("transport", "pointwise")
const DO_SIGMA = get(ENV, "RESEACT_SIGMA_CHECK", "0") == "1"   # Σ-check already proven 0.0; opt-in
# Concise error line: the failing @compile dumps the WHOLE MLIR module via showerror
# (100k+ lines). We only want the human-readable reason (last non-empty line).
function err_reason(e)
    msg = sprint(showerror, e)
    lines = filter(!isempty, split(msg, '\n'))
    reason = isempty(lines) ? "(no message)" : String(last(lines))
    # If the failure dumped the MLIR module (the naming-cap error does), quantify the
    # helper proliferation that caused it -- without echoing the 100k-line module.
    if occursin("_broadcast_scalar", msg)
        n = count(_ -> true, eachmatch(r"sym_name = \"[^\"]*_broadcast_scalar", msg))
        nmul = count(_ -> true, eachmatch(r"sym_name = \"\*_broadcast_scalar", msg))
        reason *= "  [helpers: $n total, $nmul are *]"
    end
    reason
end

# ---- 1+2. Split, then build each part's out-of-place RHS ---------------------
say("splitting + building oop RHS per part for: $MODEL")
fo = Vector{Any}(undef, 2); u0 = p = nothing; ov = mc = mp = nothing
Logging.with_logger(Logging.NullLogger()) do
    global fo, u0, p, ov, mc, mp
    file = EA.load(MODEL)
    flat = EA.flatten(file)
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
    # stencil_following_rule (split_common.jl), not the shipped
    # stencil_vs_pointwise: the air-mass equation reads `divh_fix`, an OBSERVED
    # that is a stencil, and the syntactic rule would post that term to the
    # chemistry half.
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)   # [transport, pointwise]
    docs  = [EA.flattened_to_esm(pt) for pt in parts]

    # forcing wiring identical to rx_probe.jl: const providers -> const_arrays,
    # discrete providers sampled at T0 -> param_arrays (shared across both parts).
    ff = reseact_forcing(CHEMDIR)
    mc = Dict{String,Any}(String(k)=>v for (k,v) in ff.const_arrays)
    mp = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        (EA.provider_is_const(prov) ? mc : mp)[k] = fld
    end
    ov = Dict{String,Float64}(String(k)=>Float64(v) for (k,v) in ff.parameters)

    vm_ref = nothing
    for i in 1:2
        t = time()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form=:oop,
            parameter_overrides=ov, const_arrays=mc, param_arrays=mp)
        say("  part[$i]=$(PARTNAME[i]) build_evaluator(form=:oop) OK in $(round(time()-t,digits=1)) s; nstates=$(length(u0i))")
        fo[i] = fi
        if i == 1
            u0, p, vm_ref = u0i, pi, vmi
        else
            vmi == vm_ref || error("part 2 var_map != part 1 -- state sets must match")
        end
    end
end
fo1, fo2 = fo[1], fo[2]

# ---- 3. Host reconstruction: f_transport + f_pointwise == f_full -------------
du_ref = (fo1(u0, p, T0), fo2(u0, p, T0))
say("  host part reconstruction: finite1=$(all(isfinite,du_ref[1])) finite2=$(all(isfinite,du_ref[2]))")
if DO_SIGMA
    say("  building UNSPLIT full oop RHS for the Σ-check (~75 s) ...")
    fofull = nothing
    Logging.with_logger(Logging.NullLogger()) do
        global fofull
        file = EA.load(MODEL); flat = EA.flatten(file)
        flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
        doc  = EA.flattened_to_esm(flat)
        fofull, = EA.build_evaluator(doc; form=:oop, parameter_overrides=ov, const_arrays=mc, param_arrays=mp)
    end
    du_full = fofull(u0, p, T0); s = du_ref[1] .+ du_ref[2]
    say("  Σ check: maxabs((du1+du2) - du_full)=$(maximum(abs, s .- du_full)) approx=$(isapprox(s, du_full; rtol=1e-8, atol=1e-10))")
end

# ---- 4. Compile each half via rhs_with_buffers (the supported route) ---------
# Pointwise (chemistry, cell-local) FIRST -- it is the smaller half and the stiff
# one that most wants XLA; transport SECOND (its unshared inlined stencil is the
# heavyweight). Each is independent -- a failure in one does NOT skip the other.
_dev(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
_dev(::Nothing) = nothing
ur = RX.ConcreteRArray(u0); pr = _dev(p); tr = RX.ConcreteRNumber(T0)

for i in (2, 1)                                   # pointwise first, then transport
    f = fo[i]
    say("\n=== Compile part[$i]=$(PARTNAME[i]): @compile rhs_with_buffers(fo)(u,p,t,dev) ===")
    host = EA.forcing_buffers(f)
    dev  = map(RX.ConcreteRArray, host)
    g    = EA.rhs_with_buffers(f)
    local xla
    try
        t = time()
        xla = RX.@compile sync=true g(ur, pr, tr, dev)
        say("  part[$i]=$(PARTNAME[i]): @compile SUCCEEDED in $(round(time()-t,digits=1)) s")
    catch e
        say("  part[$i]=$(PARTNAME[i]): @compile FAILED -> $(err_reason(e))")
        continue
    end
    got = Array(xla(ur, pr, tr, dev))
    say("  part[$i]=$(PARTNAME[i]): run OK. finite=$(all(isfinite,got)) maxabs(dev-host)=$(maximum(abs, got .- du_ref[i])) approx=$(isapprox(got, du_ref[i]; rtol=1e-8, atol=1e-10))")
end
say("\nRX_SPLIT_PROBE DONE")
