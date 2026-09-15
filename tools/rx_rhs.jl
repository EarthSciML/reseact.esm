# ===========================================================================
# rx_rhs.jl -- THE ONE PLACE that builds the compiled right-hand side every
# production entry point in this repository runs on.
# ===========================================================================
#
#   rx_rhs(f; var_map)  ->  EarthSciASTReactantExt.direct_rhs_with_buffers(f)
#
# The `:oop` build product is the compiled tree-walk IR a backend lowers, and
# EarthSciAST's direct StableHLO emitter is the backend: it constructs the
# module from that IR, operation by operation, with no Julia broadcast trace in
# between and no fallback. A construct the emitter cannot lower is a hard
# `EarthSciAST.DirectEmitError` naming the construct and the rule it came from.
#
# THERE IS ONE LANE. `build_evaluator(doc; form = :oop)` no longer returns
# anything callable on the host -- calling it raises
# `E_TREEWALK_OOP_NOT_EVALUABLE` -- and `EarthSciAST.rhs_with_buffers`, the
# traced emitter's entry point, is gone. `RESEACT_RHS=traced` is therefore not a
# slower option, it is a request for code that no longer exists, and this file
# refuses it by name rather than quietly running the direct lane under a log
# line that says otherwise.
#
# FIVE ENTRY POINTS, AND THE DISTINCTION BETWEEN THEM MATTERS:
#
#   rx_rhs(f; var_map)    the COMPILED right-hand side, 4-argument form
#                         `(u, p, t, buffers)`. Only ever called inside a
#                         `Reactant.@compile` trace -- it builds StableHLO and
#                         has nothing to run on the host.
#   rx_host_eval(f, p, t, bufs; var_map, compile_options)
#                         a callable `u -> du::Vector{Float64}` for the host
#                         CHECKS -- the base-point finiteness guards and
#                         `RxSymBlockJac.validate_plan`. It compiles `rx_rhs(f)`
#                         once on first use and then does a host->device->host
#                         round trip per point, which is what "evaluate this
#                         build at one point" costs now that the build product
#                         is IR rather than a callable. Deliberately NOT the
#                         same spelling as the old `rx_host_rhs`, so a site
#                         that was missed is a MethodError rather than a
#                         silently different check.
#   rx_bufs(f)            the forcing-buffer container, aligned with
#                         `forcing_buffers(f)` in the same name-sorted order.
#   rx_sync!(dev, f)      push this build's refreshed host buffers to `dev`.
#   rx_devp(p)            a NamedTuple of parameters as device scalars.
#
# `var_map` is optional and costs nothing. It NAMES the originating rule when
# the emitter refuses a construct; without it a refusal says "flat slot 4705"
# and someone has to go and find out what that is. It is also what the cell-axis
# shard checks a slab cut against.
# ===========================================================================

module RxRHS

using EarthSciAST
using Reactant
const EA = EarthSciAST
const RX = Reactant

export rx_rhs, rx_host_eval, rx_bufs, rx_sync!, rx_devp, rx_rhs_banner

# A stale `RESEACT_RHS` in an environment, a shell history or an sbatch file
# must not be able to change what runs. `direct` names the only lane there is
# and is accepted so that jobs written while the switch existed still launch;
# anything else -- `traced` above all -- is refused with the reason.
let s = lowercase(strip(get(ENV, "RESEACT_RHS", "direct")))
    s == "traced" && error(
        "RESEACT_RHS=traced: the traced lane is retired. EarthSciAST no longer " *
        "has `rhs_with_buffers`, and the `:oop` build product is not callable " *
        "on the host at all (`E_TREEWALK_OOP_NOT_EVALUABLE`). The compiled " *
        "right-hand side is EarthSciAST's direct StableHLO emitter; unset " *
        "RESEACT_RHS.")
    s == "direct" || error(
        "RESEACT_RHS=\"$s\": the only compiled lane is the direct StableHLO " *
        "emitter. Unset RESEACT_RHS.")
end

# Resolved once: `get_extension` is a lookup, but the two error messages below
# are the ones that separate "the emitter refused this model" from "this
# environment has no emitter", and they are worth stating exactly once.
const _EXT = Ref{Any}(nothing)
function _ext()
    _EXT[] === nothing || return _EXT[]
    E = Base.get_extension(EarthSciAST, :EarthSciASTReactantExt)
    E === nothing && error(
        "EarthSciAST's Reactant extension is not loaded. `using Reactant` must " *
        "come before the first rx_rhs call.")
    isdefined(E, :direct_rhs_with_buffers) || error(
        "this EarthSciAST has no direct StableHLO emitter " *
        "(`EarthSciASTReactantExt.direct_rhs_with_buffers` is not defined). " *
        "Point the project at an EarthSciAST carrying ext/reactant_direct/.")
    _EXT[] = E
    return E
end

"""
    rx_rhs(f; var_map = nothing)

The compiled 4-argument right-hand side `(u, p, t, buffers) -> du` for the
`:oop` build product `f`. Call it only inside a `Reactant.@compile` trace; use
[`rx_host_eval`](@ref) to evaluate this build at a point on the host.
"""
rx_rhs(f; var_map = nothing) = _ext().direct_rhs_with_buffers(f; var_map = var_map)

"The live forcing buffers of `f`, in the order the compiled program expects."
rx_bufs(f) = EA.forcing_buffers(f)

"Push `f`'s refreshed host forcing buffers into the device container `dev`."
rx_sync!(dev, f) = EA.sync_forcing!(dev, EA.forcing_buffers(f))

"A parameter NamedTuple as device scalars, the form every compiled program takes."
rx_devp(p::NamedTuple) = NamedTuple{keys(p)}(map(RX.ConcreteRNumber, values(p)))
rx_devp(p) = p

"""
    rx_host_eval(f, p, t, host_bufs; var_map = nothing, compile_options = nothing)
        -> (u::AbstractVector -> Vector{Float64})

A HOST evaluation of the `:oop` build `f` at the fixed parameters `p`, time `t`
and forcing buffers `host_bufs`: compile `rx_rhs(f)` on first call and evaluate
it at any host state vector, returning an ordinary `Vector{Float64}`.

This is what the checks OF THE BUILD run on -- the base-point finiteness guards
and `RxSymBlockJac.validate_plan`. They used to call the `:oop` product itself
on `Vector{Float64}`s; an `:oop` build is now the compiled IR a backend lowers
and raises `E_TREEWALK_OOP_NOT_EVALUABLE` when called, so the one evaluation
each check needs is the compiled program, run once. The cost is one `@compile`
of the program being checked plus a host->device->host round trip per point;
the alternative, an `f!` build of the same document for the sake of a guard, is
a second build of the whole model.

The returned closure compiles LAZILY, so a check that never evaluates costs
nothing. `compile_options` is forwarded to `@compile` when given; with the
default the XLA:CPU race workaround (`tools/diag/README-nondet.md`) is off,
which for a one-shot check can only produce a false ALARM -- a non-finite
derivative or a mismatch aborts the build -- never a false clearance.
"""
function rx_host_eval(f, p, t::Real, host_bufs; var_map = nothing,
                      compile_options = nothing)
    d    = rx_rhs(f; var_map = var_map)
    pr   = rx_devp(p)
    tr   = RX.ConcreteRNumber(Float64(t))
    br   = map(RX.ConcreteRArray, host_bufs)
    prog = Ref{Any}(nothing)
    copts = compile_options
    return function (u::AbstractVector)
        ur = RX.ConcreteRArray(Vector{Float64}(u))
        if prog[] === nothing
            prog[] = copts === nothing ? (RX.@compile d(ur, pr, tr, br)) :
                                         (RX.@compile compile_options=copts d(ur, pr, tr, br))
        end
        return Array(prog[](ur, pr, tr, br))
    end
end

"""
    rx_rhs_banner() -> String

One line naming the lane and the packages that decide what it means, for the
top of a driver's log. A run whose log does not say what it compiled through
cannot be compared with anything later.
"""
function rx_rhs_banner()
    ea = try string(pkgversion(EarthSciAST)) catch; "?" end
    rx = try string(pkgversion(RX)) catch; "?" end
    return "RHS LANE: direct StableHLO emitter (no traced lane)   " *
           "EarthSciAST $ea   Reactant $rx"
end

# EarthSciAST compiled-IR objects are OPAQUE host data to a trace: _Node trees,
# acc kernels and oop plans hold only host Ints/Float64 tables and never a
# traced value, yet Reactant's `make_tracer` capture walk recurses the whole
# object graph of every callable argument -- and the compiled right-hand side
# IS that graph, since `direct_rhs_with_buffers` wraps the `:oop` build product
# itself. The walk is O(IR size) per `@compile`; on the merged transport RHS
# (1.06M nodes) it costs tens of minutes. Registering the IR types as leaves
# (precedent: Reactant's own `make_tracer(::Union{ExceptionStack,MethodInstance})`)
# makes the walk skip them. Under TracedToTypes -- the compile-cache key mode --
# the object itself is pushed, so distinct kernel sets keep distinct keys.
#
# This lived in `tools/reactant_handoff/rx_native_patch.jl` beside the traced
# emitter's broadcast lowering, which retired with the traced lane. It belongs
# here instead: it is a property of the compiled IR, not of how a module gets
# emitted from it, and it is needed by every entry point this file serves.
for _T in (:_Node, :_AccKernel, :_OopAccPlan, :_AccScratch)
    isdefined(EA, _T) || continue
    @eval function RX.make_tracer(seen, @nospecialize(prev::$(getfield(EA, _T))),
                                  @nospecialize(path), mode; kwargs...)
        mode == RX.TracedToTypes && (push!(path, prev); return nothing)
        return prev
    end
end

end # module

using .RxRHS
