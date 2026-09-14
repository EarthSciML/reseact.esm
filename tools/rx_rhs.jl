# ===========================================================================
# rx_rhs.jl -- THE ONE PLACE that decides which compiled right-hand side every
# production entry point in this repository runs on.
# ===========================================================================
#
#   RESEACT_RHS=traced   (default)  EarthSciAST.rhs_with_buffers(f)
#   RESEACT_RHS=direct              EarthSciASTReactantExt.direct_rhs_with_buffers(f)
#
# `traced` is the lane every recorded result in this repository was produced on:
# the `:oop` build product is CALLED under `Reactant.@compile`, and Reactant
# traces the broadcast emitter to get StableHLO. `direct` hands the same `:oop`
# build product to EarthSciAST's direct StableHLO emitter, which constructs the
# module from the compiled tree-walk IR without tracing Julia broadcasts at all.
# Everything on either side of this seam -- the build, the operator split, the
# forcing buffers, the integrators, the symbolic Jacobian gather, the adjoint --
# is the same code in both lanes. That is the point: a difference downstream can
# only be the emitter.
#
# WHY A SEAM AND NOT A FLAG AT EACH SITE. There are six production entry points
# and each builds several compiled programs (the two operator-split halves, the
# symbolic Jacobian band model, one per capacity rung, one per chemistry shard).
# A gradient is only an oracle for the other lane if EVERY one of those went the
# same way, and the failure mode of a missed site is silent: a mixed run
# compiles, runs and answers, and the answer is a comparison of the traced lane
# with itself.
#
# FOUR ENTRY POINTS, AND THE DISTINCTION BETWEEN THEM MATTERS:
#
#   rx_rhs(f; var_map)    the COMPILED right-hand side, 4-argument form
#                         `(u, p, t, buffers)`. Switchable. Only ever called
#                         inside a `Reactant.@compile` trace -- the direct
#                         wrapper REFUSES a host call by design, because it
#                         builds StableHLO and has nothing to run on the host.
#   rx_host_rhs(f)        the same right-hand side for a HOST evaluation on
#                         ordinary `Vector{Float64}`s. NEVER switchable: the
#                         base-point finiteness guards and
#                         `RxSymBlockJac.validate_plan` are host checks of the
#                         BUILD, not of the emitter, and they must keep working
#                         identically in both lanes so that a direct run is
#                         still gated by them.
#   rx_bufs(f)            the forcing-buffer container. Both lanes take the
#                         buffers as a fourth ARGUMENT aligned with
#                         `forcing_buffers(f)` in the same name-sorted order, so
#                         this is `forcing_buffers` today; it is here so that a
#                         future divergence has one place to happen.
#   rx_sync!(dev, f)      push this build's refreshed host buffers to `dev`.
#
# `var_map` is optional and costs nothing. In the direct lane it NAMES the
# originating rule when the emitter refuses a construct; without it a refusal
# says "flat slot 4705" and someone has to go and find out what that is.
# ===========================================================================

module RxRHS

using EarthSciAST
const EA = EarthSciAST

export rx_rhs, rx_host_rhs, rx_bufs, rx_sync!, rx_rhs_mode, rx_rhs_direct, rx_rhs_banner

const MODE = let s = lowercase(strip(get(ENV, "RESEACT_RHS", "traced")))
    s in ("traced", "direct") ||
        error("RESEACT_RHS must be `traced` or `direct`, got \"$s\"")
    Symbol(s)
end

"`:traced` or `:direct` -- which compiled lane this process is running."
rx_rhs_mode() = MODE

"True when this process compiles through the direct StableHLO emitter."
rx_rhs_direct() = MODE === :direct

# Resolved once: `get_extension` is a lookup, but the two error messages below
# are the ones that separate "the emitter refused this model" from "this
# environment has no emitter", and they are worth stating exactly once.
const _EXT = Ref{Any}(nothing)
function _ext()
    _EXT[] === nothing || return _EXT[]
    E = Base.get_extension(EarthSciAST, :EarthSciASTReactantExt)
    E === nothing && error(
        "RESEACT_RHS=direct: EarthSciAST's Reactant extension is not loaded. " *
        "`using Reactant` must come before the first rx_rhs call.")
    isdefined(E, :direct_rhs_with_buffers) || error(
        "RESEACT_RHS=direct: this EarthSciAST has no direct StableHLO emitter " *
        "(`EarthSciASTReactantExt.direct_rhs_with_buffers` is not defined). " *
        "The direct lane needs an EarthSciAST carrying ext/reactant_direct/; " *
        "point the project at one, or run with RESEACT_RHS=traced.")
    _EXT[] = E
    return E
end

"""
    rx_rhs(f; var_map = nothing)

The compiled 4-argument right-hand side `(u, p, t, buffers) -> du` for the
`:oop` build product `f`, in whichever lane `RESEACT_RHS` selects. Call it only
inside a `Reactant.@compile` trace; use [`rx_host_rhs`](@ref) for a host
evaluation.
"""
rx_rhs(f; var_map = nothing) =
    MODE === :direct ? _ext().direct_rhs_with_buffers(f; var_map = var_map) :
                       EA.rhs_with_buffers(f)

"""
    rx_host_rhs(f)

The 4-argument right-hand side for a HOST evaluation on ordinary arrays. Always
the traced-lane callable, in both lanes; see the header.
"""
rx_host_rhs(f) = EA.rhs_with_buffers(f)

"The live forcing buffers of `f`, in the order both lanes expect them."
rx_bufs(f) = EA.forcing_buffers(f)

"Push `f`'s refreshed host forcing buffers into the device container `dev`."
rx_sync!(dev, f) = EA.sync_forcing!(dev, EA.forcing_buffers(f))

"""
    rx_rhs_banner() -> String

One line naming the lane and the packages that decide what it means, for the
top of a driver's log. A run whose log does not say which lane it took cannot
be compared with anything later.
"""
function rx_rhs_banner()
    ea = try string(pkgversion(EarthSciAST)) catch; "?" end
    rx = try
        RXM = Base.loaded_modules[Base.PkgId(
            Base.UUID("3c362404-f566-11ee-1572-e11a4b42c853"), "Reactant")]
        string(pkgversion(RXM))
    catch
        "(not loaded)"
    end
    return "RHS LANE: $(MODE)   EarthSciAST $ea   Reactant $rx" *
           (MODE === :direct ? "   (direct StableHLO emitter, no traced fallback)" : "")
end

end # module

using .RxRHS
