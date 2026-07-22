# Purpose-built TRACED adaptive integrators: the ENTIRE adaptive solve loop --
# stages, FD block Jacobian, batched block linear solves, error norm, PI step
# controller, accept/reject -- executes inside one Reactant trace, with the time
# loop as a stablehlo.while (`Reactant.@trace while`). Only the loop skeleton and
# the batched 13x13 block solver are new; the stage algebra, error residual, and
# controller formulas are copied from the installed OrdinaryDiffEq sources:
#   * Rosenbrock23 stages: OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl
#     `perform_step!(integrator, ::Rosenbrock23ConstantCache)` (c32 = 6+sqrt(2),
#     d = 1/(2+sqrt(2)) from rosenbrock_tableaus.jl). dT is taken == 0 to match the
#     host driver's `tgrad = zero` (forcing is held constant inside a window).
#   * SSPRK43 stages: OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl
#     `perform_step!(integrator, ::SSPRK43ConstantCache)`.
#   * error residual + norm: DiffEqBase `calculate_residuals` and
#     `ODE_DEFAULT_NORM` (hairer norm, sqrt(sum(abs2)/N)).
#   * PI controller: OrdinaryDiffEqCore/src/integrators/controllers.jl
#     `stepsize_controller!/step_accept_controller!/step_reject_controller!`
#     (PIController), with the per-algorithm defaults queried from the installed
#     OrdinaryDiffEqCore (see pictrl_ros23/pictrl_ssprk43 below).
#
# Style rule: every ARRAY broadcast here is a single 2-3 argument operation per
# statement. Julia fuses broadcasts only within one expression, so one-op
# statements guarantee each broadcast reaches rx_native_patch.jl's `elem_apply`
# fast path (a native stablehlo op) instead of minting a per-site helper func.
#
# Layout contract: the state vector is SPECIES-MAJOR -- species (block) s of the
# NS species occupies rows (s-1)*NC+1 : s*NC, with the SAME cell ordering inside
# every block (the driver asserts this from var_map). Chemistry never couples
# cells, so its Jacobian is block-diagonal per cell; batched over cells, every
# operation on a "block matrix entry" is elementwise over length-NC vectors.

module RxTracedIntegrator

using Reactant
const RX = Reactant

# ---------------- controller settings (host-side constants) -------------------
Base.@kwdef struct PICtrl
    beta1::Float64
    beta2::Float64
    qmin::Float64 = 1 / 5
    qmax::Float64 = 10.0
    gamma::Float64 = 9 / 10
    qsteady_min::Float64 = 1.0
    qsteady_max::Float64
    qoldinit::Float64 = 1e-4
end
# OrdinaryDiffEqCore defaults for the two algorithms implemented here (queried
# from the installed package: beta2 = 2/(5*order), beta1 = 7/(10*order)).
pictrl_ros23() = PICtrl(; beta1=7 / 20, beta2=1 / 5, qsteady_max=6 / 5)
pictrl_ssprk43() = PICtrl(; beta1=7 / 30, beta2=2 / 15, qsteady_max=1.0)

# ---------------- batched block linear algebra --------------------------------
# Block s of a species-major vector.
_blk(u, s::Int, NC::Int) = u[((s - 1) * NC + 1):(s * NC)]

# Solve (per cell) W x = b: W is an NS x NS Matrix whose entries are length-NC
# traced vectors (one value per cell), b a length-NS Vector of the same.
# Unrolled Gaussian elimination WITHOUT pivoting -- W = I - dt*gamma*J is
# near-identity at chemistry step sizes, so pivot-free elimination is standard;
# the validation harness cross-checks against the host LU solve.
function blocksolve(W::Matrix, b::Vector)
    NS = length(b)
    W = copy(W); b = copy(b)
    for k in 1:NS
        piv = 1.0 ./ W[k, k]
        for i in (k + 1):NS
            m = W[i, k] .* piv
            for j in (k + 1):NS
                mw = m .* W[k, j]
                W[i, j] = W[i, j] .- mw
            end
            mb = m .* b[k]
            b[i] = b[i] .- mb
        end
    end
    x = Vector{Any}(undef, NS)
    for i in NS:-1:1
        acc = b[i]
        for j in (i + 1):NS
            wx = W[i, j] .* x[j]
            acc = acc .- wx
        end
        x[i] = acc ./ W[i, i]
    end
    return x
end

# FD block Jacobian, batched over cells: perturbing species s in EVERY cell at
# once fills column s of every cell's block in one RHS eval (NS extra evals per
# Jacobian). Same h formula as prototypes/reseact_3d_chem/block_jac.jl.
# `f` is u -> du at fixed (p, t); `f0b` the pre-sliced blocks of f(u);
# `masks[s]` a HOST Float64 vector that is 1.0 on block s and 0.0 elsewhere.
function fd_block_jac(f, u, f0b, NS::Int, NC::Int, masks)
    au = abs.(u)
    au = max.(au, 1.0e-9)
    hfull = sqrt(eps(Float64)) .* au
    J = Matrix{Any}(undef, NS, NS)
    for s in 1:NS
        dus = masks[s] .* hfull
        up = u .+ dus
        dup = f(up)
        hs = _blk(hfull, s, NC)
        for r in 1:NS
            num = _blk(dup, r, NC) .- f0b[r]
            J[r, s] = num ./ hs
        end
    end
    return J
end

# ---------------- Rosenbrock23 step (stiff, cell-local chemistry) -------------
const ROS23_d = 1 / (2 + sqrt(2))
const ROS23_c32 = 6 + sqrt(2)

# One ROS23 attempt from (u, t) with step dt. `f(u, t) -> du` (species-major).
# Returns (unew, EEst). Mirrors the Rosenbrock23ConstantCache perform_step! with
# mass_matrix = I and dT = 0; the three W solves are batched per-cell blocksolves.
function ros23_step(f, u, t, dt, NS::Int, NC::Int, masks, abstol::Float64, reltol::Float64)
    N = NS * NC
    f0 = f(u, t)
    f0b = [_blk(f0, r, NC) for r in 1:NS]
    J = fd_block_jac(uu -> f(uu, t), u, f0b, NS, NC, masks)
    dtgamma = dt * ROS23_d
    W = Matrix{Any}(undef, NS, NS)
    for s in 1:NS, r in 1:NS
        tmp = dtgamma .* J[r, s]
        W[r, s] = r == s ? 1.0 .- tmp : 0.0 .- tmp
    end
    # k1 = W \ f0   (textbook ode23s form of the OrdinaryDiffEq k1*neginvdtgamma)
    k1b = blocksolve(W, f0b)
    k1 = vcat(k1b...)
    dto2 = dt / 2
    du1 = dto2 .* k1
    u1 = u .+ du1
    f1 = f(u1, t + dto2)
    f1b = [_blk(f1, r, NC) for r in 1:NS]
    # k2 = W \ (f1 - k1) + k1
    b2 = Vector{Any}(undef, NS)
    for r in 1:NS
        b2[r] = f1b[r] .- k1b[r]
    end
    k2b = blocksolve(W, b2)
    for r in 1:NS
        k2b[r] = k2b[r] .+ k1b[r]
    end
    k2 = vcat(k2b...)
    du2 = dt .* k2
    unew = u .+ du2
    # embedded 3rd stage: k3 = W \ (f(unew) - c32*(k2-f1) - 2*(k1-f0))
    f2 = f(unew, t + dt)
    b3 = Vector{Any}(undef, NS)
    for r in 1:NS
        a = k2b[r] .- f1b[r]
        a = ROS23_c32 .* a
        c = k1b[r] .- f0b[r]
        c = 2.0 .* c
        d3 = _blk(f2, r, NC) .- a
        b3[r] = d3 .- c
    end
    k3b = blocksolve(W, b3)
    # utilde = dt/6 * (k1 - 2k2 + k3); EEst = hairer norm of the residuals
    dto6 = dt / 6
    sse = nothing
    for r in 1:NS
        w2 = 2.0 .* k2b[r]
        ut = k1b[r] .- w2
        ut = ut .+ k3b[r]
        ut = dto6 .* ut
        a0 = abs.(_blk(u, r, NC))
        a1 = abs.(_blk(unew, r, NC))
        am = max.(a0, a1)
        den = am .* reltol
        den = den .+ abstol
        at = ut ./ den
        at2 = at .* at
        s2 = sum(at2)
        sse = sse === nothing ? s2 : sse + s2
    end
    EEst = sqrt(sse / N)
    return unew, EEst
end

# ---------------- SSPRK43 step (non-stiff transport) --------------------------
# One SSPRK43 attempt; pure explicit stage algebra from the ConstantCache
# perform_step! (b = (1/6,1/6,1/6,1/2), bhat via utilde as in the source).
function ssprk43_step(f, u, t, dt, abstol::Float64, reltol::Float64)
    N = length(u)
    dt_2 = dt / 2
    k = f(u, t)
    du = dt_2 .* k
    u1 = u .+ du
    k = f(u1, t + dt_2)
    du = dt_2 .* k
    u2 = u1 .+ du
    k = f(u2, t + dt)
    du = dt_2 .* k
    u3 = u2 .+ du
    a = (1.0 / 3.0) .* u
    b = (2.0 / 3.0) .* u3
    utilde = a .+ b
    a = (2.0 / 3.0) .* u
    b = (1.0 / 3.0) .* u3
    u4 = a .+ b
    k = f(u4, t + dt_2)
    du = dt_2 .* k
    unew = u4 .+ du
    ud = utilde .- unew
    utilde = 0.5 .* ud
    a0 = abs.(u)
    a1 = abs.(unew)
    am = max.(a0, a1)
    den = am .* reltol
    den = den .+ abstol
    at = utilde ./ den
    at2 = at .* at
    sse = sum(at2)
    EEst = sqrt(sse / N)
    return unew, EEst
end

# ---------------- the traced adaptive loop ------------------------------------
# `stepfn(u, t, dt, aux) -> (unew, EEst)`. All of (u0, t0, tend, dt0) must be
# traced values (they are, inside an @compile). `aux` carries EVERY traced value
# the step needs beyond (u, t, dt) -- e.g. (p=..., bufs=...) -- and rides through
# the loop as carried state. This is a hard requirement, not a convenience:
# Ops.while_loop builds the while operands only from the loop variables the
# @trace macro collects syntactically, so any traced value captured inside a
# closure the body calls becomes an extra region block arg with NO matching
# operand and the stablehlo.while verifier rejects the module
# ("expect operands to be compatible with body block arguments").
# The loop compiles to ONE stablehlo.while whose body contains the step exactly
# once; a rejected step re-runs the body with the shrunken dt. Returns
# (u, t, dt, naccept, nreject) -- counters as Float64 traced scalars.
# `clamp_nonneg=true` clamps accepted states at 0 (cheap stand-in for the host
# driver's PositiveDomain callback; leave false when validating against a plain
# host solve).
function adaptive_solve(stepfn, u0, t0, tend, dt0, ctrl::PICtrl, aux;
        maxiters::Float64=2.0e4, clamp_nonneg::Bool=false)
    beta1 = ctrl.beta1; beta2 = ctrl.beta2
    invqmax = 1.0 / ctrl.qmax; invqmin = 1.0 / ctrl.qmin
    gamma = ctrl.gamma
    qsmin = ctrl.qsteady_min; qsmax = ctrl.qsteady_max
    qoldinit = ctrl.qoldinit
    # Resolve the clamp choice at TRACE time, outside the loop body: `@trace
    # while` promotes every host number the body references to a traced value
    # (Bool <: Number included), so a host `if clamp_nonneg` inside the body
    # would branch on a TracedRNumber{Bool} and throw.
    stepfn_ = clamp_nonneg ?
        ((uu, tt, dd, ax) -> begin
            un, ee = stepfn(uu, tt, dd, ax)
            (max.(un, 0.0), ee)
        end) : stepfn

    u = u0 .+ 0.0                    # fresh tracers for the loop-carried slots
    t = t0 + zero(t0)
    dt = dt0 + zero(dt0)
    qold = zero(t0) + qoldinit
    nacc = zero(t0)
    nrej = zero(t0)
    iters = zero(t0)
    tlim = tend - 1.0e-9
    RX.@trace while (t < tlim) & (iters < maxiters)
        aux = aux                    # keep stepfn's traced deps loop-carried
        dtc = min(dt, tend - t)      # never step past tend
        unew, EEst = stepfn_(u, t, dtc, aux)
        EEst = ifelse(isnan(EEst), 1.0e10, EEst)   # NaN step => hard reject
        # PIController: q11 = EEst^beta1; q = q11/qold^beta2, clamped
        q11 = max(EEst, 1.0e-35)^beta1
        qden = qold^beta2
        q = q11 / qden
        q = max(invqmax, min(invqmin, q / gamma))
        accept = EEst <= 1.0
        # accept: snap q inside the qsteady band to 1, then dt <- dtc/q
        insteady = (qsmin <= q) & (q <= qsmax)
        qa = ifelse(insteady, one(q), q)
        dt_acc = dtc / qa
        # reject: dt <- dtc / min(1/qmin, q11/gamma)
        qr = min(invqmin, q11 / gamma)
        dt_rej = dtc / qr
        u = ifelse.(accept, unew, u)
        t = ifelse(accept, t + dtc, t)
        dt = ifelse(accept, dt_acc, dt_rej)
        qold = ifelse(accept, max(EEst, qoldinit), qold)
        nacc = nacc + ifelse(accept, 1.0, 0.0)
        nrej = nrej + ifelse(accept, 0.0, 1.0)
        iters = iters + 1.0
    end
    return u, t, dt, nacc, nrej
end

# ---------------- host-side layout helper -------------------------------------
# Derive the species-major block layout from var_map and build the per-species
# perturbation masks the traced FD Jacobian needs. Asserts every species block
# is contiguous, NC-aligned, and shares ONE cell ordering (the batched Jacobian
# and block solves silently misalign otherwise). Pure host code.
function species_masks(var_map, NS::Int, NC::Int)
    N = NS * NC
    bygroup = Dict{String,Vector{Tuple{NTuple{3,Int},Int}}}()
    for (nm, idx) in var_map
        mm = match(r"^(.*)\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && error("unparseable state name '$nm'")
        c = (parse(Int, mm.captures[2]), parse(Int, mm.captures[3]), parse(Int, mm.captures[4]))
        push!(get!(bygroup, String(mm.captures[1]), Tuple{NTuple{3,Int},Int}[]), (c, idx))
    end
    length(bygroup) == NS || error("expected $NS species, got $(length(bygroup))")
    masks = Vector{Vector{Float64}}(undef, NS)
    seen = falses(NS); cellorder_ref = nothing
    for (b, v) in bygroup
        idxs = sort!([x[2] for x in v])
        idxs == collect(idxs[1]:(idxs[1] + NC - 1)) || error("species $b not contiguous")
        (idxs[1] - 1) % NC == 0 || error("species $b block misaligned")
        sblk = (idxs[1] - 1) ÷ NC + 1
        seen[sblk] && error("duplicate block index for $b")
        seen[sblk] = true
        co = [x[1] for x in sort(v; by = x -> x[2])]
        if cellorder_ref === nothing
            cellorder_ref = co
        else
            co == cellorder_ref || error("cell order differs for species $b")
        end
        mask = zeros(N); mask[idxs[1]:idxs[end]] .= 1.0
        masks[sblk] = mask
    end
    all(seen) || error("missing species blocks")
    return masks
end

end # module
