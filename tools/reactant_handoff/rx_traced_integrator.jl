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
# Trace-time rule: Julia trace time is LINEAR in the number of traced RHS call
# sites per loop body, so repeated stages run as NESTED `Reactant.@trace for`
# loops with `f` appearing once (ssprk43_step stages 1-3; fd_block_jac's
# species loop): 2 transport + 4 chem RHS sites per window instead of 4 + 16.
# Two rules for those nested loop bodies, both regression-covered by
# rx_traced_smoke.jl:
#   * `f = f` self-assignment: a closure that is only CALLED in the body is not
#     syntactically collected by @trace, so its traced captures would surface
#     as free region values; self-assigning forces collection (the nested-loop
#     analogue of adaptive_solve's `aux = aux`).
#   * track_numbers=false: the bodies reference host Ints (N, NS, NC) as
#     static reshape dims; the default track_numbers=Number would promote them
#     to traced scalars and break reshape.
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
#
# TRACED-LOOP version: the species loop is a `Reactant.@trace for`, so `f`
# appears exactly ONCE in the emitted module (vs NS unrolled copies) -- Julia
# trace time is linear in RHS call sites, so this is the main trace-time win.
# `f` is u -> du at fixed (p, t); `f0` the full f(u) vector. The species masks
# are rebuilt per-iteration from a traced block-id vector (blkid[i] = species
# block of row i, derivable from the asserted species-major layout), because a
# host masks[] vector cannot be indexed by a traced loop variable.
# track_numbers=false: the body references host Ints (N, NS, NC) as reshape
# dims, which must stay static -- the default track_numbers=Number would
# promote them to traced and break reshape.
function fd_block_jac(f, u, f0, NS::Int, NC::Int)
    N = NS * NC
    au = abs.(u)
    au = max.(au, 1.0e-9)
    hfull = sqrt(eps(Float64)) .* au
    blkid = RX.Ops.constant(repeat(Float64.(1:NS); inner=NC))   # length N
    cidrow = RX.Ops.constant(reshape(Float64.(1:NS), 1, NS))    # 1 x NS
    J = RX.Ops.fill(0.0, (N, NS))                               # loop-carried
    RX.@trace track_numbers=false for s in 1:NS
        f = f          # closures are CALLED, not referenced, in the body; the
        #                self-assignment forces @trace to collect `f` as a loop
        #                variable so its traced captures become while operands
        #                (same reason as the `aux = aux` in adaptive_solve).
        sf = 1.0 * s
        eqm = blkid .== sf
        mask = eqm .* 1.0                # Float64 mask, native-multiply path
        dus = mask .* hfull              # h for block s, 0 elsewhere
        up = u .+ dus
        dup = f(up)
        num = dup .- f0
        # per-cell h of species s: reshape (species-major => column s of the
        # NC x NS matrix is block s), reduce the zero-padded columns away.
        hmat = reshape(dus, NC, NS)
        hs = sum(hmat; dims=2)           # NC x 1
        numm = reshape(num, NC, NS)
        colm = numm ./ hs                # every row-block divided by hs
        col = reshape(colm, N)
        # deposit as column s of J via a one-hot row (dynamic column index)
        eqr = cidrow .== sf
        srow = eqr .* 1.0                # 1 x NS one-hot
        upd = col .* srow                # N x NS, only column s nonzero
        J = J .+ upd
    end
    # static slicing back into the NS x NS blocks blocksolve consumes
    Jb = Matrix{Any}(undef, NS, NS)
    for si in 1:NS, r in 1:NS
        Jb[r, si] = J[((r - 1) * NC + 1):(r * NC), si]
    end
    return Jb
end

# Reference implementation with the species loop unrolled on the host (NS
# traced copies of `f`). Kept verbatim for A/B trace-time and bit-equality
# diagnostics; select with `ros23_step(...; unrolled=true)`.
function fd_block_jac_unrolled(f, u, f0b, NS::Int, NC::Int, masks)
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
# `unrolled=true` selects the reference host-unrolled Jacobian (16 traced RHS
# sites per step); the default traced-loop Jacobian has 4 (f0, jac loop, f1, f2).
# `masks` is only consumed by the unrolled path; the traced path derives the
# layout from (NS, NC) directly (species_masks has already asserted it).
function ros23_step(f, u, t, dt, NS::Int, NC::Int, masks, abstol::Float64, reltol::Float64;
        unrolled::Bool=false)
    N = NS * NC
    f0 = f(u, t)
    f0b = [_blk(f0, r, NC) for r in 1:NS]
    J = unrolled ? fd_block_jac_unrolled(uu -> f(uu, t), u, f0b, NS, NC, masks) :
        fd_block_jac(uu -> f(uu, t), u, f0, NS, NC)
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
#
# Stages 1-3 are the SAME affine update u <- u + (dt/2)*f(u, t + c_i*dt) with
# c = (0, 1/2, 1), so they run as a `Reactant.@trace for` with ONE traced copy
# of `f` in the body (c_i = (i-1)/2 computed from the traced loop index); the
# distinct 4th stage is the second and last RHS site. 2 traced RHS sites per
# step instead of 4. Bitwise-identical to the unrolled form: 0.0*dt == 0.0,
# 0.5*dt == dt/2, 1.0*dt == dt exactly, and t + 0.0 == t for t >= 0.
# `unrolled=true` selects the reference 4-site version for A/B diagnostics.
function ssprk43_step(f, u, t, dt, abstol::Float64, reltol::Float64;
        unrolled::Bool=false)
    unrolled && return ssprk43_step_unrolled(f, u, t, dt, abstol, reltol)
    N = length(u)
    dt_2 = dt / 2
    u3 = u .+ 0.0     # fresh tracer: loop-carried stage state must not alias u
    RX.@trace track_numbers=false for i in 1:3
        f = f         # force `f` (only CALLED below) to be loop-collected so
        #               its traced captures ride as while operands -- see the
        #               matching comment in fd_block_jac.
        ci = 0.5 * (i - 1)
        toff = ci * dt
        ts = t + toff
        k = f(u3, ts)
        du = dt_2 .* k
        u3 = u3 .+ du
    end
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

# Reference implementation with all 4 stages unrolled (4 traced copies of `f`).
# Kept verbatim for A/B trace-time and bit-equality diagnostics.
function ssprk43_step_unrolled(f, u, t, dt, abstol::Float64, reltol::Float64)
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
