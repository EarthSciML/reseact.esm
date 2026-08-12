# j04 -- WHERE in the step does the NaN enter?
#
# j03: ros_step faults on 9 of 20000 calls, always the same single cell (1,1,1),
# always all 13 species of it, always NaN, never anything else -- and in 40k
# calls the ONLY output deviations of any kind were those 9. So the step is a
# bit-exact function except when one cell's whole 13x13 block collapses.
#
# A whole cell block going NaN at once is what an unpivoted 13x13 elimination
# does when ONE lane of ONE intermediate is bad, so this job compiles two more
# programs to bracket it:
#   A  the chemistry RHS alone. If that faults, the fault is in the emitted
#      model, not the integrator algebra.
#   B  a probe copy of ros23_step (same unrolled/:fd path) that RETURNS its
#      intermediates, so on a fault we can read off the first stage that is NaN.
# Also records EEst on every fault: a NaN EEst is exactly what would turn this
# fault into DIFFERENTIABILITY_PLAN.md's FINDING 1 (a replay taking a different
# number of accepted steps), because host_adaptive! maps isnan(EEst) -> reject.
const J4RHS = parse(Int, get(ENV, "J4RHS", "40000"))
const J4PRB = parse(Int, get(ENV, "J4PRB", "20000"))
const CELLOF = [((i - 1) % NC) + 1 for i in 1:N]

# ---------------- A: the chemistry RHS on its own ----------------------------
rhs_only(u, th, t) = gC(u, th, t)
rhs_onlyT(u, th, t) = gT(u, th, t)
CRHS  = timed_compile("rhs_chem",  () -> RX.@compile sync=true rhs_only(U_R, THC, T_R))
CRHST = timed_compile("rhs_tran",  () -> RX.@compile sync=true rhs_onlyT(U_R, THT, T_R))

function soak_rhs(nm, c, TH, u, t, K)
    ref = nothing; nbad = 0; ndiff = 0; t0 = time()
    for k in 1:K
        out = Array(c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t)))
        b = findall(!isfinite, out)
        d = ref === nothing ? Int[] : bitdiff(ref, out)
        if !isempty(b) || !isempty(d)
            isempty(b) || (nbad += 1); isempty(d) || (ndiff += 1)
            say("  !! $nm k=$k nonfinite=$(length(b)) bitdiff=$(length(d)) " *
                "cells=$(sort(unique(CELLOF[i] for i in vcat(b,d)))) load=$(loadavg())")
            flush(stdout)
        end
        ref === nothing && (ref = out)
        k % 10000 == 0 && (say("    $nm $k/$K bad=$nbad diff=$ndiff $(round(time()-t0,digits=1))s"); flush(stdout))
    end
    say("  RESULT rhs $nm K=$K nonfinite=$nbad bitdiff=$ndiff s_per_call=$(round((time()-t0)/K,digits=6))")
end

say("---- j04A: the RHS alone ----")
soak_rhs("rhs_chem/stepped", CRHS, THC, USTEP, T0 + DT0C, J4RHS)
soak_rhs("rhs_tran/stepped", CRHST, THT, USTEP, T0 + DT0C, J4RHS)

# ---------------- B: ros23 with its intermediates exposed --------------------
# A verbatim copy of RxTracedIntegrator.ros23_step's unrolled/:fd path that also
# returns f0, the 13 perturbed RHS evaluations, J, W and the three k stages.
function ros_probe(u, th, t, dt)
    f(uu) = gC(uu, th, t)
    f0 = gC(u, th, t)
    f0b = [RTI._blk(f0, r, NC) for r in 1:NS]
    au = max.(abs.(u), 1.0e-9)
    hfull = sqrt(eps(Float64)) .* au
    dups = Vector{Any}(undef, NS)
    J = Matrix{Any}(undef, NS, NS)
    for s in 1:NS
        dus = MASKS[s] .* hfull
        dup = f(u .+ dus)
        dups[s] = dup
        hs = RTI._blk(hfull, s, NC)
        for r in 1:NS
            J[r, s] = (RTI._blk(dup, r, NC) .- f0b[r]) ./ hs
        end
    end
    dtgamma = dt * RTI.ROS23_d
    W = Matrix{Any}(undef, NS, NS)
    for s in 1:NS, r in 1:NS
        tmp = dtgamma .* J[r, s]
        W[r, s] = r == s ? 1.0 .- tmp : 0.0 .- tmp
    end
    k1b = RTI.blocksolve(W, f0b)
    k1 = vcat(k1b...)
    u1 = u .+ (dt / 2) .* k1
    f1 = gC(u1, th, t + dt / 2)
    f1b = [RTI._blk(f1, r, NC) for r in 1:NS]
    k2b = RTI.blocksolve(W, [f1b[r] .- k1b[r] for r in 1:NS])
    for r in 1:NS; k2b[r] = k2b[r] .+ k1b[r]; end
    k2 = vcat(k2b...)
    unew = u .+ dt .* k2
    f2 = gC(unew, th, t + dt)
    b3 = [(RTI._blk(f2, r, NC) .- RTI.ROS23_c32 .* (k2b[r] .- f1b[r])) .- 2.0 .* (k1b[r] .- f0b[r])
          for r in 1:NS]
    k3b = RTI.blocksolve(W, b3)
    k3 = vcat(k3b...)
    Jflat = vcat((J[r, s] for s in 1:NS for r in 1:NS)...)
    Wflat = vcat((W[r, s] for s in 1:NS for r in 1:NS)...)
    return unew, f0, vcat(dups...), Jflat, Wflat, k1, f1, k2, f2, k3
end

CPRB = timed_compile("ros_probe", () -> RX.@compile sync=true ros_probe(U_R, THC, T_R, DTC_R))

const PRBNAMES = ("unew", "f0", "dups", "J", "W", "k1", "f1", "k2", "f2", "k3")

function soak_probe(nm, u, t, dt, K)
    ref = nothing; nbad = 0; t0 = time()
    for k in 1:K
        r = CPRB(RX.ConcreteRArray(u), THC, RX.ConcreteRNumber(t), RX.ConcreteRNumber(dt))
        outs = [Array(x) for x in r]
        anybad = any(o -> any(!isfinite, o), outs)
        d = ref === nothing ? 0 : sum(length(bitdiff(ref[i], outs[i])) for i in 1:length(outs))
        if anybad || d > 0
            nbad += 1
            say("  !! $nm k=$k load=$(loadavg())")
            for (i, o) in enumerate(outs)
                b = findall(!isfinite, o)
                dd = ref === nothing ? Int[] : bitdiff(ref[i], o)
                if !isempty(b) || !isempty(dd)
                    lanes = sort(unique(vcat([((j-1) % NC)+1 for j in b], [((j-1) % NC)+1 for j in dd])))
                    say("     $(PRBNAMES[i]) (len $(length(o))): nonfinite=$(length(b)) " *
                        "bitdiff=$(length(dd)) lanes=$(lanes[1:min(end,20)])")
                end
            end
            flush(stdout)
        end
        ref === nothing && (ref = outs)
        k % 2500 == 0 && (say("    $nm $k/$K bad=$nbad $(round(time()-t0,digits=1))s"); flush(stdout))
    end
    say("  RESULT probe $nm K=$K faults=$nbad s_per_call=$(round((time()-t0)/K,digits=5))")
end

say("---- j04B: ros23 with intermediates exposed ----")
soak_probe("ros_probe/stepped", USTEP, T0 + DT0C, DT0C, J4PRB)
