#!/usr/bin/env julia
# ===========================================================================
# nondet_hammer.jl -- is the compiled ReSEACT step program deterministic?
# ===========================================================================
# DIFFERENTIABILITY_PLAN.md's Phase 4 section records two nondeterminism
# findings against the compiled single-step programs:
#
#   FINDING 1  a replay of a macro step from its own checkpoint took a
#              DIFFERENT number of accepted steps than the forward pass, same
#              process, same theta -- attributed to "the compiled ROS23 step is
#              not bit-deterministic call to call".
#   FINDING 2  ~2 in 1500 calls come back with non-finite entries (312 at once
#              = 24 cells x 13 state groups), and re-issuing the identical call
#              returns a finite answer.
#
# Both are claims about the SAME object: f(u, th, t, dt) as a function. This
# harness tests that function directly. It reuses tools/adjoint_gradient.jl's
# build and compile verbatim (STAGES=none runs the build and the two primal
# compiles and nothing else), so whatever it measures is measured about the
# same compiled programs the driver uses.
#
# It answers, in ONE process (the build is the expensive part):
#   E1  bit-determinism: the same (u,th,t,dt) called K times, fresh input
#       buffers each call. #distinct outputs, and where they differ.
#   E2  the same, reusing ONE input ConcreteRArray object.
#   E3  a soak: many calls, hunting a non-finite output; on a hit, dump the
#       index pattern and IMMEDIATELY re-issue the identical call to test
#       reproducibility from the same host bits.
#   E4  a walk: the same as E3 but along a real trajectory (each call's output
#       feeds the next), which is what the driver actually does -- in case the
#       fault needs a state the base point never visits.
#
# Env:
#   RESEACT_HAMMER_K       calls per determinism cell   (default 200)
#   RESEACT_HAMMER_SOAK    calls in the soak            (default 4000)
#   RESEACT_HAMMER_WALK    steps in the walk            (default 2000)
#   RESEACT_HAMMER_PROGS   subset of ssp,ros            (default "ssp,ros")
#   RESEACT_HAMMER_STAGES  subset of e1,e2,soak,walk    (default all)
#   RESEACT_HAMMER_VJP     1 = also compile+hammer the VJPs (adds ~300 s)
#   plus every env tools/adjoint_gradient.jl takes.
# ===========================================================================
ENV["RESEACT_ADJ_STAGES"] = get(ENV, "RESEACT_ADJ_STAGES", "none")
const _WANT_VJP = get(ENV, "RESEACT_HAMMER_VJP", "0") == "1"
_WANT_VJP && (ENV["RESEACT_ADJ_STAGES"] = "adj")   # makes the driver compile the VJPs
include(joinpath(@__DIR__, "..", "adjoint_gradient.jl"))

using Printf, Random, Dates

const HK    = parse(Int, get(ENV, "RESEACT_HAMMER_K", "200"))
const HSOAK = parse(Int, get(ENV, "RESEACT_HAMMER_SOAK", "4000"))
const HWALK = parse(Int, get(ENV, "RESEACT_HAMMER_WALK", "2000"))
const HPROGS = Set(String.(split(get(ENV, "RESEACT_HAMMER_PROGS", "ssp,ros"), ',')))
const HST   = Set(String.(split(get(ENV, "RESEACT_HAMMER_STAGES", "e1,e2,soak,walk"), ',')))
hwant(s) = s in HST

loadavg() = try; parse(Float64, split(read("/proc/loadavg", String))[1]); catch; NaN; end

# --------------------------------------------------------------------------- #
# The call under test, as the driver issues it: a FRESH ConcreteRArray per call.
# --------------------------------------------------------------------------- #
callstep(c, u::Vector{Float64}, TH, t, dt) =
    c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t), RX.ConcreteRNumber(dt))

# bit-exact difference report between two Float64 vectors
function bitdiff(a::Vector{Float64}, b::Vector{Float64})
    idx = findall(i -> !(a[i] === b[i]), eachindex(a))
    return idx
end

function cellsig(idx::Vector{Int})
    # PERM maps state index -> (species, cell); report the cells and groups hit
    groups = Set{String}(); cells = Set{Int}()
    for i in idx
        push!(groups, GROUP_OF[i])
        # cell-major perm: recover the cell from the variable name
        push!(cells, i)
    end
    return length(cells), sort(collect(groups))
end

say("\n" * "="^75)
say("NONDET HAMMER  K=$HK soak=$HSOAK walk=$HWALK progs=$(join(sort(collect(HPROGS)),','))")
say("  julia threads=$(Threads.nthreads())  loadavg=$(loadavg())  $(now())")
say("="^75)

# A realistic point: UBASE, and UBASE advanced by one transport + one chem step.
const PT = Dict{String,Any}()
PT["base"] = (u = copy(UBASE), t = T0)
let r = callstep(CSSP, UBASE, THT, T0, DT0T)
    u1 = Array(r[1])
    r2 = callstep(CROS, u1, THC, T0, DT0C)
    PT["stepped"] = (u = Array(r2[1]), t = T0 + DT0C)
end
say("  points: " * join([@sprintf("%s(|u|=%.4g, nonfinite=%d)", k, maximum(abs, v.u),
                                  count(!isfinite, v.u)) for (k, v) in PT], "  "))

const PROGS = Any[]
"ssp" in HPROGS && push!(PROGS, ("ssp_step", CSSP, THT, DT0T))
"ros" in HPROGS && push!(PROGS, ("ros_step", CROS, THC, DT0C))

# --------------------------------------------------------------------------- #
# E1/E2 -- determinism of the primal step under repetition.
# --------------------------------------------------------------------------- #
function determinism_cell(nm, c, TH, dt, u::Vector{Float64}, t::Float64, K::Int;
                          reuse::Bool = false)
    UR = reuse ? RX.ConcreteRArray(copy(u)) : nothing
    TR = reuse ? RX.ConcreteRNumber(t) : nothing
    DR = reuse ? RX.ConcreteRNumber(dt) : nothing
    ref = nothing; refe = 0.0
    ndiff = 0; nnonfin = 0; maxreldiff = 0.0; firstidx = Int[]
    eevals = Set{Float64}()
    t0 = time()
    for k in 1:K
        r = reuse ? c(UR, TH, TR, DR) : callstep(c, u, TH, t, dt)
        out = Array(r[1]); ee = Float64(r[2])
        push!(eevals, ee)
        nf = count(!isfinite, out)
        if nf > 0
            nnonfin += 1
            say(@sprintf("  !! %s call %d: %d/%d NON-FINITE (EEst=%g) load=%.2f",
                         nm, k, nf, length(out), ee, loadavg()))
            report_nonfinite(nm, out, u, t, dt, c, TH)
        end
        if ref === nothing
            ref = out; refe = ee
        else
            idx = bitdiff(ref, out)
            if !isempty(idx)
                ndiff += 1
                rel = maximum(i -> abs(out[i] - ref[i]) / max(abs(ref[i]), 1e-300), idx)
                maxreldiff = max(maxreldiff, rel)
                isempty(firstidx) && (firstidx = idx)
            end
        end
    end
    el = time() - t0
    say("  $nm $(reuse ? "reuse" : "fresh") K=$K differing=$ndiff maxrel=$maxreldiff " *
        "distinctEEst=$(length(eevals)) nonfinite_calls=$nnonfin $(el / K) s/call")
    if !isempty(firstidx)
        nc, gs = cellsig(firstidx)
        say(@sprintf("     first differing call: %d/%d entries, groups: %s",
                     length(firstidx), length(ref), join(gs, ", ")))
    end
    return (; ndiff, maxreldiff, nnonfin, nee = length(eevals))
end

function report_nonfinite(nm, out, u, t, dt, c, TH)
    idx = findall(!isfinite, out)
    _, gs = cellsig(idx)
    say(@sprintf("     %d non-finite at t=%.6f dt=%.9g; groups (%d): %s",
                 length(idx), t, dt, length(gs), join(gs, ", ")))
    # the cells, via the variable names
    cells = Set{Tuple{Int,Int,Int}}()
    for (vn, vi) in var_map
        vi in idx || continue
        m = match(r"\[(\d+),(\d+),(\d+)\]$", vn)
        m === nothing && continue
        push!(cells, (parse(Int, m.captures[1]), parse(Int, m.captures[2]),
                      parse(Int, m.captures[3])))
    end
    say("     cells (i,j,k): " * join(sort(collect(cells))[1:min(end, 40)], " "))
    # IMMEDIATE re-issue of the identical call
    for r in 1:4
        rr = callstep(c, u, TH, t, dt)
        o2 = Array(rr[1])
        say(@sprintf("     re-issue %d: %d non-finite, bitdiff vs bad = %d",
                     r, count(!isfinite, o2), length(bitdiff(out, o2))))
    end
    flush(stdout)
end

if hwant("e1") || hwant("e2")
    say("\n---- E1/E2: determinism of the same call repeated ----")
    for (nm, c, TH, dt) in PROGS, (pn, pt) in PT
        hwant("e1") && determinism_cell("$nm/$pn", c, TH, dt, pt.u, pt.t, HK; reuse = false)
        hwant("e2") && determinism_cell("$nm/$pn", c, TH, dt, pt.u, pt.t, HK; reuse = true)
    end
end

# --------------------------------------------------------------------------- #
# E3 -- the soak. Same input, many calls, hunting a non-finite.
# --------------------------------------------------------------------------- #
if hwant("soak")
    say("\n---- E3: soak, identical input repeated ----")
    for (nm, c, TH, dt) in PROGS
        pt = PT["stepped"]
        nbad = 0; ndiff = 0; ref = nothing
        t0 = time()
        for k in 1:HSOAK
            r = callstep(c, pt.u, TH, pt.t, dt)
            out = Array(r[1])
            if count(!isfinite, out) > 0
                nbad += 1
                say(@sprintf("  !! %s soak call %d non-finite, load=%.2f", nm, k, loadavg()))
                report_nonfinite(nm, out, pt.u, pt.t, dt, c, TH)
            end
            ref === nothing ? (ref = out) : (isempty(bitdiff(ref, out)) || (ndiff += 1))
            k % 500 == 0 && (say(@sprintf("     %s soak %d/%d  bad=%d differing=%d  %.1f s",
                                          nm, k, HSOAK, nbad, ndiff, time() - t0)); flush(stdout))
        end
        say(@sprintf("  %s soak: %d calls, %d non-finite, %d bit-differing, %.4f s/call",
                     nm, HSOAK, nbad, ndiff, (time() - t0) / HSOAK))
    end
end

# --------------------------------------------------------------------------- #
# E4 -- the walk. A real trajectory through the compiled programs, which is
#       what the driver does; the fault may need states the base point never
#       reaches.
# --------------------------------------------------------------------------- #
if hwant("walk")
    say("\n---- E4: walking a trajectory ----")
    u = copy(UBASE); t = T0
    nbad = 0; t0 = time()
    for k in 1:HWALK
        c, TH, dt = isodd(k) ? (CSSP, THT, DT0T) : (CROS, THC, DT0C)
        uin = copy(u)
        r = callstep(c, uin, TH, t, dt)
        out = Array(r[1])
        if count(!isfinite, out) > 0
            nbad += 1
            say(@sprintf("  !! walk step %d (%s) non-finite, load=%.2f",
                         k, isodd(k) ? "ssp" : "ros", loadavg()))
            report_nonfinite(isodd(k) ? "ssp" : "ros", out, uin, t, dt, c, TH)
            # do NOT propagate the bad state; take the re-issued one
            out = Array(callstep(c, uin, TH, t, dt)[1])
            count(!isfinite, out) > 0 && break
        end
        u = max.(out, 0.0)          # the production clamp, so the walk stays physical
        t += dt
        k % 500 == 0 && (say(@sprintf("     walk %d/%d bad=%d |u|max=%.4g %.1f s",
                                      k, HWALK, nbad, maximum(abs, u), time() - t0)); flush(stdout))
    end
    say(@sprintf("  walk: %d steps, %d non-finite, %.4f s/step", HWALK, nbad,
                 (time() - t0) / HWALK))
end

# --------------------------------------------------------------------------- #
# E5 -- the VJPs, if asked for.
# --------------------------------------------------------------------------- #
if _WANT_VJP
    say("\n---- E5: the VJPs under repetition ----")
    lam = copy(WOBJ)
    for (nm, c, TH, dt) in (("ssp_vjp", CSSPV, THT, DT0T), ("ros_vjp", CROSV, THC, DT0C))
        c === nothing && continue
        pt = PT["stepped"]
        ref = nothing; ndiff = 0; nbad = 0; t0 = time()
        for k in 1:HSOAK
            r = c(RX.ConcreteRArray(pt.u), TH, RX.ConcreteRArray(lam),
                  RX.ConcreteRNumber(pt.t), RX.ConcreteRNumber(dt))
            out = Array(r[1])
            nf = count(!isfinite, out)
            if nf > 0
                nbad += 1
                say(@sprintf("  !! %s call %d: %d non-finite, load=%.2f", nm, k, nf, loadavg()))
                report_nonfinite(nm, out, pt.u, pt.t, dt, CSSP, TH)
            end
            ref === nothing ? (ref = out) : (isempty(bitdiff(ref, out)) || (ndiff += 1))
            k % 250 == 0 && (say(@sprintf("     %s %d/%d bad=%d differing=%d %.1f s",
                                          nm, k, HSOAK, nbad, ndiff, time() - t0)); flush(stdout))
        end
        say(@sprintf("  %s: %d calls, %d non-finite, %d bit-differing, %.4f s/call",
                     nm, HSOAK, nbad, ndiff, (time() - t0) / HSOAK))
    end
end

say("\nHAMMER DONE  $(now())  loadavg=$(loadavg())")
