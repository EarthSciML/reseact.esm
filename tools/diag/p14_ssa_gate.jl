#!/usr/bin/env julia
# ===========================================================================
# p14_ssa_gate.jl -- the ess-oop-ssa SCATTER-SKIP GATE, N arms from ONE process
# ===========================================================================
# p13_ssa_ab.jl prices #283's read-redirect arms two at a time. This probe is
# the same machinery with an ARBITRARY NUMBER of arms, because the gate is a
# THRESHOLD and finding it needs more than an A/B: each arm is one driver build
# (the knobs are read at BUILD time) in its own module, all compiled and then
# timed round-robin on the same inputs with fresh device buffers per call, and
# every arm's values gated against the FIRST arm.
#
#   P14_ARMS         arm names, e.g. "noskip,ungated,gate512"  (order = compile
#                    order; reverse it to price the order confounder)
#   P14_ENV_<ARM>    that arm's build env, "K=V;K=V" -- ESS_OOP_SSA=1 is
#                    implied, everything else is explicit so an arm's setting
#                    is legible in the log
#   P14_PROGRAMS     default "ssp_step"; any of rhs,rhs_vjp,ssp_step,ssp_vjp,
#                    ros_step,ros_vjp
#   P14_REPS         timing reps per arm per pass (default 40)
#   P14_DUMP=1       dump optimized HLO per arm (for the copy census)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p14gate")
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Statistics
say(s) = (println(s); flush(stdout))
const PROGS = String.(split(get(ENV, "P14_PROGRAMS", "ssp_step"), ','))
const REPS  = parse(Int, get(ENV, "P14_REPS", "40"))
const ARMS  = String.(split(get(ENV, "P14_ARMS", "noskip,ungated"), ','))
const DUMP  = get(ENV, "P14_DUMP", "0") == "1"
const GRID  = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT   = joinpath(REPO, "logs", "p14-gate-$GRID-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)

function armenv(arm)
    e = Dict("ESS_OOP_SSA" => "1")
    for kv in split(get(ENV, "P14_ENV_" * uppercase(arm), ""), ';')
        kv = strip(kv); isempty(kv) && continue
        k, v = split(kv, '='; limit = 2)
        e[String(strip(k))] = String(strip(v))
    end
    return e
end

const MODS = Dict{String,Module}()
for arm in ARMS
    e = armenv(arm)
    say("P14_ARMENV arm=$arm " * join(["$k=$v" for (k, v) in sort(collect(e), by = first)], " "))
    t0 = time()
    m = Core.eval(Main, :(module $(Symbol("_Drv_", arm)) end))
    withenv((collect(e))...) do
        Base.include(m, joinpath(REPO, "tools", "adjoint_gradient.jl"))
    end
    MODS[arm] = m
    say(@sprintf("P14_BUILD arm=%s %.1f s", arm, time() - t0))
end
const RX = getfield(MODS[ARMS[1]], :RX)
const EZ = getfield(MODS[ARMS[1]], :RTI).EZ
const NC = getfield(MODS[ARMS[1]], :NC)
say("P14_GATE grid=$GRID NC=$NC arms=$(join(ARMS, ",")) progs=$(join(PROGS, ",")) out=$OUT")

for arm in ARMS
    D = MODS[arm]
    for (pi, f) in enumerate(D.fo)
        st = D.EA.oop_ssa_stats(f)
        say(@sprintf("P14_STATS arm=%-9s part=%d n_prod=%d skippable=%d skipped=%d skipped_elems=%d n_fast=%d/%d sub_fast=%d/%d",
                     arm, pi, st.n_producers,
                     get(st, :n_skippable_scatters, -1), st.n_skipped_scatters,
                     st.elems_skipped, st.n_fast, st.n_edges,
                     st.n_sub_fast, st.n_sub_edges))
        if isdefined(D.EA, :oop_ssa_producers)
            pr = D.EA.oop_ssa_producers(f)
            say("P14_SKIPPED arm=$arm part=$pi " * repr([q.pid for q in pr if q.skip]))
        end
    end
end

copts(dir) = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128,
                         (DUMP ? (; xla_dump_to = dir, xla_dump_hlo_as_text = true) : (;))...),
    (isempty(getfield(MODS[ARMS[1]], :EXCLP)) ? (;) :
     (; excluded_passes = collect(String, getfield(MODS[ARMS[1]], :EXCLP))))...)

fresh_u(D) = RX.ConcreteRArray(copy(D.UBASE))
fresh_l(D) = RX.ConcreteRArray(copy(D.WOBJ))
tR(D) = RX.ConcreteRNumber(D.T0)

function make_progs(D)
    rhs(u, th, t) = D.gT(u, th, t)
    ldot(u, p, rest, lam, t) = sum(lam .* D.gT(u, merge((p = p,), rest), t))
    rhs_vjp(u, th, lam, t) = begin
        r = EZ.gradient(EZ.Reverse, ldot, u, th.p,
                        EZ.Const(D.RTI._theta_rest(th)), EZ.Const(lam), EZ.Const(t))
        (r[1], r[2])
    end
    return rhs, rhs_vjp
end
function compile_prog(prog, D, dir)
    rhs, rhs_vjp = make_progs(D)
    co = copts(dir)
    prog == "rhs"      && return RX.@compile compile_options = co rhs(D.U_R, D.THT, D.T_R)
    prog == "rhs_vjp"  && return RX.@compile compile_options = co rhs_vjp(D.U_R, D.THT, D.LAM_R, D.T_R)
    prog == "ssp_step" && return RX.@compile compile_options = co D.ssp_step(D.U_R, D.THT, D.T_R, D.DTT_R)
    prog == "ssp_vjp"  && return RX.@compile compile_options = co D.ssp_vjp(D.U_R, D.THT, D.LAM_R, D.T_R, D.DTT_R)
    prog == "ros_step" && return RX.@compile compile_options = co D.ros_step(D.U_R, D.THC, D.T_R, D.DTC_R)
    prog == "ros_vjp"  && return RX.@compile compile_options = co D.ros_vjp(D.U_R, D.THC, D.LAM_R, D.T_R, D.DTC_R)
    error("unknown program $prog")
end
function call_prog(prog, c, D)
    prog == "rhs"      && return c(fresh_u(D), D.THT, tR(D))
    prog == "rhs_vjp"  && return c(fresh_u(D), D.THT, fresh_l(D), tR(D))
    prog == "ssp_step" && return c(fresh_u(D), D.THT, tR(D), RX.ConcreteRNumber(D.DT0T))
    prog == "ros_step" && return c(fresh_u(D), D.THC, tR(D), RX.ConcreteRNumber(D.DT0C))
    prog == "ros_vjp"  && return c(fresh_u(D), D.THC, fresh_l(D), tR(D), RX.ConcreteRNumber(D.DT0C))
    return c(fresh_u(D), D.THT, fresh_l(D), tR(D), RX.ConcreteRNumber(D.DT0T))
end
function outputs(prog, r, D)
    endswith(prog, "_vjp") || return (Array(r isa Tuple ? r[1] : r), Float64[])
    g = r[2]
    gp = hasproperty(g, :p) ? g.p : g
    return (Array(r[1]), [Float64(getfield(gp, k)) for k in D.PNAMES])
end
function time_calls(prog, c, D)
    for _ in 1:3; call_prog(prog, c, D); end
    ts = Float64[]
    for _ in 1:REPS
        t0 = time(); call_prog(prog, c, D); push!(ts, time() - t0)
    end
    return median(ts), minimum(ts)
end

for prog in PROGS
    say("\n---- $prog ----")
    C = Dict{String,Any}()
    for arm in ARMS
        dir = joinpath(OUT, "$prog-$arm"); mkpath(dir); t0 = time()
        C[arm] = compile_prog(prog, MODS[arm], dir)
        say(@sprintf("  @compile %-8s arm=%-9s %.1f s", prog, arm, time() - t0))
    end
    med = Dict(arm => Inf for arm in ARMS); mn = Dict(arm => Inf for arm in ARMS)
    for _ in 1:2, arm in ARMS
        a, b = time_calls(prog, C[arm], MODS[arm])
        med[arm] = min(med[arm], a); mn[arm] = min(mn[arm], b)
    end
    base = ARMS[1]
    for arm in ARMS
        say(@sprintf("P14_TIME %-8s NC=%d arm=%-9s %8.3f ms  min %8.3f ms  %s/arm %.3f",
                     prog, NC, arm, 1e3 * med[arm], 1e3 * mn[arm], base,
                     med[base] / med[arm]))
    end
    ref = outputs(prog, call_prog(prog, C[base], MODS[base]), MODS[base])
    for arm in ARMS
        arm == base && continue
        o = outputs(prog, call_prog(prog, C[arm], MODS[arm]), MODS[arm])
        function scal(x, y)
            isempty(x) && return (0.0, 0.0, 0)
            d = maximum(abs.(x .- y)); s = max(maximum(abs.(x)), 1e-300)
            return (d, d / s, count(>(1e-9 * s), abs.(x .- y)))
        end
        sa = scal(ref[1], o[1]); sp = scal(ref[2], o[2])
        say(@sprintf("P14_GATEQ %-8s arm=%-9s bitwise %s  state absmax %.3e /scale %.3e over-1e-9 %d of %d  p-grad absmax %.3e /scale %.3e over %d",
                     prog, arm, ref[1] == o[1] && ref[2] == o[2],
                     sa[1], sa[2], sa[3], length(ref[1]), sp[1], sp[2], sp[3]))
    end
end
say("P14_DONE dumps in $OUT")
