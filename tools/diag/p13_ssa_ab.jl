#!/usr/bin/env julia
# ===========================================================================
# p13_ssa_ab.jl -- the ess-oop-ssa READ-REDIRECT ARMS, A/B'd from ONE process:
#                  do the sub-kernel / producer-gather / ghost redirects pay?
# ===========================================================================
# EarthSciAST's `ESS_OOP_SSA` spike redirects a consumer's class-to-class read
# to its PRODUCER'S VALUE and skips the producer's scatter into the flat
# extended buffer `ue` when nothing reads its block off the buffer any more.
# The extension under test adds three arms (each with its own bisect knob, all
# default ON inside `ESS_OOP_SSA=1`):
#
#   ESS_OOP_SSA_SUB       template sub-kernel (`_NK_SUBCALL`) descriptor tables
#   ESS_OOP_SSA_PGATHER   tier 2b, the single-producer VALUE gather
#   ESS_OOP_SSA_GHOST     ghost-masked `_AK_STATE_TBL_BOX` descriptors
#
# Those are BUILD-time flags (the redirect table and the scatter-skip verdict
# are static analysis), so unlike ess-oop-levelbase they cannot ride a
# trace-time `Ref`: this probe includes the DRIVER TWICE, once per arm, into two
# modules, and then compiles + times each program in BOTH modules interleaved.
# One process, so the JIT is warm for both; `P13_ORDER=on,off` swaps which arm
# compiles first, because on this filesystem "which arm went first" is a real
# confounder (see DIFFERENTIABILITY_PLAN.md section 6).
#
# Values are gated bit for bit between the arms: a read-dataflow change may not
# move a number.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 6 6 8; CONUS via sbatch)
#   P13_PROGRAMS             default "rhs,rhs_vjp"; may add ssp_step,ssp_vjp
#   P13_ORDER                "off,on" (default) or "on,off"
#   P13_REPS                 timing reps per arm per pass (default 40)
#   P13_DUMP=0               skip the HLO dump (the copy census needs it)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_BACKEND", "cpu"); get!(ENV, "RESEACT_ADJ_JAC", "sym")
get!(ENV, "RESEACT_ADJ_CLAMP", "1")
ENV["RESEACT_ADJ_UJITTER"] = "0"; ENV["RESEACT_ADJ_STAGES"] = "none"
get!(ENV, "RESEACT_LABEL", "p13ssa")
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Statistics
say(s) = (println(s); flush(stdout))
const PROGS = String.(split(get(ENV, "P13_PROGRAMS", "rhs,rhs_vjp"), ','))
const REPS  = parse(Int, get(ENV, "P13_REPS", "40"))
const ORDER = String.(split(get(ENV, "P13_ORDER", "off,on"), ','))
const DUMP  = get(ENV, "P13_DUMP", "1") == "1"
const GRID  = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT   = joinpath(REPO, "logs", "p13-ssa-$GRID-" * get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)

# The three redirect arms move together by DEFAULT: "off" = the spike's
# coverage, "on" = #283's extension. `ESS_OOP_SSA` itself stays 1 in both, so
# this prices the EXTENSION and not the spike.
#
# P13_ENV_OFF / P13_ENV_ON override the env of either arm entirely
# ("K=V;K=V"), which is how the SCATTER-SKIP GATE is A/B'd: both arms then keep
# the redirects on and differ only in the skip verdict, e.g.
#
#   P13_ENV_OFF='ESS_OOP_SSA_SKIP=0' P13_ENV_ON='ESS_OOP_SSA_SKIP_GATE=0'
#
# is "never skip a scatter" against "#283's ungated skip".
const ARMENV = Dict(
    "off" => Dict("ESS_OOP_SSA" => "1", "ESS_OOP_SSA_GHOST" => "0",
                  "ESS_OOP_SSA_SUB" => "0", "ESS_OOP_SSA_PGATHER" => "0"),
    "on"  => Dict("ESS_OOP_SSA" => "1", "ESS_OOP_SSA_GHOST" => "1",
                  "ESS_OOP_SSA_SUB" => "1", "ESS_OOP_SSA_PGATHER" => "1"))
function armenv(arm)
    e = copy(ARMENV[arm])
    ov = get(ENV, "P13_ENV_" * uppercase(arm), "")
    for kv in split(ov, ';')
        kv = strip(kv); isempty(kv) && continue
        k, v = split(kv, '='; limit = 2)
        e[String(strip(k))] = String(strip(v))
    end
    return e
end

# One driver per arm, in its own module. The build reads the knobs, so the
# `withenv` has to wrap the include.
const MODS = Dict{String,Module}()
for arm in ORDER
    e = armenv(arm)
    say("P13_ARMENV arm=$arm " * join(["$k=$v" for (k, v) in sort(collect(e), by = first)], " "))
    t0 = time()
    m = Core.eval(Main, :(module $(Symbol("_Drv_", arm)) end))
    withenv((collect(e))...) do
        Base.include(m, joinpath(REPO, "tools", "adjoint_gradient.jl"))
    end
    MODS[arm] = m
    say(@sprintf("P13_BUILD arm=%s %.1f s", arm, time() - t0))
end
const RX = getfield(MODS[ORDER[1]], :RX)
const EZ = getfield(MODS[ORDER[1]], :RTI).EZ
say("P13_SSA_AB grid=$GRID NC=$(getfield(MODS[ORDER[1]], :NC)) order=$(join(ORDER, ",")) out=$OUT")

for arm in ORDER
    D = MODS[arm]
    for (pi, f) in enumerate(D.fo)
        say("P13_STATS arm=$arm part=$pi " * repr(D.EA.oop_ssa_stats(f)))
        if isdefined(D.EA, :oop_ssa_producers)
            pr = D.EA.oop_ssa_producers(f)
            say(@sprintf("P13_PROD arm=%s part=%d n=%d skippable=%d skipped=%d skipped_elems=%d",
                         arm, pi, length(pr), count(q -> q.skippable, pr),
                         count(q -> q.skip, pr),
                         sum(q.len for q in pr if q.skip; init = 0)))
            say("P13_SKIPPED arm=$arm part=$pi " *
                repr([(q.pid, q.len, q.nread, q.elread, q.nwhole) for q in pr if q.skip]))
        end
    end
end

copts(dir) = RX.CompileOptions(; sync = true,
    xla_debug_options = (; xla_cpu_prefer_vector_width = 128,
                         (DUMP ? (; xla_dump_to = dir, xla_dump_hlo_as_text = true) : (;))...),
    (isempty(getfield(MODS[ORDER[1]], :EXCLP)) ? (;) :
     (; excluded_passes = collect(String, getfield(MODS[ORDER[1]], :EXCLP))))...)

fresh_u(D) = RX.ConcreteRArray(copy(D.UBASE))
fresh_l(D) = RX.ConcreteRArray(copy(D.WOBJ))
tR(D) = RX.ConcreteRNumber(D.T0)

# The transport RHS and its reverse, in the arm's own module (so `gT` is the
# arm's build). Same shape as p7_rhs_vjp.jl's.
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
# `rhs` returns the bare `du`; `ssp_step`/`ros_step` return `(u_next, err)`; the
# stage VJPs return `(lambda, dtheta)` with the p-gradient under `dtheta.p`,
# while the bare `rhs_vjp` built here differentiates w.r.t. `th.p` directly, so
# its second result IS the p namedtuple. Normalize to (state-like, p-gradient).
function outputs(prog, r, D)
    endswith(prog, "_vjp") || return (Array(r isa Tuple ? r[1] : r), Float64[])
    g = r[2]
    gp = hasproperty(g, :p) ? g.p : g
    return (Array(r[1]), [Float64(getfield(gp, k)) for k in D.PNAMES])
end
function cpu_ns()
    ts = Ref((Clong(0), Clong(0)))
    ccall(:clock_gettime, Cint, (Cint, Ptr{Cvoid}), 2, ts)
    t = ts[]; return Int64(t[1]) * 1_000_000_000 + Int64(t[2])
end
function time_calls(prog, c, D)
    for _ in 1:3; call_prog(prog, c, D); end
    ts = Float64[]; c0 = cpu_ns(); w0 = time_ns()
    for _ in 1:REPS
        t0 = time(); call_prog(prog, c, D); push!(ts, time() - t0)
    end
    return median(ts), minimum(ts), (cpu_ns() - c0) / max(time_ns() - w0, 1)
end

for prog in PROGS
    say("\n---- $prog ----")
    C = Dict{String,Any}()
    for arm in ORDER
        dir = joinpath(OUT, "$prog-$arm"); mkpath(dir); t0 = time()
        C[arm] = compile_prog(prog, MODS[arm], dir)
        say(@sprintf("  @compile %-8s arm=%-3s %.1f s", prog, arm, time() - t0))
    end
    # Interleaved A/B/A/B on the same inputs, fresh buffers per call (donation).
    r1 = Dict(arm => time_calls(prog, C[arm], MODS[arm]) for arm in ORDER)
    r2 = Dict(arm => time_calls(prog, C[arm], MODS[arm]) for arm in ORDER)
    med = Dict(arm => min(r1[arm][1], r2[arm][1]) for arm in ORDER)
    mn  = Dict(arm => min(r1[arm][2], r2[arm][2]) for arm in ORDER)
    say(@sprintf("P13_TIME %-8s NC=%d  off %.3f ms  on %.3f ms  ratio(off/on) %.3f  (mins %.3f / %.3f)  cpu/wall %.2f/%.2f",
                 prog, getfield(MODS[ORDER[1]], :NC), 1e3 * med["off"], 1e3 * med["on"],
                 med["off"] / med["on"], 1e3 * mn["off"], 1e3 * mn["on"],
                 r2["off"][3], r2["on"][3]))
    oa = outputs(prog, call_prog(prog, C["off"], MODS["off"]), MODS["off"])
    ob = outputs(prog, call_prog(prog, C["on"], MODS["on"]), MODS["on"])
    # POINTWISE relative error is uninformative on near-zero components -- a
    # component that is +1e-310 in one arm and -1e-310 in the other reads 2.0,
    # which is what the first CONUS run of this probe reported. So report the
    # SCALE-RELATIVE norm (max|a-b| against max|a|, the quantity a gradient
    # user actually cares about) and how many components exceed 1e-9 of it,
    # alongside the pointwise max.
    rel(x, y) = isempty(x) ? 0.0 : maximum(abs.(x .- y) ./ max.(abs.(x), abs.(y), 1e-300))
    function scal(x, y)
        isempty(x) && return (0.0, 0.0, 0)
        d = maximum(abs.(x .- y))
        s = max(maximum(abs.(x)), 1e-300)
        return (d, d / s, count(>(1e-9 * s), abs.(x .- y)))
    end
    sa = scal(oa[1], ob[1]); sp = scal(oa[2], ob[2])
    say(@sprintf("P13_GATE %-8s bit-for-bit %s   max rel state %.3e   max rel p-grad %.3e   nonfinite %d",
                 prog, oa[1] == ob[1] && oa[2] == ob[2], rel(oa[1], ob[1]), rel(oa[2], ob[2]),
                 count(!isfinite, ob[1]) + count(!isfinite, ob[2])))
    say(@sprintf("P13_SCAL %-8s state absmax %.3e  /scale %.3e  over-1e-9 %d of %d   p-grad absmax %.3e  /scale %.3e  over-1e-9 %d of %d",
                 prog, sa[1], sa[2], sa[3], length(oa[1]), sp[1], sp[2], sp[3], length(oa[2])))
end
say("P13_DONE dumps in $OUT")
