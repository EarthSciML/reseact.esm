#!/usr/bin/env julia
# ===========================================================================
# hlo_dump.jl -- DUMP the raw StableHLO to disk so attribution is offline.
# ===========================================================================
# Every number in this workstream costs a 15-25 min build before it appears.
# hlo_census.jl prints aggregate counts and throws the module away; this dumps
# the module TEXT, so one job pays the build once and every subsequent question
# ("how many of these ops are exact duplicates?", "of which operand?", "does
# interning change the op mix?") is answered by a text pass at zero cost.
#
# Dumps, per program (chem RHS / symbolic band Jacobian / one ROS23 step):
#   <out>/<prog>.unopt.mlir        `@code_hlo optimize=false`
#   <out>/<prog>.opt.mlir          `@code_hlo` (post XLA pipeline)
# and re-dumps every program under whatever alternate emitter settings are asked
# for, so what a sharing switch is worth is measured rather than assumed.
#
# Also records, per trace: wall time, and the engagement counters of the two
# trace-time sharing features -- `EarthSciAST.oop_intern_stats` (the
# `(operand, window)` read memo) and `oop_gvn_stats` (emission value numbering).
#
#   RESEACT_NLON/NLAT/NLEV  grid (default 6/6/8)
#   HLO_OUT                 output dir (default logs/hlo-<grid>-<jobid>)
#   HLO_PROGS               comma list of rhs,jac,step (default all)
#   HLO_VARIANTS            `;`-separated variants, each a `,`-separated list of
#                           NAME=VALUE env settings, e.g.
#                           "ESS_OOP_GVN=0;ESS_OOP_GVN=0,ESS_OOP_INTERN=0"
#   HLO_VARIANT_OPT         0 to skip the (slow) optimized dump for variants
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "hlodump"

include(joinpath(@__DIR__, "_env.jl"))
using Printf
say(s) = (println(s); flush(stdout))

const GRID = string(ENV["RESEACT_NLON"], "x", ENV["RESEACT_NLAT"], "x", ENV["RESEACT_NLEV"])
const OUT = get(ENV, "HLO_OUT",
                joinpath(REPO, "logs", "hlo-$GRID-" * get(ENV, "SLURM_JOB_ID", "local")))
mkpath(OUT)
say("HLO_DUMP grid=$GRID  out=$OUT")

const tb = time()
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
RX = D.RX
EA = D.EA
say(@sprintf("driver loaded in %.1f s  (build + prepare_jacobian + the two @compiles)", time() - tb))

const PROGS = Set(String.(split(get(ENV, "HLO_PROGS", "rhs,jac,step"), ',')))

UR = RX.ConcreteRArray(copy(D.UBASE))
TR = RX.ConcreteRNumber(D.T0)
DR = RX.ConcreteRNumber(D.DT0C)

# One op per `%x = dialect.op` line.
function census(txt::AbstractString)
    h = Dict{String,Int}()
    for m in eachmatch(r"=\s+\"?([a-zA-Z_][\w]*\.[\w.]+)\"?", txt)
        h[m.captures[1]] = get(h, m.captures[1], 0) + 1
    end
    return h
end
total(h) = sum(values(h); init = 0)

# `oop_gvn_stats` postdates some checkouts; report it when it is there.
const HAS_GVN = isdefined(EA, :oop_gvn_stats)
_gvn_reset!() = HAS_GVN && EA.oop_gvn_stats_reset!()
_gvn_stats() = HAS_GVN ? EA.oop_gvn_stats() : (hits = -1, misses = -1)

function dump_one(tag::AbstractString, thunk)
    EA.oop_intern_stats_reset!(); _gvn_reset!()
    t0 = time()
    txt = try
        sprint(show, thunk())
    catch e
        say("    $tag FAILED: " * first(split(sprint(showerror, e), '\n')))
        return 0
    end
    el = time() - t0
    st = EA.oop_intern_stats(); gv = _gvn_stats()
    path = joinpath(OUT, tag * ".mlir")
    open(path, "w") do io; write(io, txt); end
    n = total(census(txt))
    @printf("  %-28s %8d ops  %9.1f s trace  intern %d/%d  gvn %d/%d  -> %s\n",
            tag, n, el, st.hits, st.hits + st.misses,
            gv.hits, gv.hits + gv.misses, basename(path))
    flush(stdout)
    return n
end

# One pass over the requested programs, under whatever emitter settings are in
# force. `tag` prefixes the file names so variants do not overwrite each other.
function dump_all(tag::AbstractString; want_opt::Bool = true)
    pre = isempty(tag) ? "" : tag * "."
    if "rhs" in PROGS
        dump_one(pre * "rhs.unopt", () -> RX.@code_hlo(optimize = false, D.gC(UR, D.THC, TR)))
        want_opt && dump_one(pre * "rhs.opt", () -> RX.@code_hlo(D.gC(UR, D.THC, TR)))
    end
    if "jac" in PROGS && D.SYMJAC
        dump_one(pre * "jac.unopt", () -> RX.@code_hlo(optimize = false, D.gJ(UR, D.THC, TR)))
        want_opt && dump_one(pre * "jac.opt", () -> RX.@code_hlo(D.gJ(UR, D.THC, TR)))
    end
    if "step" in PROGS
        dump_one(pre * "step.unopt",
                 () -> RX.@code_hlo(optimize = false, D.ros_step(UR, D.THC, TR, DR)))
        want_opt && dump_one(pre * "step.opt",
                             () -> RX.@code_hlo(D.ros_step(UR, D.THC, TR, DR)))
    end
    return nothing
end

say("\n---- dumping: baseline (emitter defaults) ----")
dump_all("")

# Alternate emitter settings, to attribute what each sharing switch is worth.
# `HLO_VARIANTS` is a `;`-separated list of variants, each a `,`-separated list
# of `NAME=VALUE` env settings applied together.
for spec in split(get(ENV, "HLO_VARIANTS", ""), ';')
    spec = strip(spec)
    isempty(spec) && continue
    kvs = [split(strip(x), '=' ; limit = 2) for x in split(spec, ',')]
    old = Dict{String,Any}(String(k) => get(ENV, String(k), nothing) for (k, _) in kvs)
    for (k, v) in kvs; ENV[String(k)] = String(v); end
    tag = replace(spec, "=" => "", "," => "_", " " => "")
    say("\n---- dumping: $spec ----")
    dump_all(tag; want_opt = get(ENV, "HLO_VARIANT_OPT", "1") == "1")
    for (k, v) in old
        v === nothing ? delete!(ENV, k) : (ENV[k] = v)
    end
end

# ---- compile wall time, the metric this whole workstream is about ----------
#
# Op counts are the mechanism; `@compile` seconds are the cost. Each mode gets a
# FRESH closure: Reactant keys its compilation cache on the callee, so reusing
# one would hand the second mode the first one's already-compiled program and
# report a few milliseconds for it.
if get(ENV, "HLO_COMPILE", "1") == "1"
    say("\n---- @compile wall time ----")
    for spec in split(get(ENV, "HLO_COMPILE_MODES", "ESS_OOP_GVN=1;ESS_OOP_GVN=0"), ';')
        spec = strip(spec)
        isempty(spec) && continue
        kvs = [split(strip(x), '=' ; limit = 2) for x in split(spec, ',')]
        old = Dict{String,Any}(String(k) => get(ENV, String(k), nothing) for (k, _) in kvs)
        for (k, v) in kvs; ENV[String(k)] = String(v); end
        # Fresh closures per mode -- Reactant caches compilations on the callee.
        crhs  = (uu, th, tt) -> D.gC(uu, th, tt)
        cros  = (uu, th, tt, dd) -> D.ros_step(uu, th, tt, dd)
        cssp  = (uu, th, tt, dd) -> D.ssp_step(uu, th, tt, dd)
        crvjp = (uu, th, ll, tt, dd) -> D.ros_vjp(uu, th, ll, tt, dd)
        csvjp = (uu, th, ll, tt, dd) -> D.ssp_vjp(uu, th, ll, tt, dd)
        function tc(nm, thunk)
            t0 = time(); thunk(); el = time() - t0
            @printf("  %-32s %-10s %9.1f s\n", spec, nm, el); flush(stdout)
            return el
        end
        tc("rhs", () -> RX.@compile compile_options = D.COPTS crhs(UR, D.THC, TR))
        tc("ros_step", () -> RX.@compile compile_options = D.COPTS cros(UR, D.THC, TR, DR))
        if get(ENV, "HLO_COMPILE_ALL", "1") == "1"
            tc("ssp_step", () -> RX.@compile compile_options = D.COPTS cssp(UR, D.THT, TR, D.DTT_R))
            tc("ros_vjp", () -> RX.@compile compile_options = D.COPTS crvjp(UR, D.THC, D.LAM_R, TR, DR))
            tc("ssp_vjp", () -> RX.@compile compile_options = D.COPTS csvjp(UR, D.THT, D.LAM_R, TR, D.DTT_R))
        end
        for (k, v) in old
            v === nothing ? delete!(ENV, k) : (ENV[k] = v)
        end
    end
end

say("\nHLO_DUMP_DONE  $OUT")
