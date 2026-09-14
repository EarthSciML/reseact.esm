#!/usr/bin/env julia
# ===========================================================================
# direct_rhs_census.jl -- CAN THE DIRECT StableHLO EMITTER REPLACE THE TRACED
# `:oop` RIGHT-HAND SIDE THIS MODEL RUNS ON?
# ===========================================================================
# EarthSciAST grew a second compiled lane: instead of TRACING the broadcast
# emitter through Reactant, `EarthSciASTReactantExt.direct_rhs` walks the
# compiled tree-walk IR and builds StableHLO operation by operation. There is
# deliberately no fallback -- a construct the walk cannot lower is a hard
# `EarthSciAST.DirectEmitError` -- so the first question about it is not how
# fast it is but whether it emits this model at all.
#
# This probe answers four questions for each of the THREE programs the adjoint
# driver compiles out of `build_evaluator(...; form = :oop)`:
#
#   transport   split part 1's RHS (the stencil half)
#   chemistry   split part 2's RHS (the mechanism half)
#   jacobian    EarthSciASTDiff's analytic band model `jacE.fJ!`
#               (the `jac = :sym` program Rosenbrock23 needs)
#
#   1 EMISSION   does `direct_rhs_with_buffers` emit, and does `@compile` take
#                the result?  A refusal is recorded with its full text.
#   2 AGREEMENT  direct vs traced at three time points across a day, on the
#                jittered base point the differentiability harnesses use.
#   3 COST       @compile wall time, per-call median, optimized module size.
#   4 GRADIENT   reverse mode through each lane, w.r.t. `u` and w.r.t. `p`.
#
# WHY THE BUILD IS SPELLED OUT HERE rather than taken from
# `tools/adjoint_gradient.jl`. That driver compiles `ssp_step` and `ros_step`
# unconditionally, which is the single most expensive thing it does and none of
# it is wanted here. The build block below is a transcription of the driver's
# (its section 1 and 1b) with the time-loop machinery removed; it uses the same
# split rule, the same overrides, the same const/param arrays and the same
# `source_layout` hand-off into `prepare_jacobian`, so the three programs are
# the programs production runs on.
#
# THE JITTER IS NOT OPTIONAL. The default initial condition is spatially
# uniform, which puts it exactly on the switching surface of essentially every
# PPM limiter -- an agreement check there compares two programs at a kink and
# says nothing about the interior. `RESEACT_ADJ_UJITTER` (default 1e-1, the
# harnesses' seeded stream) displaces it.
#
# ESS_OOP_SSA IS READ AT BUILD TIME, so the two traced variants the cost table
# wants are two BUILDS, not two compiles -- run this script once per setting
# and the direct lane is measured against the traced lane of its own build.
#
# Env:
#   RESEACT_NLON/NLAT/NLEV    grid (default 6/6/8; CONUS is 13/7/72)
#   RESEACT_RES               GEOS-FP grid row (default 4x5)
#   RESEACT_T0                seconds from the epoch (default 5400)
#   RESEACT_ADJ_UJITTER       relative jitter on the base point (default 1e-1)
#   ESS_OOP_SSA               EarthSciAST's SSA class-to-class emitter (default 1)
#   RESEACT_CENSUS_STAGES     subset of emit,agree,cost,grad (default all)
#   RESEACT_CENSUS_NCALL      timed calls after warm-up (default 20)
#   RESEACT_CENSUS_EXCL       excluded StableHLO passes, or "none"
#                             (default: the adjoint driver's own default)
#   RESEACT_CENSUS_TRACED     compile the traced ORACLE too (default 1; set 0
#                             where its own failure would cost the whole run)
#   RESEACT_CENSUS_JSON       write the machine-readable record here
#   RESEACT_RXENV             the Julia environment to activate
# ===========================================================================
import Pkg
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Printf, Statistics, Logging, Random
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system
using EarthSciASTDiff
using Reactant
const EA = EarthSciAST
const RX = Reactant
const EZ = Reactant.Enzyme
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))

get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "ESS_OOP_SSA", "1")
const SSA     = ENV["ESS_OOP_SSA"]
const MODEL   = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const T0      = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const UJIT    = parse(Float64, get(ENV, "RESEACT_ADJ_UJITTER", "1e-1"))
const NCALL   = parse(Int, get(ENV, "RESEACT_CENSUS_NCALL", "20"))
const STAGES  = Set(String.(split(get(ENV, "RESEACT_CENSUS_STAGES", "emit,agree,cost,grad"), ',')))
const JSONOUT = get(ENV, "RESEACT_CENSUS_JSON", "")
const WANT_TRACED = get(ENV, "RESEACT_CENSUS_TRACED", "1") == "1"
want(s) = s in STAGES

_envi(k) = haskey(ENV, "RESEACT_$k") ? parse(Int, ENV["RESEACT_$k"]) : nothing
const RES   = get(ENV, "RESEACT_RES", "4x5")
const SLICE = native_slice(res = RES, lon0 = _envi("LON0"), lat0 = _envi("LAT0"),
                           nlon = _envi("NLON"), nlat = _envi("NLAT"),
                           nlev = _envi("NLEV"))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
# Three probes spread over a day. The forcing is re-sampled at each of them
# (`refresh_forcing`), so the day's diurnal cycle is actually exercised rather
# than three times against one frozen meteorology epoch.
const TPTS  = Float64[T0, T0 + 8 * 3600, T0 + 16 * 3600]
const NDAYS = forcing_days_for(T0, last(TPTS))

# The XLA:CPU intra-op race workaround (`xla_cpu_prefer_vector_width = 128`) and
# the driver's excluded-pass default, applied IDENTICALLY to both lanes -- a
# cost table that compiled the two arms under different options would be
# measuring the options.
const _EXCL_RAW = get(ENV, "RESEACT_CENSUS_EXCL", "dynamic_update_to_concat,sub_const_prop")
const EXCLP = _EXCL_RAW == "none" ? String[] :
    String.(filter(!isempty, strip.(split(_EXCL_RAW, ','))))
const COPTS = RX.CompileOptions(; sync = true,
                                xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
                                (isempty(EXCLP) ? (;) : (; excluded_passes = EXCLP))...)

say("="^78)
say(@sprintf("DIRECT-RHS CENSUS  grid=%dx%dx%d  res=%s  T0=%.0f  ESS_OOP_SSA=%s",
             GRID_MP["NLON"], GRID_MP["NLAT"], NLEV_EFF, RES, T0, SSA))
say("  stages: " * join(sort(collect(STAGES)), ", "))
say("  compile options: sync=true xla_cpu_prefer_vector_width=128 excluded_passes=" *
    (isempty(EXCLP) ? "(none)" : join(EXCLP, ",")))
say("="^78)

# --------------------------------------------------------------------------- #
# 1. Build -- tools/adjoint_gradient.jl section 1, minus the time loop.
# --------------------------------------------------------------------------- #
validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing; splitparts = nothing; merged_const = nothing; ov = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff
    global splitparts, merged_const, ov
    file = EA.load_path(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in splitparts]
    f0 = reseact_forcing(CHEMDIR; ndays = NDAYS, res = SLICE.res)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld; discrete[k] = prov
        end
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part 2 var_map != part 1")
        end
    end
end
foreach(d -> d.materialize!(), dms)
const BUILD_RHS_S = time() - tb
say(@sprintf("BUILD (two :oop halves) %.2f s   nstates=%d  nparams=%d  discrete_providers=%d",
             BUILD_RHS_S, length(u0), length(p), length(discrete)))

# Seed m(0) from the real GEOS-FP surface pressure (species-major state).
let dp0 = hydrostatic_dp(merged_param, ff.const_arrays, T0; slice = SLICE)
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dp0(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
                      parse(Int, mm.captures[3]))
    end
end

# The analytic band model, from the SAME split part with the SAME overrides.
# RESEACT_CENSUS_JAC=0 drops it. It is the one program here whose BUILD is not
# EarthSciAST's: `prepare_jacobian` differentiates the mechanism symbolically,
# and on this branch that is the most expensive thing in the probe by a wide
# margin. Dropping it leaves the two RHS halves measurable on a machine where
# it does not fit.
const WANT_JAC = get(ENV, "RESEACT_CENSUS_JAC", "1") == "1"
tj = time()
const jacE = WANT_JAC ? Logging.with_logger(Logging.NullLogger()) do
    EarthSciASTDiff.prepare_jacobian(splitparts[2]; wrt = :states,
        build_kwargs = (; form = :oop, parameter_overrides = ov,
                          const_arrays = merged_const, param_arrays = merged_param),
        source_layout = (u0 = u0, p = p, var_map = var_map))
end : nothing
const BUILD_JAC_S = time() - tj
WANT_JAC && (jacE.oop ||
    error("the band model came back IN-PLACE; it captures host scratch and cannot be traced"))
if WANT_JAC
    say(@sprintf("BUILD (band model)      %.2f s   structure=%s  band states=%d  entries=%d  scatter=%d",
                 BUILD_JAC_S, jacE.structure, length(jacE.uj),
                 length(jacE.entries), length(jacE.scatter)))
    # A band model with no entries is not a Jacobian, and measuring the emitter
    # on it would report a verdict about an empty program. Say so rather than
    # let a plausible-looking row into the table.
    isempty(jacE.entries) &&
        say("  WARNING: the band model carries NO Jacobian entries; the `jacobian` " *
            "row below is a census of an empty program.")
else
    say("BUILD (band model)      SKIPPED (RESEACT_CENSUS_JAC=0)")
end

const N = length(u0)
const UBASE = let u = copy(u0)
    UJIT > 0 && (u .*= (1 .+ UJIT .* randn(Random.MersenneTwister(31337), N)))
    u
end
say(@sprintf("  base point jittered by %.0e relative (seed 31337)", UJIT))

# The objective weights the adjoint driver scores: the surface-O3 mean.
const WOBJ = let idxs = Int[], rx = r"^SuperFast\.O3\[(\d+),(\d+),(\d+)\]$"
    for (nm, idx) in var_map
        mm = match(rx, nm)
        mm !== nothing && parse(Int, mm.captures[3]) == 1 && push!(idxs, idx)
    end
    isempty(idxs) && error("no surface SuperFast.O3 states found")
    w = zeros(Float64, N); w[idxs] .= 1 / length(idxs); w
end

# --------------------------------------------------------------------------- #
# 2. The three programs, each with its own host buffers and its own input.
# --------------------------------------------------------------------------- #
# `uj` is the band model's OWN state vector; `umap` is the only correct way to
# fill it from the source state (a positional copy would silently evaluate the
# Jacobian of a different point).
band_input(u) = (uj = copy(jacE.uj); for (i, s) in enumerate(jacE.umap); uj[s] = u[i]; end; uj)

struct Program
    name::String
    f::Any                       # the `_OopRHS` build product
    var_map::Any                 # names for the emitter's refusals (nothing = slot numbers)
    input::Vector{Float64}       # a host state vector of the right width
    grad::Bool                   # does the gradient stage cover it?
end
const PROGS = let ps = Program[
        Program("transport", fo[1], var_map, UBASE, true),
        Program("chemistry", fo[2], var_map, UBASE, true),
    ]
    WANT_JAC && push!(ps, Program("jacobian", jacE.fJ!, nothing, band_input(UBASE), false))
    ps
end

const HOSTBUFS = [EA.forcing_buffers(P.f) for P in PROGS]
const DEVBUFS  = [map(RX.ConcreteRArray, hb) for hb in HOSTBUFS]
function refresh_forcing(t)
    for (k, prov) in discrete
        merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), dms)
    for i in eachindex(PROGS)
        EA.sync_forcing!(DEVBUFS[i], EA.forcing_buffers(PROGS[i].f))
    end
    return nothing
end
refresh_forcing(T0)
for (i, P) in enumerate(PROGS)
    say(@sprintf("  %-10s n_states=%6d  forcing buffers=%d",
                 P.name, length(P.input), length(HOSTBUFS[i])))
end

_devp(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
const PR = _devp(p)

const EXT = Base.get_extension(EarthSciAST, :EarthSciASTReactantExt)
EXT === nothing && error("the Reactant extension did not load")

# One record per program, filled in as the stages run and dumped as JSON.
const REC = Dict{String,Any}()
for P in PROGS; REC[P.name] = Dict{String,Any}(); end

# --------------------------------------------------------------------------- #
# 3. EMISSION.
# --------------------------------------------------------------------------- #
# `@compile` needs a literal call expression, so each lane gets its own tiny
# entry point; Julia specializes them on the runtime types, so the compile is
# inferred exactly as a spelled-out script's would be.
compile4(g, u, pp, t, b) = RX.@compile compile_options = COPTS g(u, pp, t, b)
hlo_opt(g, u, pp, t, b)  = RX.@code_hlo compile_options = COPTS optimize = true g(u, pp, t, b)

# One op per line of the form `%name = dialect.op(...)`; counted by `dialect.op`,
# the same census tools/diag/hlo_census.jl reports.
function census(txt::AbstractString)
    h = Dict{String,Int}()
    for m in eachmatch(r"=\s+\"?([a-zA-Z_][\w]*\.[\w.]+)\"?", txt)
        h[m.captures[1]] = get(h, m.captures[1], 0) + 1
    end
    return h
end

function timed_calls(c, args...; n::Int = NCALL)
    c(args...); c(args...)                     # warm-up
    ts = Float64[]
    for _ in 1:n
        t0 = time(); c(args...); push!(ts, time() - t0)
    end
    return (median = median(ts), min = minimum(ts), max = maximum(ts))
end

const DIRECT = Dict{String,Any}()   # name -> (callable, compiled) or nothing
const TRACED = Dict{String,Any}()

for (i, P) in enumerate(PROGS)
    r = REC[P.name]
    r["n_states"] = length(P.input)
    r["n_buffers"] = length(HOSTBUFS[i])
    u_dev = RX.ConcreteRArray(copy(P.input))
    t_dev = RX.ConcreteRNumber(T0)
    b_dev = DEVBUFS[i]

    say("\n" * "-"^78)
    say("  $(P.name): DIRECT emission")
    dd = nothing; cd = nothing; t_emit = NaN
    try
        dd = EXT.direct_rhs_with_buffers(P.f; var_map = P.var_map)
        t0 = time()
        cd = compile4(dd, u_dev, PR, t_dev, b_dev)
        t_emit = time() - t0
        say(@sprintf("    EMITTED and COMPILED in %.1f s   stats=%s", t_emit, dd.d.stats))
        r["direct_emitted"] = true
        r["direct_compile_s"] = t_emit
        r["direct_stats"] = Dict(String(k) => v for (k, v) in dd.d.stats)
    catch e
        r["direct_emitted"] = false
        r["direct_error"] = sprint(showerror, e)
        say("    REFUSED:")
        for ln in split(sprint(showerror, e), '\n'); say("      " * ln); end
        dd = nothing; cd = nothing
    end
    DIRECT[P.name] = (dd, cd)

    # The census patch (EarthSciAST, uncommitted) records every refusal instead
    # of stopping at the first. When it is not applied this block does nothing,
    # which is what makes the committed probe run against a clean worktree.
    if isdefined(EXT, :_DE_CENSUS_LOG)
        EXT._DE_CENSUS[] = true
        empty!(EXT._DE_CENSUS_LOG)
        try
            d2 = EXT.direct_rhs_with_buffers(P.f; var_map = P.var_map)
            RX.@code_hlo compile_options = COPTS optimize = false d2(u_dev, PR, t_dev, b_dev)
        catch e
            say("    census walk itself failed: " * first(split(sprint(showerror, e), '\n')))
        finally
            EXT._DE_CENSUS[] = false
        end
        log = copy(EXT._DE_CENSUS_LOG)
        r["gaps"] = [Dict("construct" => g[1], "rule" => g[2], "detail" => g[3]) for g in log]
        say(@sprintf("    census walk: %d refusal(s) total", length(log)))
        kinds = Dict{String,Int}()
        for g in log; kinds[g[1]] = get(kinds, g[1], 0) + 1; end
        for (k, v) in sort(collect(kinds); by = last, rev = true)
            say(@sprintf("      %6d x  %s", v, k))
        end
        # THE STUBBED PROGRAM, and what it is and is not. With the refused units
        # replaced by zero constants the walk runs to the end, so the rest of
        # the model DOES emit and compile. Its cost is a LOWER BOUND on what a
        # gap-free direct lane would cost -- the arithmetic of the skipped units
        # is missing, and so is everything downstream that the zeros
        # const-folded away -- and its VALUES are wrong by construction, so it
        # is never compared against anything. It is here because "how big is the
        # part that already lowers" is otherwise unanswerable.
        if cd === nothing && !isempty(log)
            try
                EXT._DE_CENSUS[] = true
                d3 = EXT.direct_rhs_with_buffers(P.f; var_map = P.var_map)
                t0 = time()
                c3 = compile4(d3, u_dev, PR, t_dev, b_dev)
                ts = time() - t0
                tm = timed_calls(c3, u_dev, PR, t_dev, b_dev)
                txt = sprint(show, hlo_opt(d3, u_dev, PR, t_dev, b_dev))
                nops = sum(values(census(txt)); init = 0)
                r["stub_compile_s"] = ts
                r["stub_call_s"] = tm.median
                r["stub_hlo_lines"] = count(==('\n'), txt)
                r["stub_hlo_ops"] = nops
                r["stub_skipped_units"] = length(log)
                say(@sprintf("    STUBBED direct (%d unit(s) zeroed): compile %.1f s  call %.4f s  module %d lines / %d ops",
                             length(log), ts, tm.median, r["stub_hlo_lines"], nops))
            catch e
                say("    stubbed direct compile failed: " * first(split(sprint(showerror, e), '\n')))
            finally
                EXT._DE_CENSUS[] = false
            end
        end
    end

    say("  $(P.name): TRACED emission")
    g = EA.rhs_with_buffers(P.f)
    ct = nothing
    # `RESEACT_CENSUS_TRACED=0` skips the ORACLE's own compile. It exists because
    # on Reactant 0.2.285 the traced lane does not compile at all (section 4 of
    # the report) and the way it fails is expensive: the chemistry half climbs
    # past 32 GB resident before it gives up, which in this cgroup takes the
    # process -- and the direct lane's numbers -- with it. Skipping it is how the
    # DIRECT rows get measured on a machine shared with another Julia.
    if !WANT_TRACED
        say("    SKIPPED (RESEACT_CENSUS_TRACED=0)")
        r["traced_compiled"] = false
        r["traced_error"] = "skipped: RESEACT_CENSUS_TRACED=0"
        TRACED[P.name] = (g, nothing)
        continue
    end
    try
        t0 = time()
        ct = compile4(g, u_dev, PR, t_dev, b_dev)
        tt = time() - t0
        say(@sprintf("    compiled in %.1f s", tt))
        r["traced_compile_s"] = tt
        r["traced_compiled"] = true
    catch e
        # The traced lane is the ORACLE, not the subject, so its failure must
        # not end the run: the direct lane's own numbers are still wanted, and
        # "the oracle did not compile" is itself a result. Only the first line
        # of the message is kept -- a Reactant compile failure prints the whole
        # module, which is megabytes.
        r["traced_compiled"] = false
        msg = first(split(sprint(showerror, e), '\n'))
        r["traced_error"] = String(msg)
        say("    TRACED COMPILE FAILED: " * String(msg))
    end
    TRACED[P.name] = (g, ct)
end

# --------------------------------------------------------------------------- #
# 4. AGREEMENT -- direct vs traced at three points across a day.
# --------------------------------------------------------------------------- #
# `maxrel` is measured only where the traced reference is above the absolute
# floor; entries below it are the `maxabs` column's business. Reporting a
# relative difference against a reference of 1e-300 is how a passing program
# gets called a failing one.
const ATOL_FLOOR = 1e-14
function compare(a::Vector{Float64}, b::Vector{Float64})
    maxabs = 0.0; maxrel = 0.0; nrel = 0; nnf = 0
    for i in eachindex(a)
        (isfinite(a[i]) && isfinite(b[i])) || (nnf += 1; continue)
        d = abs(a[i] - b[i]); d > maxabs && (maxabs = d)
        if abs(b[i]) > ATOL_FLOOR
            nrel += 1
            rr = d / abs(b[i]); rr > maxrel && (maxrel = rr)
        end
    end
    return (maxabs = maxabs, maxrel = maxrel, nrel = nrel, nonfinite = nnf)
end

if want("agree")
    say("\n" * "="^78)
    say("AGREEMENT  (direct vs traced; relative measured where |traced| > $ATOL_FLOOR)")
    say("="^78)
    for (i, P) in enumerate(PROGS)
        dd, cd = DIRECT[P.name]
        _, ct = TRACED[P.name]
        r = REC[P.name]
        cd === nothing && (say(@sprintf("  %-10s  -- direct refused, nothing to compare", P.name)); continue)
        ct === nothing && (say(@sprintf("  %-10s  -- traced did not compile, nothing to compare against", P.name)); continue)
        rows = Any[]
        for t in TPTS
            refresh_forcing(t)
            u_dev = RX.ConcreteRArray(copy(P.input))
            t_dev = RX.ConcreteRNumber(t)
            a = Array(cd(u_dev, PR, t_dev, DEVBUFS[i]))
            b = Array(ct(u_dev, PR, t_dev, DEVBUFS[i]))
            c = compare(a, b)
            say(@sprintf("  %-10s t=%8.0f   maxabs %.3e   maxrel %.3e   (%d of %d above floor)  nonfinite %d",
                         P.name, t, c.maxabs, c.maxrel, c.nrel, length(a), c.nonfinite))
            push!(rows, Dict("t" => t, "maxabs" => c.maxabs, "maxrel" => c.maxrel,
                             "n_above_floor" => c.nrel, "n" => length(a),
                             "nonfinite" => c.nonfinite))
        end
        r["agreement"] = rows
    end
    refresh_forcing(T0)
end

# --------------------------------------------------------------------------- #
# 5. COST.
# --------------------------------------------------------------------------- #
if want("cost")
    say("\n" * "="^78)
    say("COST  (per-call median of $NCALL calls after warm-up; module = optimize=true)")
    say("="^78)
    for (i, P) in enumerate(PROGS)
        dd, cd = DIRECT[P.name]
        g, ct = TRACED[P.name]
        r = REC[P.name]
        u_dev = RX.ConcreteRArray(copy(P.input))
        t_dev = RX.ConcreteRNumber(T0)
        if cd !== nothing
            tm = timed_calls(cd, u_dev, PR, t_dev, DEVBUFS[i])
            txt = try
                sprint(show, hlo_opt(dd, u_dev, PR, t_dev, DEVBUFS[i]))
            catch e
                say("    direct module dump failed: " * first(split(sprint(showerror, e), '\n')))
                ""
            end
            h = census(txt)
            r["direct_call_s"] = tm.median
            r["direct_hlo_lines"] = count(==('\n'), txt)
            r["direct_hlo_ops"] = sum(values(h); init = 0)
            say(@sprintf("  %-10s DIRECT  call %.4f s (min %.4f)   module %7d lines / %7d ops",
                         P.name, tm.median, tm.min, r["direct_hlo_lines"], r["direct_hlo_ops"]))
        end
        if ct !== nothing
            tm = timed_calls(ct, u_dev, PR, t_dev, DEVBUFS[i])
            txt = sprint(show, hlo_opt(g, u_dev, PR, t_dev, DEVBUFS[i]))
            h = census(txt)
            r["traced_call_s"] = tm.median
            r["traced_hlo_lines"] = count(==('\n'), txt)
            r["traced_hlo_ops"] = sum(values(h); init = 0)
            say(@sprintf("  %-10s TRACED  call %.4f s (min %.4f)   module %7d lines / %7d ops",
                         P.name, tm.median, tm.min, r["traced_hlo_lines"], r["traced_hlo_ops"]))
        else
            say(@sprintf("  %-10s TRACED  did not compile; no cost row", P.name))
        end
    end
end

# --------------------------------------------------------------------------- #
# 6. GRADIENT -- reverse mode through each lane.
# --------------------------------------------------------------------------- #
# The payload is an EXPLICIT argument of the differentiated callee, never a
# closure: a closure over `p` makes dJ/dp invisible rather than wrong-loudly,
# which is the trap the adjoint driver's `gT`/`gC` exist to avoid.
_wobj(f, u, pp, t, b, w) = sum(w .* f(u, pp, t, b))
grad_u(f, u, pp, t, b, w) = EZ.gradient(EZ.Reverse, _wobj, EZ.Const(f), u,
                                        EZ.Const(pp), EZ.Const(t), EZ.Const(b), EZ.Const(w))
grad_p(f, u, pp, t, b, w) = EZ.gradient(EZ.Reverse, _wobj, EZ.Const(f), EZ.Const(u),
                                        pp, EZ.Const(t), EZ.Const(b), EZ.Const(w))

function grad_lane(tag, f, c_u, c_p, u_dev, t_dev, b_dev, w_dev, r)
    gu = gp = nothing
    try
        t0 = time(); cu = RX.@compile compile_options = COPTS grad_u(f, u_dev, PR, t_dev, b_dev, w_dev)
        gu = Array(cu(f, u_dev, PR, t_dev, b_dev, w_dev)[2])
        say(@sprintf("    %-6s d/du  compiled %.1f s   ||g||=%.6e  nonfinite=%d",
                     tag, time() - t0, norm(gu), count(!isfinite, gu)))
        r[tag * "_gradu_ok"] = true
    catch e
        r[tag * "_gradu_ok"] = false
        r[tag * "_gradu_error"] = sprint(showerror, e)
        say("    $tag d/du FAILED: " * first(split(sprint(showerror, e), '\n')))
    end
    try
        t0 = time(); cp = RX.@compile compile_options = COPTS grad_p(f, u_dev, PR, t_dev, b_dev, w_dev)
        res = cp(f, u_dev, PR, t_dev, b_dev, w_dev)[3]
        gp = Float64[Float64(getfield(res, k)) for k in keys(res)]
        say(@sprintf("    %-6s d/dp  compiled %.1f s   ||g||=%.6e  nonfinite=%d  (%d params)",
                     tag, time() - t0, norm(gp), count(!isfinite, gp), length(gp)))
        r[tag * "_gradp_ok"] = true
    catch e
        r[tag * "_gradp_ok"] = false
        r[tag * "_gradp_error"] = sprint(showerror, e)
        say("    $tag d/dp FAILED: " * first(split(sprint(showerror, e), '\n')))
    end
    return gu, gp
end

if want("grad")
    say("\n" * "="^78)
    say("GRADIENT  (reverse mode, objective = the surface-O3 mean of du)")
    say("="^78)
    w_dev = RX.ConcreteRArray(copy(WOBJ))
    for (i, P) in enumerate(PROGS)
        P.grad || continue
        r = REC[P.name]
        dd, cd = DIRECT[P.name]
        g, _ = TRACED[P.name]
        u_dev = RX.ConcreteRArray(copy(P.input))
        t_dev = RX.ConcreteRNumber(T0)
        say("  $(P.name):")
        gud = gpd = nothing
        if dd !== nothing
            gud, gpd = grad_lane("direct", dd, nothing, nothing, u_dev, t_dev, DEVBUFS[i], w_dev, r)
        else
            say("    direct refused at emission; no gradient to take")
        end
        _, ctp = TRACED[P.name]
        gut = gpt = nothing
        if ctp === nothing
            say("    traced primal did not compile on this Reactant; its gradient is not attempted")
        else
            gut, gpt = grad_lane("traced", g, nothing, nothing, u_dev, t_dev, DEVBUFS[i], w_dev, r)
        end
        if gud !== nothing && gut !== nothing
            c = compare(gud, gut)
            say(@sprintf("    d/du agreement  maxabs %.3e  maxrel %.3e  (%d above floor)  nonfinite %d",
                         c.maxabs, c.maxrel, c.nrel, c.nonfinite))
            r["gradu_agreement"] = Dict("maxabs" => c.maxabs, "maxrel" => c.maxrel,
                                        "n_above_floor" => c.nrel, "nonfinite" => c.nonfinite)
        end
        if gpd !== nothing && gpt !== nothing
            c = compare(gpd, gpt)
            say(@sprintf("    d/dp agreement  maxabs %.3e  maxrel %.3e  (%d above floor)  nonfinite %d",
                         c.maxabs, c.maxrel, c.nrel, c.nonfinite))
            r["gradp_agreement"] = Dict("maxabs" => c.maxabs, "maxrel" => c.maxrel,
                                        "n_above_floor" => c.nrel, "nonfinite" => c.nonfinite)
        end
    end
end

if !isempty(JSONOUT)
    payload = Dict("grid" => [GRID_MP["NLON"], GRID_MP["NLAT"], NLEV_EFF],
                   "res" => RES, "t0" => T0, "ess_oop_ssa" => SSA,
                   "excluded_passes" => EXCLP,
                   "build_rhs_s" => BUILD_RHS_S, "build_jac_s" => BUILD_JAC_S,
                   "programs" => REC)
    open(JSONOUT, "w") do io; JSON3.pretty(io, payload); end
    say("\nwrote $JSONOUT")
end
say("\nCENSUS_DONE")
