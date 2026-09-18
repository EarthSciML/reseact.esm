#!/usr/bin/env julia
# ===========================================================================
# direct_rhs_compile_cost.jl -- WHERE DOES THE DIRECT LANE'S COMPILE TIME GO?
# ===========================================================================
# ReSEACT's production entry points compile through the direct StableHLO
# emitter (tools/rx_rhs.jl), and at 6x6x8 that compile is expensive. Of the
# adjoint driver's four programs the chemistry half is fine; it is the
# TRANSPORT half, and reverse mode over the transport half most of all. The
# numbers are in COMPILE_COST.md.
#
# This probe takes that apart on ONE program at a time, without paying for a
# whole adjoint run, by separating the three costs `@compile` rolls into one:
#
#   trace    `@code_hlo` with `optimization_passes = false`
#                                          -- no MLIR pipeline, no XLA
#   opt      `@code_hlo` with the production pipeline   the above PLUS Enzyme-JAX +
#                                          StableHLO pass pipeline
#   compile  `@compile`                    the above PLUS XLA:CPU codegen
#
# and by counting the ops in the module at each stage. A program SHAPE the
# emitter chooses (slices-plus-concatenate against one gather, an inlined body
# against a shared `func.func`) shows up in the census; a PASS superlinear in
# one of those op kinds shows up as `opt` time against a census the trace stage
# already reported; anything left in `compile - opt` is XLA:CPU's own codegen,
# which is what prints the "Very slow compile?" alarm.
#
# Env:
#   RESEACT_CC_PROG      rhsT | rhsC | ssp_step | ros_step | ssp_vjp | ros_vjp
#                        (default ssp_step) -- ONE per process. These are the
#                        adjoint driver's own programs, defined identically
#                        except that the Jacobian mode is `:fd`, so the probe
#                        needs no band model and no EarthSciASTDiff prepare.
#   RESEACT_CC_STAGES    subset of trace,opt,raw2,split,compile,cost
#                        (default trace,opt,raw2). `cost` implies `compile` and
#                        times RESEACT_CC_NCALL device calls of the compiled
#                        program, because a read form that compiles faster can
#                        execute slower and the two have to be measured
#                        together.
#                        The stages run IN THAT ORDER and the split is
#                        differences between them, because the first Reactant
#                        call in a process also pays for Julia's own JIT of the
#                        whole tracing stack:
#                          raw2         = warm EMISSION (no passes)
#                          opt  - raw2  = the Enzyme-JAX / StableHLO pipeline
#                          trace - raw2 = Julia JIT of the tracing stack
#                          compile - opt = XLA:CPU codegen
#                        `split` runs the `:all` pipeline one stage at a time
#                        on one module, timing and censusing each -- use it
#                        when `opt` is the number that has to be taken apart.
#   RESEACT_CC_DUMP      directory the module texts are STREAMED to (default:
#                        tools/diag/logs/cc, which is gitignored). Every stage
#                        writes its module there and the census reads it back
#                        line by line, because a module the probe holds in a
#                        Julia `String` is a copy the census then makes a second
#                        copy of, and at a continental grid one of those copies
#                        is hundreds of megabytes inside a 40 GiB cgroup. Point
#                        it at a scratch filesystem for the large grids.
#   RESEACT_CC_TAG       prefix for the written module files (default: the
#                        program name and the grid), so several runs can share
#                        one directory.
#   RESEACT_NLON/NLAT/NLEV, RESEACT_RES, RESEACT_T0, RESEACT_ADJ_UJITTER,
#   RESEACT_EXCLUDED_PASSES, RESEACT_ADJ_XLAFIX, RESEACT_RXENV
#
# Every stage prints RESIDENT SIZE beside its wall time, read from
# /proc/self/status. A compile that is expensive in TIME and one that is
# expensive in MEMORY are different problems with different levers, and the
# driver's peak is a compile rather than a run, so the two have to be read
# together.
#
# ONE PROGRAM PER PROCESS, and keep the heap hint small: this is meant to run
# beside other work in a 40 GiB cgroup.
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
using Printf, Statistics
say(s) = (println(s); flush(stdout))

# RESIDENT SIZE, from /proc and not from `Sys.maxrss`. Two reasons the kernel's
# number is the one that matters here: the compile's footprint is native (MLIR
# contexts, XLA arenas, LLVM section memory), which Julia's own accounting never
# sees; and `Sys.maxrss` is per-PROCESS-lifetime, so it cannot say that a stage
# handed its memory back. `VmRSS` is what the stage is holding now, `VmHWM` the
# high-water mark the scheduler will record.
function _rss()
    rss = hwm = 0
    for ln in eachline("/proc/self/status")
        startswith(ln, "VmRSS:") && (rss = parse(Int, split(ln)[2]))
        startswith(ln, "VmHWM:") && (hwm = parse(Int, split(ln)[2]))
    end
    return (rss = rss / 2^20, hwm = hwm / 2^20)   # GiB
end
_rssstr() = (r = _rss(); @sprintf("rss=%.2f hwm=%.2f GiB", r.rss, r.hwm))

const PROG = Symbol(get(ENV, "RESEACT_CC_PROG", "ssp_step"))
PROG in (:rhsT, :rhsC, :ssp_step, :ros_step, :ssp_vjp, :ros_vjp) ||
    error("RESEACT_CC_PROG must be rhsT, rhsC, ssp_step, ros_step, ssp_vjp or ros_vjp")
const STAGES = Set(String.(split(get(ENV, "RESEACT_CC_STAGES", "trace,opt,raw2"), ',')))
want(s) = s in STAGES
const DUMPDIR = let d = get(ENV, "RESEACT_CC_DUMP", "")
    isempty(d) ? joinpath(REPO, "tools", "diag", "logs", "cc") : d
end
mkpath(DUMPDIR)

import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Logging, Random
using EarthSciAST, EarthSciIO
using EarthSciASTSplitter: split_system
using Reactant
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize

get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const UJIT  = parse(Float64, get(ENV, "RESEACT_ADJ_UJITTER", "1e-1"))
_envi(k) = haskey(ENV, "RESEACT_$k") ? parse(Int, ENV["RESEACT_$k"]) : nothing
const RES   = get(ENV, "RESEACT_RES", "4x5")
const SLICE = native_slice(res = RES, lon0 = _envi("LON0"), lat0 = _envi("LAT0"),
                           nlon = _envi("NLON"), nlat = _envi("NLAT"),
                           nlev = _envi("NLEV"))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const GRIDSTR  = "$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$(NLEV_EFF)"
const TPTS  = Float64[T0, T0 + 8 * 3600, T0 + 16 * 3600]
const NDAYS = forcing_days_for(T0, last(TPTS))

say("="^78)
say(@sprintf("COMPILE COST  prog=%s  grid=%s  Reactant %s",
             PROG, GRIDSTR, pkgversion(RX)))
say("="^78)

validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing; merged_const = nothing; ov = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff, merged_const, ov
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

# --------------------------------------------------------------------------- #
# The op census: every `dialect.op` in the module text, then the kinds that
# distinguish the two emitters' program shapes.
# --------------------------------------------------------------------------- #
const _WATCH = ["stablehlo.slice", "stablehlo.concatenate", "stablehlo.dynamic_slice",
                "stablehlo.dynamic_update_slice", "stablehlo.gather", "stablehlo.scatter",
                "stablehlo.reshape", "stablehlo.broadcast_in_dim", "stablehlo.select",
                "stablehlo.reduce", "stablehlo.constant", "stablehlo.iota",
                "stablehlo.transpose", "stablehlo.pad", "stablehlo.convert",
                "stablehlo.while", "stablehlo.case", "stablehlo.custom_call",
                "enzyme.autodiff", "enzymexla.jit_call", "func.call"]

#
# THE CENSUS READS A FILE, not a string. Two reasons. It keeps the probe's peak
# heap independent of the module size, which is what lets the large grids run at
# all. And it gives the census a second axis a flat op count hides: the CONSTANT
# ELEMENT VOLUME. A program whose reads are gathers carries an i64 index
# constant per gather, so trading thousands of `slice` ops for one `gather` can
# move cost out of the op count and into the data the module ships; the op
# totals alone cannot tell the two apart. Every constant's widest tensor type on
# its line is summed, and `iota` beside it because a lowered index vector can
# arrive either way.
# Float and integer tensor types are counted SEPARATELY, because they answer
# different questions. Float constant volume is the model's own frozen data
# (stencil coefficients, cell geometry); integer constant volume is almost
# entirely GATHER INDICES, the thing a read form trades ops for. A fix that
# moves a module from thousands of slices to hundreds of gathers has to be read
# on both axes at once or it just relocates the cost.
const _TENSOR_RE_F = r"tensor<([0-9x]*)x?f(?:8|16|32|64)>"
const _TENSOR_RE_I = r"tensor<([0-9x]*)x?i(?:1|8|16|32|64)>"

function _tensor_elems(re::Regex, ln::AbstractString)
    best = 0
    for m in eachmatch(re, ln)
        d = m.captures[1]
        isempty(d) && (best = max(best, 1); continue)
        n = 1
        for tok in split(d, 'x')
            isempty(tok) && continue
            n *= parse(Int, tok)
        end
        best = max(best, n)
    end
    return best
end

function census(path::AbstractString)
    c = Dict{String,Int}(); nfunc = 0; nlines = 0
    felems = 0; ielems = 0; iotaelems = 0; nbytes = 0
    for ln in eachline(path)
        nlines += 1
        nbytes += sizeof(ln) + 1
        s = lstrip(ln)
        startswith(s, "func.func") && (nfunc += 1)
        m = match(r"\"?([a-z_]+\.[a-z_0-9]+)\"?[ (]", s)
        m === nothing && continue
        k = m.captures[1]
        c[k] = get(c, k, 0) + 1
        if k == "stablehlo.constant"
            felems += _tensor_elems(_TENSOR_RE_F, ln)
            ielems += _tensor_elems(_TENSOR_RE_I, ln)
        elseif k == "stablehlo.iota"
            iotaelems += max(_tensor_elems(_TENSOR_RE_F, ln),
                             _tensor_elems(_TENSOR_RE_I, ln))
        end
    end
    return (lines = nlines, funcs = nfunc, ops = c, total = sum(values(c); init = 0),
            felems = felems, ielems = ielems, iotaelems = iotaelems, bytes = nbytes)
end

function report_census(tag, mod)
    cs = census(mod)
    say(@sprintf("    %-24s %9d lines  %9d ops  %5d func.func  f64const=%d (%.1f MB)  i64const=%d (%.1f MB)  iota=%d  textMB=%.1f",
                 tag, cs.lines, cs.total, cs.funcs,
                 cs.felems, 8 * cs.felems / 2^20, cs.ielems, 8 * cs.ielems / 2^20,
                 cs.iotaelems, cs.bytes / 2^20))
    rows = sort!([(k, v) for (k, v) in cs.ops if v > 0]; by = last, rev = true)
    say("      top:     " * join([@sprintf("%s=%d", k, v) for (k, v) in first(rows, 12)], "  "))
    watched = [(w, get(cs.ops, w, 0)) for w in _WATCH]
    say("      watched: " * join([@sprintf("%s=%d", split(k, '.')[2], v)
                                  for (k, v) in watched if v > 0], "  "))
    return cs
end

# Write a module to `DUMPDIR` and return its PATH, which is what every stage
# hands the census. `show(io, mod)` streams; `repr(mod)` would not.
_dumptag() = get(ENV, "RESEACT_CC_TAG", "$(PROG)-$(GRIDSTR)")
function _dump(nm, mod)
    p = joinpath(DUMPDIR, "$(_dumptag())-$(nm).mlir")
    open(io -> show(io, mod), p, "w")
    return p
end

foreach(d -> d.materialize!(), dms)
say(@sprintf("BUILD %.2f s   nstates=%d  nparams=%d   %s",
             time() - tb, length(u0), length(p), _rssstr()))

# `Transport3D.m` from the real GEOS-FP surface pressure, then the harnesses'
# jitter -- the same base point every other probe and driver uses.
let dp0 = hydrostatic_dp(merged_param, ff.const_arrays, T0; slice = SLICE)
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dp0(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
                      parse(Int, mm.captures[3]))
    end
end
const N = length(u0)
const UBASE = let u = copy(u0)
    UJIT > 0 && (u .*= (1 .+ UJIT .* randn(Random.MersenneTwister(31337), N)))
    u
end

const RXDIR = joinpath(REPO, "tools", "reactant_handoff")
include(joinpath(REPO, "prototypes", "reseact_3d_chem", "blockdiag_local.jl")); using .BlockDiag
include(joinpath(REPO, "prototypes", "reseact_3d_chem", "block_jac.jl"))
include(joinpath(RXDIR, "rx_traced_integrator.jl"))
const RTI = RxTracedIntegrator
include(joinpath(REPO, "tools", "rx_rhs.jl"))
say(rx_rhs_banner())

const HOSTBUFS = [rx_bufs(fo[i]) for i in 1:2]
const G4 = [rx_rhs(fo[i]; var_map = var_map) for i in 1:2]
const PERM = cellmajor_perm(var_map)
const NS = PERM.NS; const NC = PERM.NC
const MASKS = RTI.species_masks(var_map, NS, NC)
const DEVBUFS = [map(RX.ConcreteRArray, hb) for hb in HOSTBUFS]
for i in 1:2; rx_sync!(DEVBUFS[i], fo[i]); end
_devp(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))
const PRd = _devp(p)
const THT = (p = PRd, bufs = DEVBUFS[1])
const THC = (p = PRd, bufs = DEVBUFS[2])

const RTOL, ATOL_T, ATOL_C = 1e-4, 1e-6, 1e-9
const DT0T = parse(Float64, get(ENV, "RESEACT_DT0T", "15.0"))
const DT0C = parse(Float64, get(ENV, "RESEACT_DT0C", "0.5"))

gT(u, th, t) = G4[1](u, th.p, t, th.bufs)
gC(u, th, t) = G4[2](u, th.p, t, th.bufs)

rhsT(u, th, t) = gT(u, th, t)
rhsC(u, th, t) = gC(u, th, t)
ssp_step(u, th, t, dt) = RTI.ssprk43_step_unrolled((uu, tt) -> gT(uu, th, tt),
                                                   u, t, dt, ATOL_T, RTOL)
ros_step(u, th, t, dt) = RTI.ros23_step((uu, tt) -> gC(uu, th, tt), u, t, dt,
                                        NS, NC, MASKS, ATOL_C, RTOL;
                                        unrolled = true, jac = :fd)
ssp_vjp(u, th, lam, t, dt) = RTI.ssprk43_step_vjp(gT, u, th, t, dt, lam, ATOL_T, RTOL;
                                                  active_bufs = false)
ros_vjp(u, th, lam, t, dt) = RTI.ros23_step_vjp(gC, u, th, t, dt, lam,
                                                NS, NC, MASKS, ATOL_C, RTOL;
                                                jac = :fd, gj = nothing,
                                                active_bufs = false)

const U_R   = RX.ConcreteRArray(copy(UBASE))
const T_R   = RX.ConcreteRNumber(T0)
const DTT_R = RX.ConcreteRNumber(DT0T)
const DTC_R = RX.ConcreteRNumber(DT0C)
const LAM_R = RX.ConcreteRArray(fill(1.0 / N, N))

const XLAFIX = get(ENV, "RESEACT_ADJ_XLAFIX", "1") == "1"
const _EXCL2 = get(ENV, "RESEACT_EXCLUDED_PASSES", "dynamic_update_to_concat,sub_const_prop")
const EXCL2 = _EXCL2 == "none" ? String[] : String.(filter(!isempty, strip.(split(_EXCL2, ','))))
# A TRAP WORTH NAMING. `@code_hlo compile_options = X optimize = false` does NOT
# turn the pass pipeline off: `__compile_options_from_kwargs` returns `X`
# unchanged the moment `compile_options` is present (Reactant
# src/CompileOptions.jl), so `optimize` is silently dropped -- and `optimize` is
# only ever a spelling of the `optimization_passes` FIELD anyway. The
# decomposition therefore builds two option sets that differ in that one field,
# and nothing else, so `opt - raw` is the pass pipeline and nothing else.
_copts(passes) = XLAFIX ?
    RX.CompileOptions(; sync = true, optimization_passes = passes,
                      xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
                      (isempty(EXCL2) ? (;) : (; excluded_passes = EXCL2))...) :
    RX.CompileOptions(; sync = true, optimization_passes = passes,
                      (isempty(EXCL2) ? (;) : (; excluded_passes = EXCL2))...)
const COPTS0 = _copts(false)    # trace / emit only
const COPTS2 = _copts(:all)     # the production pipeline

say("="^78)
say(@sprintf("COMPILE COST  prog=%s  grid=%s  stages=%s",
             PROG, GRIDSTR, join(sort(collect(STAGES)), ",")))
say(@sprintf("  excluded_passes=%s  xlafix=%s  XLA_FLAGS=%s",
             isempty(EXCL2) ? "(none)" : join(EXCL2, ","), XLAFIX,
             get(ENV, "XLA_FLAGS", "(unset)")))
say("="^78)

# --------------------------------------------------------------------------- #
# THE PASS SPLIT. `opt` is one number over a pipeline of a dozen stages, and on
# the reverse-mode program that number is hours, so it has to be taken apart.
# Reactant assembles `optimization_passes = :all` for a CPU backend as the list
# below (`src/compiler/Compiler.jl`, the `:all` branch, with `raise = false`);
# this reproduces it stage by stage on ONE module, timing and censusing each,
# so the stage that does not finish is named rather than guessed. The sum of
# the stage times and the final census are checked against a plain `opt` run on
# the same program -- do that on `ssp_step`, where `opt` is seconds.
#
#   RESEACT_CC_SPLIT_STOP   stop after this stage (default: run them all)
#   RESEACT_CC_SPLIT_QUIET  comma-separated stages to skip the census of, for
#                           when printing the module text is itself the cost
# --------------------------------------------------------------------------- #
const MLIR = RX.MLIR
const RXC  = RX.Compiler
const SPLIT_STOP  = get(ENV, "RESEACT_CC_SPLIT_STOP", "")
const SPLIT_QUIET = Set(String.(filter(!isempty, strip.(split(get(ENV, "RESEACT_CC_SPLIT_QUIET", ""), ',')))))

function split_stages(co::RX.CompileOptions)
    kw = (; recognize_comms = true, lower_comms = true, backend = "cpu", is_sharded = false)
    opt1 = RXC.optimization_passes(co; sroa = true, hlo_opts = true, kw...)
    opt2 = RXC.optimization_passes(co; sroa = false, kw...)
    biw  = sizeof(LinearAlgebra.BlasInt) * 8
    lower = join(["lower-enzymexla-linalg{backend=cpu blas_int_width=$biw}",
                  "lower-enzymexla-blas{backend=cpu blas_int_width=$biw}",
                  "lower-enzymexla-lapack{backend=cpu blas_int_width=$biw}",
                  "lower-enzymexla-math", "lower-enzymexla-mpi{backend=cpu}"], ",")
    return [("mark",   "mark-func-memory-effects"),
            ("opt1",   opt1),
            ("ebatch", "enzyme-batch"),
            ("opt2a",  opt2),
            ("enzyme", RXC.enzyme_pass),
            ("opt2b",  opt2),
            ("clean",  "canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math"),
            ("opt2c",  opt2),
            ("kern",   "lower-kernel{backend=cpu},canonicalize"),
            ("raise",  "canonicalize"),
            ("lower",  lower),
            ("jit",    "lower-jit{openmp=$(RXC.OpenMP[]) backend=cpu},symbol-dce")]
end

function run_split(f, args)
    stages = split_stages(COPTS2)
    say(@sprintf("  split: %d stages, stop=%s", length(stages),
                 isempty(SPLIT_STOP) ? "(none)" : SPLIT_STOP))
    tot = Ref(0.0)
    MLIR.IR.@dispose ctx = RX.ReactantContext() begin
        t0 = time()
        mod = RXC.code_hlo(ctx, f, args; compile_options = COPTS0)
        say(@sprintf("  %-8s %9.1f s   (emission, no passes)   %s",
                     "emit", time() - t0, _rssstr()))
        MLIR.IR.activate(ctx)
        try
            report_census("emit", _dump("split-emit", mod))
            for (nm, pipe) in stages
                # Announced BEFORE it runs, so a run killed by the wall clock
                # still names the stage that did not finish.
                say(@sprintf("  %-8s ... running", nm))
                t = time(); RXC.run_pass_pipeline!(mod, pipe, nm); dt = time() - t
                tot[] += dt
                say(@sprintf("  %-8s %9.1f s   (cumulative %.1f s)   %s",
                             nm, dt, tot[], _rssstr()))
                nm in SPLIT_QUIET || report_census(nm, _dump("split-$nm", mod))
                nm == SPLIT_STOP && break
            end
        finally
            MLIR.IR.deactivate(ctx)
        end
    end
    say(@sprintf("  SPLIT TOTAL %.1f s   %s", tot[], _rssstr()))
    return nothing
end

# `@code_hlo` / `@compile` need a LITERAL call expression, so each program gets
# its own three lines rather than a thunk that takes the macro.
# --------------------------------------------------------------------------- #
# READ ATTRIBUTION. The direct emitter tallies every emitted slice / gather /
# concatenate against the emitter SITE that was running (`_de_at!`), so the
# surviving single-position reads can be charged to the scalar spine, a fill
# level, an access kernel, a scan or the output assembly rather than guessed at.
# One emission of one right-hand side; `stats` holds the LAST one.
# --------------------------------------------------------------------------- #
function report_sites(tag, g)
    st = try
        getfield(getfield(g, :d), :stats)
    catch
        nothing
    end
    st === nothing && return nothing
    rows = sort!([(String(k), v) for (k, v) in st if occursin('@', String(k))];
                 by = last, rev = true)
    isempty(rows) && return nothing
    say("  SITE ATTRIBUTION ($tag), one emission:")
    tot = Dict{String,Int}()
    for (k, v) in rows
        pre, rest = split(k, '@')
        site, why = split(rest, '.')
        say(@sprintf("    %-10s %-14s %-14s %8d", pre, site, why, v))
        tot[pre] = get(tot, pre, 0) + v
    end
    say("    TOTALS: " * join([@sprintf("%s=%d", k, v) for (k, v) in sort(collect(tot))], "  "))
    return nothing
end

# WHAT THE PROGRAM COSTS TO RUN, which is the other half of every read-form
# decision. A read emitted as one `stablehlo.gather` is one operation where
# slices-plus-concatenate are several, and the passes that deduplicate slices
# are quadratic in how many there are -- but a gather is an indexed copy at
# runtime and a slice is a contiguous one, so a shape that compiles faster can
# execute slower. Both numbers have to come out of the same process or the
# trade cannot be read at all.
const NCALL = parse(Int, get(ENV, "RESEACT_CC_NCALL", "20"))
function cost(thunk, args)
    thunk(args...)                                   # warm-up, not timed
    ts = Float64[]
    for _ in 1:NCALL
        t0 = time(); thunk(args...); push!(ts, time() - t0)
    end
    sort!(ts)
    say(@sprintf("  %-8s %9.2f ms/call median (min %.2f, max %.2f, n=%d)   %s",
                 "cost", 1e3 * ts[(length(ts) + 1) ÷ 2], 1e3 * first(ts),
                 1e3 * last(ts), length(ts), _rssstr()))
    return nothing
end

MOD0 = Ref{Any}(nothing); MOD1 = Ref{Any}(nothing); MOD2 = Ref{Any}(nothing)
function stage(tag, f)
    t0 = time(); m = f(); dt = time() - t0
    say(@sprintf("  %-8s %9.1f s   %s", tag, dt, _rssstr()))
    return m
end

if PROG === :rhsT
    want("trace")   && (MOD0[] = stage("trace",   () -> _dump("raw", RX.@code_hlo compile_options=COPTS0 rhsT(U_R, THT, T_R))))
    want("opt")     && (MOD1[] = stage("opt",     () -> _dump("opt", RX.@code_hlo compile_options=COPTS2 rhsT(U_R, THT, T_R))))
    want("raw2")   && (MOD2[] = stage("raw2",    () -> _dump("raw2", RX.@code_hlo compile_options=COPTS0 rhsT(U_R, THT, T_R))))
    want("split")   && run_split(rhsT, (U_R, THT, T_R,))
    if want("compile") || want("cost")
        th = stage("compile", () -> RX.@compile compile_options=COPTS2 rhsT(U_R, THT, T_R))
        want("cost") && cost(th, (U_R, THT, T_R))
    end
elseif PROG === :rhsC
    want("trace")   && (MOD0[] = stage("trace",   () -> _dump("raw", RX.@code_hlo compile_options=COPTS0 rhsC(U_R, THC, T_R))))
    want("opt")     && (MOD1[] = stage("opt",     () -> _dump("opt", RX.@code_hlo compile_options=COPTS2 rhsC(U_R, THC, T_R))))
    want("raw2")   && (MOD2[] = stage("raw2",    () -> _dump("raw2", RX.@code_hlo compile_options=COPTS0 rhsC(U_R, THC, T_R))))
    want("split")   && run_split(rhsC, (U_R, THC, T_R,))
    if want("compile") || want("cost")
        th = stage("compile", () -> RX.@compile compile_options=COPTS2 rhsC(U_R, THC, T_R))
        want("cost") && cost(th, (U_R, THC, T_R))
    end
elseif PROG === :ssp_step
    want("trace")   && (MOD0[] = stage("trace",   () -> _dump("raw", RX.@code_hlo compile_options=COPTS0 ssp_step(U_R, THT, T_R, DTT_R))))
    want("opt")     && (MOD1[] = stage("opt",     () -> _dump("opt", RX.@code_hlo compile_options=COPTS2 ssp_step(U_R, THT, T_R, DTT_R))))
    want("raw2")   && (MOD2[] = stage("raw2",    () -> _dump("raw2", RX.@code_hlo compile_options=COPTS0 ssp_step(U_R, THT, T_R, DTT_R))))
    want("split")   && run_split(ssp_step, (U_R, THT, T_R, DTT_R,))
    if want("compile") || want("cost")
        th = stage("compile", () -> RX.@compile compile_options=COPTS2 ssp_step(U_R, THT, T_R, DTT_R))
        want("cost") && cost(th, (U_R, THT, T_R, DTT_R))
    end
elseif PROG === :ros_step
    want("trace")   && (MOD0[] = stage("trace",   () -> _dump("raw", RX.@code_hlo compile_options=COPTS0 ros_step(U_R, THC, T_R, DTC_R))))
    want("opt")     && (MOD1[] = stage("opt",     () -> _dump("opt", RX.@code_hlo compile_options=COPTS2 ros_step(U_R, THC, T_R, DTC_R))))
    want("raw2")   && (MOD2[] = stage("raw2",    () -> _dump("raw2", RX.@code_hlo compile_options=COPTS0 ros_step(U_R, THC, T_R, DTC_R))))
    want("split")   && run_split(ros_step, (U_R, THC, T_R, DTC_R,))
    if want("compile") || want("cost")
        th = stage("compile", () -> RX.@compile compile_options=COPTS2 ros_step(U_R, THC, T_R, DTC_R))
        want("cost") && cost(th, (U_R, THC, T_R, DTC_R))
    end
elseif PROG === :ssp_vjp
    want("trace")   && (MOD0[] = stage("trace",   () -> _dump("raw", RX.@code_hlo compile_options=COPTS0 ssp_vjp(U_R, THT, LAM_R, T_R, DTT_R))))
    want("opt")     && (MOD1[] = stage("opt",     () -> _dump("opt", RX.@code_hlo compile_options=COPTS2 ssp_vjp(U_R, THT, LAM_R, T_R, DTT_R))))
    want("raw2")   && (MOD2[] = stage("raw2",    () -> _dump("raw2", RX.@code_hlo compile_options=COPTS0 ssp_vjp(U_R, THT, LAM_R, T_R, DTT_R))))
    want("split")   && run_split(ssp_vjp, (U_R, THT, LAM_R, T_R, DTT_R,))
    if want("compile") || want("cost")
        th = stage("compile", () -> RX.@compile compile_options=COPTS2 ssp_vjp(U_R, THT, LAM_R, T_R, DTT_R))
        want("cost") && cost(th, (U_R, THT, LAM_R, T_R, DTT_R))
    end
else
    want("trace")   && (MOD0[] = stage("trace",   () -> _dump("raw", RX.@code_hlo compile_options=COPTS0 ros_vjp(U_R, THC, LAM_R, T_R, DTC_R))))
    want("opt")     && (MOD1[] = stage("opt",     () -> _dump("opt", RX.@code_hlo compile_options=COPTS2 ros_vjp(U_R, THC, LAM_R, T_R, DTC_R))))
    want("raw2")   && (MOD2[] = stage("raw2",    () -> _dump("raw2", RX.@code_hlo compile_options=COPTS0 ros_vjp(U_R, THC, LAM_R, T_R, DTC_R))))
    want("split")   && run_split(ros_vjp, (U_R, THC, LAM_R, T_R, DTC_R,))
    if want("compile") || want("cost")
        th = stage("compile", () -> RX.@compile compile_options=COPTS2 ros_vjp(U_R, THC, LAM_R, T_R, DTC_R))
        want("cost") && cost(th, (U_R, THC, LAM_R, T_R, DTC_R))
    end
end

report_sites(PROG in (:rhsC, :ros_step, :ros_vjp) ? "chemistry" : "transport",
             PROG in (:rhsC, :ros_step, :ros_vjp) ? G4[2] : G4[1])
MOD0[] === nothing || report_census("raw   (no passes)", MOD0[])
MOD1[] === nothing || report_census("opt   (full pipeline)",  MOD1[])
MOD2[] === nothing || report_census("raw2  (warm, no passes)", MOD2[])
say("  module text: $DUMPDIR ($(_dumptag())-*.mlir)")
say("  resident at exit: " * _rssstr())
say("DONE $(PROG)")
