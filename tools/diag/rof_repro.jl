#!/usr/bin/env julia
# ===========================================================================
# rof_repro.jl -- BISECTION HARNESS for the reverse-over-forward SEGFAULT.
# ===========================================================================
# `tools/rx_adjoint_toy.jl` shows the reverse-over-forward adjoint
# (`ros23_step_vjp` with `jac=:ad`, i.e. Enzyme reverse over the
# `enzyme.fwddiff` calls `ad_block_jac` emits) is EXACT on a 3-species/5-cell
# toy, while the same nesting SEGFAULTS on ReSEACT inside Enzyme-MLIR:
#
#     AutoDiffCallRev::createReverseModeAdjoint -> func::CallOp::build ->
#     Operation::getAttr on a NULL FuncOp,   from DifferentiatePass
#
# This script walks the gap between the two, ONE KNOB PER PROCESS (a segfault
# takes the process with it, so every configuration has to be its own run and
# the verdict has to be written by the PARENT -- see rof_sweep.sh).
#
# Knobs, all via the environment:
#   ROF_NS      species count = number of nested `enzyme.fwddiff` calls  [3]
#   ROF_NC      cells per species block                                  [5]
#   ROF_THETA   vec | nt    shape of the differentiable payload          [vec]
#               vec: a plain Vector, as in the toy
#               nt : (p = NamedTuple of traced SCALARS, bufs = Tuple of
#                    traced ARRAYS), as ReSEACT's `gC(u, th, t)` takes
#   ROF_EXTRA   comma-separated RHS constructs to switch on              [none]
#               none log10 floor sumreduce cast scatter gather select pow
#               dyngather (index computed from t -- a traced Int, i.e. what a
#               forcing interpolation does) and traceif (`@trace if`, a
#               stablehlo.if region) are the two that are NOT modelled by
#               rx_adjoint_toy.jl's variants.
#   ROF_ASM     vcat | dus  how the RHS assembles its NS output blocks    [vcat]
#               vcat: `vcat(blocks...)` -- ONE NS-operand stablehlo.concatenate.
#                     At NS >= ~6 this trips a SEPARATE Reactant/EnzymeXLA HLO
#                     bug (adjacent-slice merge across the wrong axis, see
#                     UPSTREAM_reverse_over_forward.md #2) and never reaches the
#                     question this script is asking.
#               dus : blocks written into one buffer by `setindex!`
#                     (stablehlo.dynamic_update_slice), which has no concat.
#   ROF_JAC     ad | fd     block Jacobian flavour                       [ad]
#   ROF_MODE    vjp | jvp | out   what to compile                        [vjp]
#   ROF_PATCH   1 | 0       include rx_native_patch.jl                   [1]
#   ROF_EXCL    comma-separated enzyme-hlo pattern names to exclude from the
#               pipeline (Reactant `CompileOptions(; excluded_passes=...)`).
#               `concat_reshape_slice` is the one bug #2 lives in.
#   ROF_HLO     path        dump the pre-pipeline module here and exit
#
# Exit 0 = compiled. Anything else (incl. signal death) = the configuration
# reproduces. `ROF_HLO` never runs the pipeline, so it survives a crashing
# configuration and is how the MLIR for the bug report is captured.
# ===========================================================================
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf, LinearAlgebra, Random
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end
const EZ = Reactant.Enzyme

const NS      = parse(Int, get(ENV, "ROF_NS", "3"))
const NC      = parse(Int, get(ENV, "ROF_NC", "5"))
const THETA   = get(ENV, "ROF_THETA", "vec")
const EXTRA   = Set(split(get(ENV, "ROF_EXTRA", "none"), ","))
const ASM     = get(ENV, "ROF_ASM", "vcat")
const JAC     = Symbol(get(ENV, "ROF_JAC", "ad"))
const MODE    = get(ENV, "ROF_MODE", "vjp")
const PATCH   = get(ENV, "ROF_PATCH", "1") == "1"
const HLO     = get(ENV, "ROF_HLO", "")
const EXCL    = filter(!isempty, String.(split(get(ENV, "ROF_EXCL", ""), ",")))
const N       = NS * NC

PATCH && include(joinpath(REPO, "tools", "reactant_handoff", "rx_native_patch.jl"))
include(joinpath(REPO, "tools", "reactant_handoff", "rx_traced_integrator.jl"))
const RTI = RxTracedIntegrator

const MASKS = [begin m = zeros(N); m[((s-1)*NC+1):(s*NC)] .= 1.0; m end for s in 1:NS]
const ATOL = 1e-9; const RTOL = 1e-4
const CFG = "NS=$NS NC=$NC theta=$THETA extra=$(join(sort(collect(EXTRA)),'+')) " *
            "asm=$ASM jac=$JAC mode=$MODE patch=$(PATCH ? 1 : 0)"
say(s) = (println(s); flush(stdout))
say("=== rof_repro: $CFG ===")

# --------------------------------------------------------------------------
# the RHS. NS coupled cell-local mass-action blocks, plus whichever of the
# constructs ReSEACT's emitted RHS uses are switched on.
# --------------------------------------------------------------------------
_perm = let r = Random.MersenneTwister(7); randperm(r, NC); end

# `pk(th, s)` -- rate constant s, however theta is shaped.
_pk(th::AbstractVector, s) = th[s:s]
_pk(th::NamedTuple, s) = values(th.p)[s]
_bufs(th::AbstractVector) = ()
_bufs(th::NamedTuple) = th.bufs

function extra_term(C, th, t)
    e = 0.0 .* C
    if "log10" in EXTRA
        l = log10.(C .+ 1.0); l = 1e-3 .* l; e = e .+ l
    end
    if "floor" in EXTRA
        fl = floor.(C); fl = 1e-3 .* fl; e = e .+ fl
    end
    if "sumreduce" in EXTRA
        q = C ./ C; sq = sum(C) .* q; sq = 1e-3 .* sq; e = e .+ sq
    end
    if "cast" in EXTRA               # TypeCast_broadcast_scalar
        i = floor.(Int, C .* 10.0); f = Float64.(i); f = 1e-4 .* f; e = e .+ f
    end
    if "scatter" in EXTRA            # update_computation (stablehlo.scatter)
        y = similar(C); y[_perm] = C; y = 1e-3 .* y; e = e .+ y
    end
    if "gather" in EXTRA
        g = C[_perm]; g = 1e-3 .* g; e = e .+ g
    end
    if "select" in EXTRA
        sel = ifelse.(C .> 0.5, C, 0.5 .* C); sel = 1e-3 .* sel; e = e .+ sel
    end
    if "pow" in EXTRA
        pw = C .^ 1.5; pw = 1e-3 .* pw; e = e .+ pw
    end
    if "dyngather" in EXTRA          # traced-Int index, as an interpolation makes
        k = 1 + (floor(Int, t) % NC)
        v = RX.Ops.dynamic_slice(C, [k], [1])   # stablehlo.dynamic_slice
        dg = v .* (C ./ C); dg = 1e-3 .* dg; e = e .+ dg
    end
    if "traceif" in EXTRA            # a stablehlo.if region
        local br
        Reactant.@trace if t > 0.0
            br = 1e-3 .* C
        else
            br = 2e-3 .* C
        end
        e = e .+ br
    end
    bs = _bufs(th)
    if !isempty(bs)                # forcing buffers, as ReSEACT's theta carries
        b1 = bs[1][1:NC]; b1 = 1e-3 .* b1; e = e .+ b1
        if length(bs) > 1
            b2 = bs[2][1:NC]; b2 = b2 .* C; b2 = 1e-3 .* b2; e = e .+ b2
        end
    end
    return e
end

_blk(u, s) = u[((s - 1) * NC + 1):(s * NC)]
_nxt(s) = s == NS ? 1 : s + 1
_nxt2(s) = _nxt(_nxt(s))

function grhs(u, th, t)
    sc = 1.0 + 0.1 * sin(t)
    blocks = [_blk(u, s) for s in 1:NS]
    out = Vector{Any}(undef, NS)
    for s in 1:NS
        A = blocks[s]; B = blocks[_nxt(s)]; C = blocks[_nxt2(s)]
        ab = A .* B
        r1 = _pk(th, s) .* ab
        r1 = sc .* r1
        r2 = _pk(th, _nxt(s)) .* C
        cc = C .* C
        r3 = _pk(th, _nxt2(s)) .* cc
        d = r2 .- r1
        d = d .+ r3
        out[s] = s == NS ? (d .+ extra_term(A, th, t)) : d
    end
    ASM == "vcat" && return vcat(out...)
    ASM == "dus" || error("unknown ROF_ASM '$ASM'")
    y = 0.0 .* u
    for s in 1:NS
        y[((s - 1) * NC + 1):(s * NC)] = out[s]
    end
    return y
end

# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
Random.seed!(11)
uh = 0.5 .+ rand(N)
lamh = randn(N); vuh = randn(N)
U = RX.ConcreteRArray(uh); LAM = RX.ConcreteRArray(lamh); VU = RX.ConcreteRArray(vuh)
T = RX.ConcreteRNumber(100.0); DT = RX.ConcreteRNumber(0.05)

ph = [0.2 + 0.5 * rand() for _ in 1:NS]
if THETA == "vec"
    TH  = RX.ConcreteRArray(ph)
    dTH = RX.ConcreteRArray(randn(NS))
elseif THETA == "nt"
    keysyms = Tuple(Symbol("k", s) for s in 1:NS)
    pnt  = NamedTuple{keysyms}(Tuple(RX.ConcreteRNumber(x) for x in ph))
    dpnt = NamedTuple{keysyms}(Tuple(RX.ConcreteRNumber(randn()) for _ in 1:NS))
    bufs  = (RX.ConcreteRArray(rand(NC)), RX.ConcreteRArray(rand(NC)))
    dbufs = (RX.ConcreteRArray(randn(NC)), RX.ConcreteRArray(randn(NC)))
    TH  = (p = pnt,  bufs = bufs)
    dTH = (p = dpnt, bufs = dbufs)
else
    error("unknown ROF_THETA '$THETA'")
end

rout(u, th, t, dt)         = RTI.ros23_step_out(grhs, u, th, t, dt, NS, NC, MASKS, ATOL, RTOL; jac = JAC)
rvjp(u, th, lam, t, dt)    = RTI.ros23_step_vjp(grhs, u, th, t, dt, lam, NS, NC, MASKS, ATOL, RTOL; jac = JAC)
rjvp(u, du, th, dth, t, dt) = RTI.ros23_step_jvp(grhs, u, du, th, dth, t, dt, NS, NC, MASKS, ATOL, RTOL; jac = JAC)

# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
if !isempty(HLO)
    m = MODE == "vjp" ? (@code_hlo optimize=false rvjp(U, TH, LAM, T, DT)) :
        MODE == "jvp" ? (@code_hlo optimize=false rjvp(U, VU, TH, dTH, T, DT)) :
                        (@code_hlo optimize=false rout(U, TH, T, DT))
    open(HLO, "w") do io; show(io, m); end
    say("HLO written to $HLO ($(filesize(HLO)) bytes)")
    exit(0)
end

const COPTS = isempty(EXCL) ? RX.CompileOptions() :
              RX.CompileOptions(; excluded_passes = EXCL)
isempty(EXCL) || say("  excluded_passes = $EXCL")

t0 = time()
if MODE == "vjp"
    c = @compile compile_options = COPTS rvjp(U, TH, LAM, T, DT)
    r = c(U, TH, LAM, T, DT)
    say(@sprintf("OK vjp  compile %.1f s  ||lam_in||=%.6e", time() - t0, norm(Array(r[1]))))
elseif MODE == "jvp"
    c = @compile compile_options = COPTS rjvp(U, VU, TH, dTH, T, DT)
    r = c(U, VU, TH, dTH, T, DT)
    say(@sprintf("OK jvp  compile %.1f s  ||Jv||=%.6e", time() - t0, norm(Array(r))))
else
    c = @compile compile_options = COPTS rout(U, TH, T, DT)
    r = c(U, TH, T, DT)
    say(@sprintf("OK out  compile %.1f s  ||unew||=%.6e", time() - t0, norm(Array(r))))
end
say("DONE $CFG")
