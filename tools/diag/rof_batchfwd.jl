#!/usr/bin/env julia
# Does BATCHED forward mode work under Reactant? -- the "remove the nesting"
# option floated for `ad_block_jac` (one width-NS `enzyme.fwddiff` instead of NS
# separate ones). Standalone, no integrator, no model: 60 s to answer.
#   julia --project=$RESEACT_RXENV tools/diag/rof_batchfwd.jl
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf, LinearAlgebra
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end
const EZ = Reactant.Enzyme
include(joinpath(REPO, "tools", "reactant_handoff", "rx_native_patch.jl"))

const NS = 4; const NC = 3; const N = NS * NC
blk(u, s) = u[((s - 1) * NC + 1):(s * NC)]
function rhs(u)
    o = Vector{Any}(undef, NS)
    for s in 1:NS
        a = blk(u, s); b = blk(u, s == NS ? 1 : s + 1)
        ab = a .* b; sq = a .* a; o[s] = ab .- sq
    end
    return vcat(o...)
end
seeds = ntuple(NS) do s
    m = zeros(Float64, N); m[((s - 1) * NC + 1):(s * NC)] .= 1.0; m
end

# (a) NS separate width-1 fwddiffs -- what ad_block_jac does today
function cols_serial(u)
    cs = ntuple(Val(NS)) do s
        v = RX.Ops.constant(seeds[s])
        EZ.autodiff(EZ.Forward, EZ.Const(rhs), EZ.Duplicated, EZ.Duplicated(u, v))[1]
    end
    return vcat(cs...)
end
# (b) ONE width-NS fwddiff
function cols_batched(u)
    vs = ntuple(s -> RX.Ops.constant(seeds[s]), Val(NS))
    r = EZ.autodiff(EZ.Forward, EZ.Const(rhs), EZ.BatchDuplicated,
                    EZ.BatchDuplicated(u, vs))[1]
    return vcat(ntuple(s -> r[s], Val(NS))...)
end

# (c) ONE fwddiff inside an `enzyme.batch` region: `Ops.batch` maps over a
# leading batch dim, so the NS colours become the batch and the derivative is
# emitted once. `u` has to be replicated into the batch because Ops.batch takes
# all of its inputs batched.
function cols_opsbatch(u)
    seedmat = RX.Ops.constant(reduce(vcat, [reshape(s, 1, N) for s in seeds]))  # NS x N
    umat = RX.Ops.constant(zeros(NS, N)) .+ reshape(u, 1, N)                    # NS x N
    r = only(RX.Ops.batch([umat, seedmat], [NS]) do uu, vv
        EZ.autodiff(EZ.Forward, EZ.Const(rhs), EZ.Duplicated, EZ.Duplicated(uu, vv))[1]
    end)
    return reshape(r, NS * N)     # layout differs from (a)/(b); we only ask if it LOWERS
end

uh = 0.5 .+ collect(1:N) ./ N
U = RX.ConcreteRArray(uh)
for (nm, f) in (("serial ", cols_serial), ("batched", cols_batched),
                ("opsbatch", cols_opsbatch))
    try
        t0 = time(); c = @compile f(U); tc = time() - t0
        s = sprint(show, @code_hlo optimize=false f(U))
        nfwd = count(_ -> true, eachmatch(r"enzyme\.fwddiff", s))
        @printf("%s  compile %6.1f s  ||cols||=%.10e  enzyme.fwddiff ops=%d\n",
                nm, tc, norm(Array(c(U))), nfwd)
    catch e
        msg = sprint(showerror, e)
        println("$nm  FAILED (", length(msg), " chars); tail:")
        println(msg[max(1, end - 2500):end])
    end
    flush(stdout)
end
