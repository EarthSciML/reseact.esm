#!/usr/bin/env julia
# ===========================================================================
# c2d_structural.jl -- DID `CONCATS_TO_DUS` ACTUALLY FIRE?
# ===========================================================================
# Measured at NC=2184: CONCATS_TO_DUS[]=true moved the ros23 step 45.765 ->
# 40.878 ms (1.12x), and the bare chem RHS the WRONG way, 7.441 -> 7.938 ms.
# The prediction from the CONUS dump was 1.6-2.0x, so something is off. There
# are two very different reasons it could be:
#
#   (a) the `concat_to_onedim_dus` pattern matched few or none of the
#       whole-buffer concatenates -- the LEVER is weak, and the traffic claim is
#       untested; or
#   (b) it matched them all and the traffic simply is not what the step is
#       spending its time on -- the MECHANISM claim is wrong.
#
# Those are distinguishable WITHOUT any timing: count `stablehlo.concatenate`
# and `stablehlo.dynamic_update_slice` in the module Reactant hands to XLA, with
# the flag off and on, in ONE process off ONE build. If the big concatenates
# survive, it is (a). If they are gone and the time barely moved, it is (b) and
# the buffer-traffic story does not survive.
#
# Prints the count AND the size distribution, because only the >=400k ones carry
# the 864 MB the claim is about.
#
#   RESEACT_NLON/NLAT/NLEV  grid (default 13/7/24)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "24")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "c2dstruct"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX
say(s) = (println(s); flush(stdout))

UR = RX.ConcreteRArray(copy(D.UBASE))
TR = RX.ConcreteRNumber(D.T0)
DR = RX.ConcreteRNumber(D.DT0C)

# every `stablehlo.OP` occurrence, and for concatenate/DUS also the result size
function opcounts(txt::AbstractString)
    h = Dict{String,Int}()
    for m in eachmatch(r"=\s+\"?(stablehlo\.[\w.]+)\"?", txt)
        h[m.captures[1]] = get(h, m.captures[1], 0) + 1
    end
    return h
end

# result element counts for lines mentioning a given op, from the `-> tensor<Nxf64>`
# (or `tensor<NxMxf64>`) result type at the end of the line
function sizes_for(txt::AbstractString, op::AbstractString)
    out = Int[]
    for ln in split(txt, '\n')
        occursin(op, ln) || continue
        m = match(r"->\s*tensor<([0-9x]+)x[a-z]", ln)
        m === nothing && continue
        n = 1
        for d in split(m.captures[1], 'x'); n *= parse(Int, d); end
        push!(out, n)
    end
    return out
end

function survey(label::AbstractString)
    t0 = time_ns()
    txt = sprint(show, RX.@code_hlo(D.ros_step(UR, D.THC, TR, DR)))
    el = (time_ns() - t0) / 1e9
    h = opcounts(txt)
    cc = sizes_for(txt, "stablehlo.concatenate")
    du = sizes_for(txt, "stablehlo.dynamic_update_slice")
    bigc = filter(>=(400_000), cc); bigd = filter(>=(400_000), du)
    say("")
    say("  ---- $label  (compile $(round(el; digits=1)) s, $(sum(values(h))) stablehlo ops) ----")
    @printf("    concatenate           total %5d   >=400k %4d   elems in big %8.1f M = %6.0f MB\n",
            length(cc), length(bigc), sum(bigc; init=0)/1e6, sum(bigc; init=0)*8/1e6)
    @printf("    dynamic_update_slice  total %5d   >=400k %4d   elems in big %8.1f M = %6.0f MB\n",
            length(du), length(bigd), sum(bigd; init=0)/1e6, sum(bigd; init=0)*8/1e6)
    @printf("    scatter %d   slice %d   pad %d\n",
            get(h,"stablehlo.scatter",0), get(h,"stablehlo.slice",0), get(h,"stablehlo.pad",0))
    return (nbigc = length(bigc), bigc = sum(bigc; init=0),
            nbigd = length(bigd), bigd = sum(bigd; init=0))
end

say("\n" * "="^78)
say(@sprintf("CONCATS_TO_DUS STRUCTURAL A/B   NS=%d NC=%d N=%d", D.NS, D.NC, D.NS*D.NC))
say("="^78)

RX.Compiler.CONCATS_TO_DUS[] = false
off = survey("CONCATS_TO_DUS = false")
RX.Compiler.CONCATS_TO_DUS[] = true
on  = survey("CONCATS_TO_DUS = true")
RX.Compiler.CONCATS_TO_DUS[] = false

say("")
say("="^78)
@printf("  big (>=400k) concatenates : %d -> %d   (%.0f MB -> %.0f MB copied)\n",
        off.nbigc, on.nbigc, off.bigc*8/1e6, on.bigc*8/1e6)
@printf("  big dynamic_update_slices : %d -> %d\n", off.nbigd, on.nbigd)
conv = off.bigc == 0 ? 0.0 : 100 * (off.bigc - on.bigc) / off.bigc
@printf("  => %.0f%% of the whole-buffer concatenate traffic was rewritten\n", conv)
say("")
say("  READ: near 100% converted but only 1.12x measured  => the buffer-traffic")
say("        mechanism is WRONG and should be retracted.")
say("        Low conversion                                => the lever is weak;")
say("        the mechanism is still untested and needs a different lever.")
say("="^78)
