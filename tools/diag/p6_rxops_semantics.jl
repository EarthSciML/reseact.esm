#!/usr/bin/env julia
# p6_rxops_semantics.jl -- do Reactant's reverse / broadcast_to_size /
# permutedims / mixed Int-range getindex agree with Base on a 3-D array?
# The strided-box emitter (EarthSciAST ESS_OOP_SHIFT_SLICE) composes exactly
# these; its host emulation is exact, so a CONUS mismatch must be one of them.
import Pkg; Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(@__DIR__, "..", "..", "run-model-jl")); io = devnull)
using Reactant
RX = Reactant
A = reshape(collect(1.0:(4*5*6)), 4, 5, 6)
Ad = RX.ConcreteRArray(A)
f_rev(x)  = RX.Ops.reverse(x; dimensions = [2])
f_rev13(x) = RX.Ops.reverse(x; dimensions = [1, 3])
f_bc(x)   = RX.broadcast_to_size(reshape(x[2, 1:3, 2:5], (1, 3, 4)), (3, 3, 4))
f_pd(x)   = permutedims(x, (3, 1, 2))
f_ix(x)   = x[3, 2:4, 1:5]
f_rs(x)   = reshape(x[:], (2, 10, 6))[1, 2:5, 3:6]
checks = [("reverse dim2", f_rev, reverse(A; dims = 2)),
          ("reverse dims13", f_rev13, reverse(A; dims = (1, 3))),
          ("broadcast_to_size", f_bc, reshape(A[2, 1:3, 2:5], (1, 3, 4)) .+ zeros(3, 3, 4)),
          ("permutedims", f_pd, permutedims(A, (3, 1, 2))),
          ("int+ranges getindex", f_ix, A[3, 2:4, 1:5]),
          ("reshape+slice", f_rs, reshape(A[:], (2, 10, 6))[1, 2:5, 3:6])]
for (name, f, ref) in checks
    c = RX.@compile compile_options=RX.CompileOptions(; sync = true) f(Ad)
    got = Array(c(Ad))
    println("P6OPS ", rpad(name, 22), size(got) == size(ref) && got == ref ? "OK" : "MISMATCH size=$(size(got)) vs $(size(ref))")
end
println("P6OPS_DONE")
