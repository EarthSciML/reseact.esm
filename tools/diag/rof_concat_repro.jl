#!/usr/bin/env julia
# ===========================================================================
# rof_concat_repro.jl -- MINIMAL reproducer for upstream bug #2:
# `concat_broadcast_slice` builds a malformed `stablehlo.concatenate`.
# ===========================================================================
# No Enzyme, no autodiff, no model. `vcat` of N row-vectors, each of which is a
# reshape of a contiguous slice of ONE source vector, is rewritten by the
# `concat_broadcast_slice` pattern into a concatenate whose merged operand has
# the wrong shape, and the module then fails its own verifier.
#
#   julia --project=$RESEACT_RXENV tools/diag/rof_concat_repro.jl [NBLK] [BLK]
#
# Expected: FAIL for NBLK >= 3 (the pattern merges runs of >= 2 adjacent
# operands, and a merge of 2 length-1 slices is explicitly declined), OK when
# `concat_broadcast_slice` is excluded from the pipeline.
import Pkg
const REPO = dirname(dirname(@__DIR__))
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Reactant, Printf
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

const NBLK = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 13
const BLK  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
const N    = NBLK * BLK

# rows[k] = reshape(x[k*BLK+1 : (k+1)*BLK], 1, BLK)   ->  concatenate(dim=0)
rowstack(x) = vcat([reshape(x[(k * BLK + 1):((k + 1) * BLK)], 1, BLK) for k in 0:(NBLK - 1)]...)

X = RX.ConcreteRArray(collect(1.0:N))
println("=== concat repro: $NBLK blocks x $BLK  (N=$N) ===")
for (nm, opts) in (("default            ", RX.CompileOptions()),
                   ("no concat_broadcast_slice", RX.CompileOptions(; excluded_passes = ["concat_broadcast_slice"])))
    try
        c = @compile compile_options = opts rowstack(X)
        ok = Array(c(X)) == reshape(collect(1.0:N), BLK, NBLK)'
        @printf("  %-26s OK   (values %s)\n", nm, ok ? "correct" : "WRONG")
    catch e
        msg = sprint(showerror, e)
        m = match(r"error: [^\n]*", msg)
        @printf("  %-26s FAIL %s\n", nm, m === nothing ? first(msg, 160) : m.match)
    end
    flush(stdout)
end
