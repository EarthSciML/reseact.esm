#!/usr/bin/env julia
# ===========================================================================
# c2d_driver.jl -- profile_adjoint.jl with Reactant's concat->dynamic_update_slice
#                  rewrite forced ON (or OFF), to A/B the whole-buffer copies.
# ===========================================================================
# The CONUS HLO dump (logs/hlodump-10105946) shows the ros23 step's extended
# observed buffer (f64[1310641] = 200 quantities/cell vs a 85176 state) being
# functionally updated by 67 WHOLE-BUFFER `concatenate`s -- e.g.
# `concatenate(slice[0:596232], newvals[471744], slice[1067976:1310641])`, a full
# 10.5 MB copy to place 471744 elements. Those concatenates are ALREADY in
# `before_optimizations.txt`, i.e. Reactant hands them to XLA, and a concatenate
# can never be done in place: buffer assignment gives each one a NEW offset.
# The writes that Reactant spells as `scatter` instead ARE aliased in place (36
# of them share their input's buffer offset in this very module).
#
# `Reactant.Compiler.CONCATS_TO_DUS[]` (default false) enables the
# `concat_to_onedim_dus` MLIR pattern, rewriting exactly this shape into
# `stablehlo.dynamic_update_slice`, which XLA:CPU CAN alias in place.
#
# PREDICTION: the concatenate family is 827 MB of the step's 1530 MB of
# whole-buffer copies; if the step is bound by that traffic, turning this on
# should take a visible bite out of the step time.
# FALSIFIED if the step time is unchanged (or if the op counts show the pattern
# never matched -- check the emitted concatenate count).
#
#   AB_C2D=1|0   set Reactant.Compiler.CONCATS_TO_DUS[]
# ===========================================================================
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV",
                 normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io = devnull)
using Reactant
Reactant.Compiler.CONCATS_TO_DUS[] = (get(ENV, "AB_C2D", "0") == "1")
println("=== Reactant.Compiler.CONCATS_TO_DUS[] = ",
        Reactant.Compiler.CONCATS_TO_DUS[], " ===")
flush(stdout)
include(joinpath(@__DIR__, "profile_adjoint.jl"))
