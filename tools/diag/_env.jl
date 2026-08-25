# Shared environment bootstrap for the diagnostic probes: activate the ReSEACT
# Reactant environment and bring `EarthSciAST` into scope. Included by the
# probes that do not activate an environment themselves.
#
# This file REPLACES the former `_env.jl`, which additionally forwarded
# `EarthSciAST.load_path` to `load` for the window when reseact.esm a98b4d0
# ("chore: follow EarthSciAST's load split") had moved the call sites but the
# EarthSciAST commit introducing `load_path` (9ff93e509) was not yet on main.
# That window is CLOSED -- `load_path` is defined at src/resolve.jl:69 and
# exported from src/EarthSciAST.jl:208 -- so the forwarding is gone.
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV",
                 normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io = devnull)
using EarthSciAST
