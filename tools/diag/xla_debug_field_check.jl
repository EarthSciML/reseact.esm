#!/usr/bin/env julia
# ===========================================================================
# xla_debug_field_check.jl -- which XLA:CPU debug options does THIS Reactant
# actually accept?
# ===========================================================================
# README-nondet.md offers `xla_cpu_use_fusion_emitters=false` as an equally good
# alternative to `xla_cpu_prefer_vector_width=128`. That advice was written
# against an older Reactant. This prints the truth for whatever Reactant the
# environment resolves, so the README can be corrected rather than guessed at.
#
#   julia --project=run-model-jl tools/diag/xla_debug_field_check.jl
# ===========================================================================
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV",
                 normpath(joinpath(@__DIR__, "..", "..", "run-model-jl"))); io = devnull)
using Reactant
const RX = Reactant

vers = Pkg.dependencies()
for (_, p) in vers
    p.name in ("Reactant", "Reactant_jll") && println("$(p.name) $(p.version)")
end

const DO = Reactant.Proto.xla.DebugOptions
fs = fieldnames(DO)
println("\nDebugOptions has $(length(fs)) fields; the xla_cpu_* ones:")
foreach(f -> println("  ", f), filter(f -> startswith(String(f), "xla_cpu"), collect(fs)))

resv = try
    Reactant.Proto.PB.reserved_fields(DO).names
catch
    try
        Reactant.ProtoBuf.reserved_fields(DO).names
    catch e
        ["<reserved_fields unavailable: $(sprint(showerror, e))>"]
    end
end
println("\nRESERVED (removed upstream, NOT settable), xla_cpu_* only:")
foreach(n -> println("  ", n), filter(n -> startswith(n, "xla_cpu"), resv))

# NB: `CompileOptions(; xla_debug_options = ...)` validates NOTHING -- it just
# stores the NamedTuple. The name is only resolved later, when
# Reactant.XLA.get_debug_options splats it into a Setfield lens, i.e. at
# @compile time. Testing the constructor would report a false ACCEPTED, so test
# the function that actually consumes it.
for (k, v) in ((:xla_cpu_prefer_vector_width, Int32(128)),
               (:xla_cpu_use_fusion_emitters, false))
    ok = try
        d = Reactant.XLA.get_debug_options(; (; k => v)...)
        "ACCEPTED (field now reads $(getproperty(d, k)))"
    catch e
        # the ArgumentError lists all 316 field names; the first clause is the point
        first(split("REJECTED -- " * first(split(sprint(showerror, e), '\n')), " to object with fields"))
    end
    println("\nget_debug_options(; $k = $v)\n  => $ok")
end
