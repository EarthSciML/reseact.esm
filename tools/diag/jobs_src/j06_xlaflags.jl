# j06 -- THE DISCRIMINATOR. Recompile the faulting object (the chemistry RHS)
# under different XLA/Reactant compile options and measure the fault rate of
# each. Reactant threads a `xla_debug_options` NamedTuple straight into the
# executable's DebugOptions proto, so this needs no rebuild -- every variant is
# a ~6 s compile in the process that is already up.
#
# The hypothesis under test is threading: XLA:CPU has no documented determinism
# guarantee, `--xla_cpu_multi_thread_eigen` is the documented switch for its
# multi-threaded Eigen mode, and the fault rate here rises and falls with the
# machine's load average. If the rate goes to zero with multi-threading off and
# stays put under the other knobs, that is the answer.
const J6N = parse(Int, get(ENV, "J6N", "40000"))

# what CPU-side knobs does this XLA actually have?
let dbg = RX.XLA.get_default_debug_options()
    fs = [f for f in fieldnames(typeof(dbg)) if occursin("cpu", String(f))]
    say("  DebugOptions cpu fields: " * join(fs, " "))
    for f in fs
        say("    $f = $(getfield(dbg, f))")
    end
end

callrhs6(c, TH, u, t) = Array(c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t)))

function rate(nm, c, TH, u, t, K)
    ref = callrhs6(c, TH, u, t)
    nbad = 0; lanes = Set{Int}(); t0 = time(); l0 = loadavg()
    for k in 1:K
        out = callrhs6(c, TH, u, t)
        b = findall(i -> !(out[i] === ref[i]), eachindex(out))
        isempty(b) && continue
        nbad += 1
        for i in b; push!(lanes, ((i - 1) % NC) + 1); end
    end
    el = time() - t0
    say("  RATE $nm  K=$K faults=$nbad rate=$(round(nbad/K, sigdigits=3)) " *
        "lanes=$(sort(collect(lanes))) load=$l0..$(loadavg()) s_per_call=$(round(el/K, digits=6))")
    flush(stdout)
    return nbad
end

const VARIANTS = Any[
    ("baseline",              () -> RX.@compile sync=true rhs_only(U_R, THC, T_R)),
    ("eigen_mt_off",          () -> RX.@compile sync=true xla_debug_options=(; xla_cpu_multi_thread_eigen=false) rhs_only(U_R, THC, T_R)),
    ("ftz_off",               () -> RX.@compile sync=true xla_debug_options=(; xla_cpu_ftz=false) rhs_only(U_R, THC, T_R)),
    ("backend_opt0",          () -> RX.@compile sync=true xla_debug_options=(; xla_backend_optimization_level=0) rhs_only(U_R, THC, T_R)),
    ("no_reactant_passes",    () -> RX.@compile sync=true optimize=false rhs_only(U_R, THC, T_R)),
    ("async",                 () -> RX.@compile sync=false rhs_only(U_R, THC, T_R)),
    ("no_donate",             () -> RX.@compile sync=true donated_args=:none rhs_only(U_R, THC, T_R)),
]

say("---- j06: fault rate vs compile options, N=$J6N each ----")
# interleave two passes so a drifting machine load cannot fake a difference
for pass in 1:2
    say("  == pass $pass ==")
    for (nm, mk) in VARIANTS
        c = try
            timed_compile("v:$nm", mk)
        catch e
            say("  $nm COMPILE FAILED: $(sprint(showerror, e))"); continue
        end
        rate("$nm/p$pass", c, THC, USTEP, T0 + DT0C, J6N)
    end
end
