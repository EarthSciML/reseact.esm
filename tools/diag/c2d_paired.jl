#!/usr/bin/env julia
# ===========================================================================
# c2d_paired.jl -- CONCATS_TO_DUS A/B: one build, both flag settings,
#                  INTERLEAVED timing, plus the structural op counts.
# ===========================================================================
# Replaces the two-separate-jobs A/B. Everything that matters is measured in ONE
# process off ONE build, and the two arms are timed ALTERNATELY rep by rep, so
# node contention and any mid-run drift hit both arms equally. Reports the
# paired per-rep ratio (median of t_on/t_off over reps), which is the honest
# statistic here, alongside the raw medians.
#
# Answers three things at once:
#   1. the step ratio at THIS grid -- run it at 13x7x72 and it is the decider
#      (the mechanism predicts the ratio must GROW vs the 1.12x seen at NC=2184,
#      because the extended buffer is 10.5 MB at CONUS vs 3.5 MB there);
#   2. whether the bare chem RHS really regresses (measured 7.441 -> 7.938 ms in
#      a cross-job pair, one sample each);
#   3. whether the `concat_to_onedim_dus` pattern FIRED at all -- if the big
#      concatenates survive, the 1.12x says nothing about the traffic claim.
#
#   RESEACT_NLON/NLAT/NLEV   grid (default 13/7/72)
#   C2D_REPS                 interleaved reps per arm (default 30)
#   C2D_STRUCT               1 to also emit the op counts (default 1)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "13"); get!(ENV, "RESEACT_NLAT", "7"); get!(ENV, "RESEACT_NLEV", "72")
get!(ENV, "RESEACT_ADJ_UJITTER", "0")
ENV["RESEACT_ADJ_STAGES"] = "none"
ENV["RESEACT_LABEL"] = "c2dpaired"

include(joinpath(@__DIR__, "_env.jl"))
Base.include(Core.eval(Main, :(module _Drv end)), joinpath(REPO, "tools", "adjoint_gradient.jl"))
const D = Main._Drv
using Printf, Statistics
RX = D.RX
say(s) = (println(s); flush(stdout))

const REPS = parse(Int, get(ENV, "C2D_REPS", "30"))
const DOSTRUCT = get(ENV, "C2D_STRUCT", "1") == "1"

UD = RX.ConcreteRArray(copy(D.UBASE))
TD = RX.ConcreteRNumber(D.T0)
DD = RX.ConcreteRNumber(D.DT0C)
rhs_only(uu, th, tt) = D.gC(uu, th, tt)

say("\n" * "="^78)
say(@sprintf("CONCATS_TO_DUS PAIRED A/B   NS=%d NC=%d N=%d  reps=%d  threads=%d",
             D.NS, D.NC, D.NS * D.NC, REPS, Threads.nthreads()))
say("="^78)

# ---- structural counts, both settings, before any timing -------------------
function struct_counts(label)
    txt = sprint(show, RX.@code_hlo(D.ros_step(UD, D.THC, TD, DD)))
    function big(op)
        s = Int[]
        for ln in split(txt, '\n')
            occursin(op, ln) || continue
            m = match(r"->\s*tensor<([0-9x]+)x[a-z]", ln)
            m === nothing && continue
            n = 1; for d in split(m.captures[1], 'x'); n *= parse(Int, d); end
            n >= 400_000 && push!(s, n)
        end
        return s
    end
    c = big("stablehlo.concatenate"); u = big("stablehlo.dynamic_update_slice")
    @printf("  %-26s big concat %4d (%6.0f MB)   big DUS %4d (%6.0f MB)\n",
            label, length(c), sum(c; init=0)*8/1e6, length(u), sum(u; init=0)*8/1e6)
    return (nc = length(c), mb = sum(c; init=0)*8/1e6)
end

if DOSTRUCT
    say("\n---- structure of the module Reactant hands to XLA ----")
    try
        RX.Compiler.CONCATS_TO_DUS[] = false; s_off = struct_counts("CONCATS_TO_DUS=false")
        RX.Compiler.CONCATS_TO_DUS[] = true;  s_on  = struct_counts("CONCATS_TO_DUS=true")
        conv = s_off.mb == 0 ? 0.0 : 100 * (s_off.mb - s_on.mb) / s_off.mb
        @printf("  => %.0f%% of the whole-buffer concatenate traffic was rewritten to DUS\n", conv)
    catch e
        say("  structural pass FAILED: " * first(split(sprint(showerror, e), '\n')))
    end
end

# ---- compile both arms -----------------------------------------------------
say("\n---- compiling both arms ----")
RX.Compiler.CONCATS_TO_DUS[] = false
t0 = time(); STEP_OFF = RX.@compile sync=true D.ros_step(UD, D.THC, TD, DD)
@printf("  step  OFF compiled %6.1f s\n", time() - t0); flush(stdout)
t0 = time(); RHS_OFF = RX.@compile sync=true rhs_only(UD, D.THC, TD)
@printf("  rhs   OFF compiled %6.1f s\n", time() - t0); flush(stdout)
RX.Compiler.CONCATS_TO_DUS[] = true
t0 = time(); STEP_ON = RX.@compile sync=true D.ros_step(UD, D.THC, TD, DD)
@printf("  step  ON  compiled %6.1f s\n", time() - t0); flush(stdout)
t0 = time(); RHS_ON = RX.@compile sync=true rhs_only(UD, D.THC, TD)
@printf("  rhs   ON  compiled %6.1f s\n", time() - t0); flush(stdout)
RX.Compiler.CONCATS_TO_DUS[] = false

# ---- interleaved paired timing --------------------------------------------
function paired(name, foff, fon, reps)
    foff(); fon()                                    # warm both
    toff = Float64[]; ton = Float64[]; ratio = Float64[]
    for i in 1:reps
        t = time_ns(); foff(); a = (time_ns() - t) / 1e6
        t = time_ns(); fon();  b = (time_ns() - t) / 1e6
        push!(toff, a); push!(ton, b); push!(ratio, a / b)
    end
    say("")
    @printf("  %s\n", name)
    @printf("    OFF   median %8.3f ms   min %8.3f\n", median(toff), minimum(toff))
    @printf("    ON    median %8.3f ms   min %8.3f\n", median(ton),  minimum(ton))
    @printf("    paired speedup OFF/ON: median %.3fx   (min-based %.3fx)\n",
            median(ratio), minimum(toff) / minimum(ton))
    return median(ratio)
end

say("\n---- interleaved paired timing ----")
rs = paired("ROS23 step", () -> STEP_OFF(UD, D.THC, TD, DD), () -> STEP_ON(UD, D.THC, TD, DD), REPS)
rr = paired("bare chemistry RHS", () -> RHS_OFF(UD, D.THC, TD), () -> RHS_ON(UD, D.THC, TD), REPS)

say("")
say("="^78)
@printf("  NC=%d   step %.3fx   RHS %.3fx   (>1 means CONCATS_TO_DUS is faster)\n", D.NC, rs, rr)
say("  NC=2184 cross-job reference: step 1.12x, RHS 0.94x")
say("="^78)
say("C2D_PAIRED_DONE")
