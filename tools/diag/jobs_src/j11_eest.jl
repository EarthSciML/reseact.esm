# j11 -- does a fault make EEst NaN? That is the link to FINDING 1: host_adaptive!
# maps isnan(EEst) -> EEst = 1e10 -> REJECT, so one fault inside a macro step
# adds a rejection and shifts the whole subsequent dt sequence, which is exactly
# the "95 accepts / 2 rejects vs 92 / 1 on a replay of the same checkpoint"
# signature FINDING 1 attributed to ULP-level nondeterminism.
const J11N = parse(Int, get(ENV, "J11N", "40000"))
say("---- j11: EEst on a faulting ros23 step, N=$J11N ----")
let ref = nothing, refee = 0.0, n = 0, nee = 0
    for k in 1:J11N
        r = CROS(RX.ConcreteRArray(USTEP), THC, RX.ConcreteRNumber(T0 + DT0C), RX.ConcreteRNumber(DT0C))
        out = Array(r[1]); ee = Float64(r[2])
        if ref === nothing
            ref = out; refee = ee; continue
        end
        if any(i -> !(out[i] === ref[i]), eachindex(out)) || !(ee === refee)
            n += 1
            isfinite(ee) || (nee += 1)
            n <= 6 && say("  fault $n: nonfinite_u=$(count(!isfinite,out)) EEst=$ee (clean EEst=$refee) " *
                          "-> host_adaptive! would $(isnan(ee) ? "REJECT (EEst:=1e10)" : "accept")")
        end
    end
    say("  RESULT11 K=$J11N faults=$n with_nonfinite_EEst=$nee clean_EEst=$refee")
end
