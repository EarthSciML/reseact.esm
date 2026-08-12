# j03 -- a big soak that CAPTURES every fault in full: which cells, what the
# bad values are, what the same call returns when re-issued, and what the input
# looked like at the faulting cell. Everything else in the output is expected to
# be bit-identical (j01 measured 0 bit-differences in 3200 clean calls), so the
# diff between a bad output and a good one is the whole fault.
const J3N   = parse(Int, get(ENV, "J3N", "20000"))
const J3SSP = parse(Int, get(ENV, "J3SSP", "20000"))

# state index -> (group, cell) using the species-major layout the integrator asserts
const CELL_OF = let v = zeros(Int, N); for i in 1:N; v[i] = ((i - 1) % NC) + 1; end; v end
const IJK_OF = let d = Dict{Int,Tuple{Int,Int,Int}}()
    for (vn, vi) in var_map
        m = match(r"\[(\d+),(\d+),(\d+)\]$", vn)
        m === nothing || (d[vi] = (parse(Int, m.captures[1]), parse(Int, m.captures[2]),
                                   parse(Int, m.captures[3])))
    end
    d
end

isdefined(Main, :FAULTS) || (FAULTS = Any[])


function capture(nm, c, TH, dt, u::Vector{Float64}, t::Float64, K::Int)
    ref = nothing; nbad = 0; ndiff = 0; t0 = time()
    for k in 1:K
        out = Array(callstep(c, u, TH, t, dt)[1])
        bad = findall(!isfinite, out)
        idx = ref === nothing ? Int[] : bitdiff(ref, out)
        if !isempty(bad) || !isempty(idx)
            good = Array(callstep(c, u, TH, t, dt)[1])
            cells = sort(unique(CELL_OF[i] for i in vcat(bad, idx)))
            ijk = sort(unique(get(IJK_OF, i, (0,0,0)) for i in vcat(bad, idx)))
            isempty(bad) || (nbad += 1)
            isempty(idx) || (ndiff += 1)
            say("  !! $nm k=$k nonfinite=$(length(bad)) bitdiff_vs_ref=$(length(idx)) " *
                "cells=$(length(cells)) load=$(loadavg()) t=$(round(time()-t0,digits=1))")
            say("     cell offsets: $(cells)")
            say("     (i,j,k): $(ijk)")
            show_i = sort(unique(vcat(bad, idx)))[1:min(end, 14)]
            for i in show_i
                say("       idx=$i $(GROUP_OF[i])$(get(IJK_OF,i,(0,0,0))) " *
                    "bad=$(out[i]) reissue=$(good[i]) ref=$(ref === nothing ? NaN : ref[i]) u_in=$(u[i])")
            end
            say("     re-issue nonfinite=$(count(!isfinite, good)) bitdiff(reissue,ref)=" *
                "$(ref === nothing ? -1 : length(bitdiff(ref, good)))")
            push!(FAULTS, (; nm, k, bad = copy(bad), idx = copy(idx), cells,
                           t = time(), load = loadavg()))
            flush(stdout)
        end
        ref === nothing && (ref = out)
        k % 2500 == 0 && (say("    $nm $k/$K bad=$nbad differing=$ndiff $(round(time()-t0,digits=1))s"); flush(stdout))
    end
    say("  RESULT capture $nm K=$K nonfinite_calls=$nbad bitdiff_calls=$ndiff " *
        "rate=$(round(nbad/K, sigdigits=3)) s_per_call=$(round((time()-t0)/K, digits=5))")
end

say("---- j03: capture, ros N=$J3N, ssp N=$J3SSP ----")
capture("ros/stepped", CROS, THC, DT0C, USTEP, T0 + DT0C, J3N)
capture("ssp/stepped", CSSP, THT, DT0T, USTEP, T0 + DT0C, J3SSP)
say("  total faults recorded: $(length(FAULTS))")
