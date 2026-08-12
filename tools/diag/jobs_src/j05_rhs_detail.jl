# j05 -- the chemistry RHS is the faulting object (j04A). Characterise it.
#   * WHICH states go bad, what the bad and good values are.
#   * Is it always lane 1? Does it depend on u, on t, on the forcing buffers?
#   * Rate vs wall time and machine load.
#   * Dump the HLO (both stages) so the emitted program can be read offline.
const J5N = parse(Int, get(ENV, "J5N", "30000"))
const HLODIR = joinpath(@__DIR__, "..", "hlo")
mkpath(HLODIR)

const CELLOF5 = [((i - 1) % NC) + 1 for i in 1:N]
const IJK5 = let d = Dict{Int,Tuple{Int,Int,Int}}()
    for (vn, vi) in var_map
        m = match(r"\[(\d+),(\d+),(\d+)\]$", vn)
        m === nothing || (d[vi] = (parse(Int,m.captures[1]), parse(Int,m.captures[2]), parse(Int,m.captures[3])))
    end; d
end

callrhs(c, TH, u, t) = Array(c(RX.ConcreteRArray(u), TH, RX.ConcreteRNumber(t)))

"""Soak the RHS at one point; return the multiset of faulting state indices."""
function rhs_detail(nm, c, TH, u, t, K; verbose_first = 5)

    ref = callrhs(c, TH, u, t)
    hits = Dict{Int,Int}(); nbad = 0; times = Float64[]; loads = Float64[]
    t0 = time()
    for k in 1:K
        out = callrhs(c, TH, u, t)
        b = findall(i -> !(out[i] === ref[i]), eachindex(out))
        isempty(b) && continue
        nbad += 1
        for i in b; hits[i] = get(hits, i, 0) + 1; end
        push!(times, time() - t0); push!(loads, loadavg())
        if nbad <= verbose_first
            say("  !! $nm k=$k  $(length(b)) deviating entries, lanes=$(sort(unique(CELLOF5[i] for i in b)))")
            for i in b
                say("     idx=$i $(GROUP_OF[i])$(get(IJK5,i,(0,0,0)))  bad=$(out[i])  good=$(ref[i])  u=$(u[i])")
            end
            flush(stdout)
        end
    end
    el = time() - t0
    say("  RESULT rhs_detail $nm K=$K faulting_calls=$nbad rate=$(round(nbad/K, sigdigits=3)) " *
        "s_per_call=$(round(el/K, digits=6))")
    if !isempty(hits)
        say("    faulting states (idx => count): " *
            join(("$(GROUP_OF[i])$(get(IJK5,i,(0,0,0)))[$i]=>$c" for (i,c) in sort(collect(hits))), " "))
        say("    lanes touched: $(sort(unique(CELLOF5[i] for i in keys(hits))))")
        # is the fault clustered in time?
        if length(times) > 3
            q = [count(<=(f*el), times) for f in (0.25, 0.5, 0.75, 1.0)]
            say("    cumulative faults by wall quartile: $q ; load range $(round(minimum(loads),digits=1))-$(round(maximum(loads),digits=1))")
        end
    end
    return hits
end

say("---- j05: chemistry RHS detail, N=$J5N ----")
rhs_detail("chem/USTEP", CRHS, THC, USTEP, T0 + DT0C, J5N)
rhs_detail("chem/UBASE", CRHS, THC, UBASE, T0, J5N)
let ur = UBASE .* (1 .+ 0.3 .* randn(Random.MersenneTwister(7), N))
    rhs_detail("chem/URAND", CRHS, THC, ur, T0, J5N)
end
rhs_detail("chem/USTEP@t+1234", CRHS, THC, USTEP, T0 + 1234.0, J5N)
rhs_detail("tran/USTEP", CRHST, THT, USTEP, T0 + DT0C, J5N)

say("---- dumping HLO to $HLODIR ----")
for (nm, ex) in (("rhs_chem_opt", () -> RX.@code_hlo optimize=true rhs_only(U_R, THC, T_R)),
                 ("rhs_chem_raw", () -> RX.@code_hlo optimize=false rhs_only(U_R, THC, T_R)))
    try
        s = sprint(show, ex())
        open(joinpath(HLODIR, nm * ".mlir"), "w") do io; write(io, s); end
        say("  wrote $nm.mlir  $(length(s)) bytes, $(count(==('\n'), s)) lines")
    catch e
        say("  $nm FAILED: $(sprint(showerror, e))")
    end
end
