# j01 -- is the compiled step a FUNCTION? Same bits in, same bits out?
# Repeats one identical call K times and compares every output bit-for-bit
# against the first. Also reads each result buffer TWICE, which separates
# "the computation is nondeterministic" from "the host read the buffer before
# the device finished writing it".
const J1K = parse(Int, get(ENV, "J1K", "400"))

# Where does a set of state indices sit in the grid? The state is SPECIES-MAJOR
# (13 blocks of NC=288 cells), so a fault confined to one chunk of the cell axis
# shows up as the SAME contiguous cell-offset run inside every species block --
# which is the signature of a partitioned loop over cells.
function report_idx(tag, idx::Vector{Int}, n::Int)
    isempty(idx) && (say("  $tag: none"); return)
    groups = Dict{String,Int}()
    for i in idx; groups[GROUP_OF[i]] = get(groups, GROUP_OF[i], 0) + 1; end
    off = sort(unique(((i - 1) % NC) + 1 for i in idx))
    contig = length(off) > 1 && off[end] - off[1] + 1 == length(off)
    say("  $tag: $(length(idx))/$n entries, $(length(groups)) groups, $(length(off)) cell-offsets")
    say("    groups: " * join(("$k=$v" for (k, v) in sort(collect(groups))), " "))
    say("    cell offsets (contiguous=$contig): " * join(off[1:min(end, 80)], ","))
    cells = Tuple{Int,Int,Int}[]
    S = Set(idx)
    for (vn, vi) in var_map
        vi in S || continue
        m = match(r"\[(\d+),(\d+),(\d+)\]$", vn)
        m === nothing || push!(cells, (parse(Int, m.captures[1]), parse(Int, m.captures[2]),
                                       parse(Int, m.captures[3])))
    end
    uc = sort(unique(cells))
    say("    $(length(uc)) distinct (i,j,k): " * join(uc[1:min(end, 60)], " "))
    flush(stdout)
end

function det_cell(nm, c, TH, dt, u::Vector{Float64}, t::Float64, K::Int; reuse::Bool=false)
    UR = reuse ? RX.ConcreteRArray(copy(u)) : nothing
    TR = reuse ? RX.ConcreteRNumber(t) : nothing
    DR = reuse ? RX.ConcreteRNumber(dt) : nothing
    ref = nothing
    ndiff = 0; nbad = 0; nreread = 0; maxrel = 0.0
    ees = Set{Float64}(); firstidx = Int[]
    t0 = time()
    for k in 1:K
        r = reuse ? c(UR, TH, TR, DR) : callstep(c, u, TH, t, dt)
        out = Array(r[1])
        out2 = Array(r[1])                       # SECOND read of the SAME buffer
        isempty(bitdiff(out, out2)) || (nreread += 1)
        push!(ees, Float64(r[2]))
        if count(!isfinite, out) > 0
            nbad += 1
            say("  !! $nm call $k NON-FINITE load=$(loadavg())")
            report_idx("non-finite", findall(!isfinite, out), length(out))
            for j in 1:3
                o = Array(callstep(c, u, TH, t, dt)[1])
                say("     re-issue $j: nonfinite=$(count(!isfinite, o)) bitdiff_vs_bad=$(length(bitdiff(out, o)))")
            end
        end
        if ref === nothing
            ref = out
        else
            idx = bitdiff(ref, out)
            if !isempty(idx)
                ndiff += 1
                maxrel = max(maxrel, maximum(i -> abs(out[i]-ref[i])/max(abs(ref[i]),1e-300), idx))
                isempty(firstidx) && (firstidx = idx)
            end
        end
    end
    el = time() - t0
    say("  RESULT $nm reuse=$reuse K=$K differing=$ndiff maxrel=$maxrel " *
        "distinctEEst=$(length(ees)) nonfinite=$nbad reread_mismatch=$nreread " *
        "s_per_call=$(round(el/K, digits=5))")
    isempty(firstidx) || report_idx("first bit-differing call", firstidx, length(ref))
    return nothing
end

say("---- j01: determinism, K=$J1K, threads=$(Threads.nthreads()) ----")
for (nm, c, TH, dt) in (("ssp_step", CSSP, THT, DT0T), ("ros_step", CROS, THC, DT0C))
    for (pn, uu, tt) in (("base", UBASE, T0), ("stepped", USTEP, T0 + DT0C))
        det_cell("$nm/$pn", c, TH, dt, uu, tt, J1K; reuse=false)
        det_cell("$nm/$pn", c, TH, dt, uu, tt, J1K; reuse=true)
    end
end
