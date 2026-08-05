# Cell-major permutation + 13-color block-diagonal FD Jacobian for the chem part.
#
# EarthSciAST lays the state out SPECIES-MAJOR ([CH2O×NC, CH3O2×NC, …, m×NC]), so
# the chemistry Jacobian (each cell's species couple only among themselves) is
# block-diagonal only after permuting to CELL-MAJOR ([cell1: NS states, cell2: …]).
# We run the whole split problem in cell-major order so EarthSciMLBase's
# `BlockDiagonal` (contiguous NS×NS blocks) applies directly.
#
# The block FD Jacobian exploits cell independence: perturbing species s in EVERY
# cell at once (cells don't couple in chemistry) fills column s of every block in a
# single RHS eval — NS colors => NS+1 chem RHS evals per Jacobian, not N.
using .BlockDiag
using LinearAlgebra
using ForwardDiff

# Build the species-major <-> cell-major permutation purely from the state names.
# Returns sm_of_cm[cm] = species-major index at cell-major position cm (and inverse).
function cellmajor_perm(var_map)
    function parse_name(nm)
        m = match(r"^(.*)\[(\d+),(\d+),(\d+)\]$", nm)
        m === nothing && error("cellmajor_perm: unparseable state name '$nm'")
        (String(m.captures[1]),
         (parse(Int, m.captures[2]), parse(Int, m.captures[3]), parse(Int, m.captures[4])))
    end
    N = length(var_map)
    parsed = Dict(nm => parse_name(nm) for nm in keys(var_map))
    bases = sort(unique(v[1] for v in values(parsed)))
    cells = sort(unique(v[2] for v in values(parsed)))
    NS = length(bases); NC = length(cells)
    NS * NC == N || error("cellmajor_perm: NS*NC=$(NS*NC) != N=$N (non-rectangular lift?)")
    base_pos = Dict(b => k for (k, b) in enumerate(bases))
    cell_pos = Dict(c => k for (k, c) in enumerate(cells))
    sm_of_cm = Vector{Int}(undef, N)
    for (nm, sm) in var_map
        b, c = parsed[nm]
        sm_of_cm[(cell_pos[c]-1)*NS + base_pos[b]] = sm
    end
    sort(sm_of_cm) == collect(1:N) || error("cellmajor_perm: permutation is not a bijection")
    return (; sm_of_cm, cm_of_sm = invperm(sm_of_cm), NS, NC, N, bases, base_pos, cells, cell_pos)
end

# Wrap a species-major in-place RHS `f_sm!` as a cell-major in-place RHS (private
# scratch buffers; serial-solve only).
function cellmajor_rhs(f_sm!, sm_of_cm)
    N = length(sm_of_cm)
    usm = Vector{Float64}(undef, N); dusm = Vector{Float64}(undef, N)
    return function (du_cm, u_cm, p, t)
        @inbounds for i in 1:N; usm[sm_of_cm[i]] = u_cm[i]; end
        f_sm!(dusm, usm, p, t)
        @inbounds for i in 1:N; du_cm[i] = dusm[sm_of_cm[i]]; end
        return nothing
    end
end

# NS-color block-diagonal FD Jacobian for a cell-major pointwise RHS `f_cm!`.
# Returns (jac_cm!(J,u,p,t) filling a BlockDiagonal, mkjp() building the prototype).
function block_fd_jac(f_cm!, NS, NC)
    N = NS * NC
    du0 = Vector{Float64}(undef, N); up = Vector{Float64}(undef, N); dup = Vector{Float64}(undef, N)
    hbuf = Vector{Float64}(undef, NC)
    function jac_cm!(J, u_cm, p, t)
        data = J.data
        f_cm!(du0, u_cm, p, t)
        @inbounds for s in 1:NS
            copyto!(up, u_cm)
            for c in 1:NC
                idx = (c-1)*NS + s
                h = sqrt(eps()) * max(abs(u_cm[idx]), 1e-9)
                up[idx] += h; hbuf[c] = h
            end
            f_cm!(dup, up, p, t)
            for c in 1:NC
                base = (c-1)*NS; h = hbuf[c]
                for r in 1:NS
                    data[r, s, c] = (dup[base+r] - du0[base+r]) / h
                end
            end
        end
        return nothing
    end
    mkjp() = BlockDiagonal(zeros(NS, NS, NC), MapBroadcast())
    return jac_cm!, mkjp
end

# EXACT block-diagonal Jacobian by forward-mode AD. Same signature as
# `block_fd_jac`, so it is a drop-in replacement.
#
# WHY THIS EXISTS. The FD version above has no defensible step size for this
# model. `h = sqrt(eps)*max(|u|, 1e-9)` carries an ABSOLUTE floor equal to the
# solver's ATOL, and species here span thirty orders: NO legitimately sits at
# 1e-26 at night (titrated by NO + O3 -> NO2 with no photolysis to return it),
# OH lives at 1e-13. At NO = 1e-26 that floor makes h ~ 1e-17, a perturbation
# 1e9 times the value itself -- a secant across nine decades, not a tangent.
# Shrinking it does not help either: a purely relative step makes dup - du0
# underflow to exactly zero against an RHS of O(1), which is how the earlier
# A/B measured the FD Jacobian SILENTLY LOSING 22% of its nonzeros. There is no
# good h, which is the argument for not needing one.
#
# WHY IT IS CHEAP. Chemistry couples species only WITHIN a cell, so seeding each
# cell's NS species with the NS unit partials makes every cell's block appear in
# ONE evaluation: NS partials per Dual, 1 RHS eval, versus NS+1 for the coloring
# above. The block structure is what makes AD affordable -- a dense AD Jacobian
# of the whole state would need N partials, not NS.
#
# The permutation is redone here over Dual buffers rather than reusing
# `cellmajor_rhs`, whose scratch is concretely Vector{Float64}; keeping that path
# monomorphic leaves the hot Float64 RHS untouched.
function block_ad_jac(f_sm!, sm_of_cm, NS, NC)
    N = NS * NC
    N == length(sm_of_cm) || error("block_ad_jac: NS*NC=$N != length(sm_of_cm)=$(length(sm_of_cm))")
    D = ForwardDiff.Dual{typeof(f_sm!),Float64,NS}
    seeds = ntuple(s -> ForwardDiff.Partials(ntuple(k -> k == s ? 1.0 : 0.0, NS)), NS)
    usm = Vector{D}(undef, N); dusm = Vector{D}(undef, N)
    function jac_cm!(J, u_cm, p, t)
        data = J.data
        @inbounds for c in 1:NC
            base = (c - 1) * NS
            for s in 1:NS
                usm[sm_of_cm[base+s]] = D(u_cm[base+s], seeds[s])
            end
        end
        f_sm!(dusm, usm, p, t)
        @inbounds for c in 1:NC
            base = (c - 1) * NS
            for r in 1:NS
                pr = ForwardDiff.partials(dusm[sm_of_cm[base+r]])
                for s in 1:NS
                    data[r, s, c] = pr[s]
                end
            end
        end
        return nothing
    end
    mkjp() = BlockDiagonal(zeros(NS, NS, NC), MapBroadcast())
    return jac_cm!, mkjp
end
