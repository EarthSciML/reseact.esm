# ===========================================================================
# capacity_chem.jl -- a FIXED-LANE-CAPACITY build of the chemistry half.
# ===========================================================================
# WHY. The chemistry half is exactly block-diagonal, so a per-cell (rather than
# per-domain) step size would cut the chemistry step count ~7.5x at CONUS. That
# only pays if the cells that are NOT being stepped are REMOVED from the vector
# -- a masked SIMD lane costs exactly what a busy one does. Removing them means
# evaluating the pointwise RHS on a COMPACTED SUB-BATCH of C cells, and the
# built evaluator's slot/scatter vectors are computed for the full NC. So the
# sub-batch needs its own build, at lane capacity C.
#
# THE OBSTACLE, and it is the whole content of this file. Chemistry is pointwise
# in the STATE but NOT in its INPUTS: it re-derives per-cell geometry from the
# GRID INDEX. Measured on the split part 2 (tools/diag/capC_struct_probe.jl):
# of the 310 names the chemistry RHS transitively needs, 37 read a raw loop
# index, and they fall into exactly two classes.
#
#   * CONTENT-CONTROLLED (no surgery). Ten of them are reads of a GEOS-FP native
#     forcing array at an AFFINE index -- `GEOSFP.T[r, gk, LAT0+gj, LON0+gi]`
#     and friends. The array is a `param_arrays` buffer WE fill, and its shape
#     is whatever we pass, so a gathered lane value can simply be written into
#     the slot the lane will read. Nothing in the document changes. The time
#     dependence stays exact for free: the interpolation weights `w_time_*` are
#     SCALARS, so `Tc(t) = (1-w(t))*T[1,lane] + w(t)*T[2,lane]` is still a
#     function of t after the gather.
#   * INDEX-DERIVED (surgery, 8 sites). `Pc` reads the hybrid coefficients at
#     `Ap[gk]`/`Ap[gk+1]`; `cos_sza_c` reads `latp[gj]`/`lonp[gi]`; `levc` IS
#     `gk`; five emission terms gate on `gk < 2`; `NEIRegrid.lonc` is
#     `lon0_deg + (gi-1/2)*dlon_deg`. None of these can be reached through a
#     buffer, because the index itself carries the meaning.
#
# MEASURED (2026-08-24, reference grid 6x6x8, NC = 288). The capacity RHS
# reproduces the reference chemistry RHS to 2.888e-14 relative, CELL FOR CELL,
# under a random cell permutation; identical under the identity map and with 37
# padding lanes appended; a padding lane's inputs cannot move a real lane at all
# (max |Δdu| exactly 0); and the same check reports 7.6e10 when two lanes'
# inputs are deliberately swapped, so it has teeth. Costs, against the
# reference build of the SAME chemistry half:
#
#     reference build (NC = 288)             274.6 s
#     capacity build  (C   = 288)              4.6 s
#     capacity build  (C   = 4096)            11.4 s   + prepare_jacobian 46.5 s
#                                                        (structure=block_diagonal)
#     CONUS reference (NC = 6552)            650.2 s   + prepare_jacobian 567.8 s
#
# The capacity build's cost is a function of C alone, so it is the same for
# CONUS and for 4x CONUS -- and at C = 4096 it is ~20x cheaper than the full
# CONUS build it replaces.
#
# ONE CAPACITY IS NOT ENOUGH, and the scheduler must know it. A dyadic level j
# costs `2^j * (lanes run)`, and the SPARSE HIGH levels are where the money is:
# at CONUS, levels 6-8 hold 0.5% of the cell-windows but, rounded up to a single
# chunk of C = 1024, they cost more than half the total -- the whole 5.14x
# dyadic saving is spent on padding and the scheme comes out at 0.75x, i.e.
# SLOWER than the global controller. With a LADDER of capacities (e.g.
# 8/32/128/512/2048), each level covered by the largest rung that fits and one
# rung for the remainder, padding waste falls to 2.7% and the realised saving is
# 5.00x. A ladder is affordable only because a capacity build is seconds; that
# is why the two facts belong in the same file.
#
# THE LANE GRID. Make the capacity grid `C x 1 x 1`: lat and lev extent 1, so a
# lane IS `gi` and EVERY array shaped [lon], [lon,lat] or [lon,lat,lev] is
# automatically per-lane. That is what collapses the surgery to 8 sites -- at
# any lat/lev extent > 1 the whole [lon,lat]-shaped tier (PS, the emission
# factors, the A1 surface fields) would be shared down a column and would have
# to be promoted to 3-D as well. `(gj-1)*NLON + gi`, the one NLON-baked literal
# in the chemistry closure, is index-correct for free because `gj-1` is 0.
#
# The document is LOADED at a legal grid (NLEV >= 6, or the transport half's
# vertical stencil regions invert) and the index-set sizes are rewritten
# afterwards. That is sound because NLEV enters the chemistry closure ONLY
# through index-set sizes -- verified by diffing the split part-2 document at
# NLEV=8 against NLEV=6: every difference is an index-set size, one ppmflux
# makearray region, or a ppmflux template bound, and all of those are transport.
#
# WHAT ELSE FALLS OUT. `NEIRegrid.E_*` is the conservative regrid of the NEI
# inventory -- 137,241 source cells, const-folded at BUILD time, and the reason
# `build_evaluator` costs 725 s at CONUS. The capacity build needs those values
# GATHERED per lane, so it takes them as a runtime buffer instead, and the whole
# regrid subtree prunes away. Combined with pruning to the reachable closure,
# the capacity build's cost is a function of C alone, not of the real grid.
# ===========================================================================
module CapacityChem

using Printf

const LANE_VARS = ("Cap.apc", "Cap.bpc", "Cap.lat", "Cap.lon", "Cap.lev")
const EMIS_E    = ("NEIRegrid.E_CO", "NEIRegrid.E_NO", "NEIRegrid.E_NO2",
                   "NEIRegrid.E_ISOP", "NEIRegrid.E_FORM")
const EMIS_EQS  = ("NEIRegrid.CO_emis", "NEIRegrid.NO_emis", "NEIRegrid.NO2_emis",
                   "NEIRegrid.ISOP_emis", "NEIRegrid.CH2O_emis")

# --------------------------------------------------------------------------- #
# raw-document helpers. `EA.flattened_to_esm` hands back plain Dict/Vector data
# (that is what makes it JSON), so the walk is over `AbstractDict` nodes with
# "op"/"args", plus "expr" (aggregate body) and "bindings" (template arguments).
# --------------------------------------------------------------------------- #
lhs_name(eq) = (l = get(eq, "lhs", nothing); l isa AbstractString ? l : nothing)

function collect_names!(acc::Set{String}, e)
    if e isa AbstractString
        push!(acc, e)
    elseif e isa AbstractDict
        haskey(e, "args")      && collect_names!(acc, e["args"])
        haskey(e, "expr")      && collect_names!(acc, e["expr"])
        haskey(e, "expr_body") && collect_names!(acc, e["expr_body"])
        haskey(e, "body")      && collect_names!(acc, e["body"])
        if haskey(e, "bindings") && e["bindings"] isa AbstractDict
            for (_, v) in e["bindings"]; collect_names!(acc, v); end
        end
    elseif e isa AbstractVector
        for a in e; collect_names!(acc, a); end
    end
    return acc
end
names_of(e) = collect_names!(Set{String}(), e)

"index(name, idx...) as a raw node"
ix(name, idxs...) = Dict{String,Any}("op" => "index",
                                     "args" => Any[name, idxs...])

"Depth-first rewrite: `f(node)` returns a replacement or `nothing` to descend."
function rewrite!(e, f)
    r = f(e)
    r === nothing || return r
    if e isa AbstractDict
        for k in ("args", "expr", "expr_body", "body")
            haskey(e, k) && (e[k] = rewrite!(e[k], f))
        end
        if haskey(e, "bindings") && e["bindings"] isa AbstractDict
            for (bk, bv) in e["bindings"]; e["bindings"][bk] = rewrite!(bv, f); end
        end
        return e
    elseif e isa AbstractVector
        for i in eachindex(e); e[i] = rewrite!(e[i], f); end
        return e
    end
    return e
end

"Replace `index(from, fidx)` with `index(to, tidx)` anywhere below `e`."
function swap_index!(e, from::AbstractString, fidx::AbstractString,
                        to::AbstractString, tidx::AbstractString)
    n = Ref(0)
    rewrite!(e, node -> begin
        node isa AbstractDict && get(node, "op", "") == "index" || return nothing
        a = node["args"]
        length(a) == 2 && a[1] == from && a[2] == fidx || return nothing
        n[] += 1
        return ix(to, tidx)
    end)
    return n[]
end

# --------------------------------------------------------------------------- #
# capacity_doc -- the surgery.
# --------------------------------------------------------------------------- #
"""
    capacity_doc(doc, C; say = println) -> (doc, meta)

`doc` is the SPLIT CHEMISTRY document (`EA.flattened_to_esm(splitparts[2])`,
after `index_promoted_refs_by_loop!`), loaded at ANY legal grid. Returns a copy
rewritten to a `C x 1 x 1` lane grid whose per-cell geometry arrives through
runtime buffers, plus a `meta` NamedTuple naming the buffers the caller must
fill (`lane_arrays`, `emis_arrays`, `lonc`) and the closure that survived.
"""
function capacity_doc(doc::AbstractDict, C::Int; say = println)
    d = deepcopy(doc)
    idx = d["index_sets"]
    for (k, v) in ("lon" => C, "lat" => 1, "lev" => 1, "lev_nodes" => 2,
                   "ov.tgt_cells" => C)
        haskey(idx, k) && (idx[k]["size"] = v)
    end
    M = d["models"]["Flattened"]; V = M["variables"]; E = M["equations"]
    byname = Dict{String,Int}()
    for (i, eq) in enumerate(E)
        n = lhs_name(eq); n === nothing && continue
        haskey(byname, n) || (byname[n] = i)
    end
    nsurg = 0
    _need(n) = haskey(byname, n) ? byname[n] :
        error("capacity_doc: the chemistry document has no equation for `$n`; " *
              "the split or the model changed and the surgery below is stale")

    # --- new lane buffers ---------------------------------------------------
    for nm in LANE_VARS
        V[nm] = Dict{String,Any}("shape" => Any["lon"], "type" => "parameter",
            "units" => "1", "description" => "capacity-build lane input (capacity_chem.jl)")
    end

    # --- 1. Pc: the hybrid-coefficient reads are level-indexed --------------
    # P_c = 1/2 (Ap[k] + Ap[k+1]) + 1/2 (Bp[k] + Bp[k+1]) PS. The two edge sums
    # are per-cell constants, so they come in as lanes and PS stays live.
    let eq = E[_need("Transport3D.Pc")]
        eq["rhs"]["args"] = Any["Cap.apc", "Cap.bpc", "Transport3D.PS"]
        eq["rhs"]["expr"] = Dict{String,Any}("op" => "*", "args" => Any[0.5,
            Dict{String,Any}("op" => "+", "args" => Any[ix("Cap.apc", "gi"),
                Dict{String,Any}("op" => "*", "args" => Any[ix("Cap.bpc", "gi"),
                    ix("Transport3D.PS", "gi", "gj")])])])
        nsurg += 1
    end

    # --- 2. cos_sza_c: solar geometry from the lat/lon INDEX ----------------
    let eq = E[_need("Transport3D.cos_sza_c")]
        a = swap_index!(eq["rhs"], "Transport3D.latp", "gj", "Cap.lat", "gi")
        b = swap_index!(eq["rhs"], "Transport3D.lonp", "gi", "Cap.lon", "gi")
        a >= 1 && b >= 1 || error("capacity_doc: cos_sza_c no longer reads latp[gj]/lonp[gi] " *
                                  "(got $a/$b); the solar chain changed")
        eq["rhs"]["args"] = Any["Cap.lat", "Cap.lon"]
        nsurg += 1
    end

    # --- 3. levc IS the level index -----------------------------------------
    let eq = E[_need("Transport3D.levc")]
        eq["rhs"]["args"] = Any["Cap.lev"]
        eq["rhs"]["expr"] = ix("Cap.lev", "gi")
        nsurg += 1
    end

    # --- 4. the five emission terms gate on `gk < 2` ------------------------
    for nm in EMIS_EQS
        eq = E[_need(nm)]
        hit = Ref(0)
        rewrite!(eq["rhs"], node -> begin
            node isa AbstractDict && get(node, "op", "") == "<" || return nothing
            a = node["args"]
            length(a) == 2 && a[1] == "gk" && a[2] == 2 || return nothing
            hit[] += 1
            return Dict{String,Any}("op" => "<", "args" => Any[ix("Cap.lev", "gi"), 2])
        end)
        hit[] == 1 || error("capacity_doc: $nm has $(hit[]) `gk < 2` gates, expected 1")
        push!(eq["rhs"]["args"], "Cap.lev")
        nsurg += 1
    end

    # --- 5. lonc (the emissions time zone) and E_* (the NEI regrid) become
    #        runtime buffers; both are constant in t, so nothing is lost.
    drop = Set{String}()
    for nm in ("NEIRegrid.lonc", EMIS_E...)
        push!(drop, nm)
        haskey(V, nm) || error("capacity_doc: no variable `$nm`")
        shp = get(V[nm], "shape", Any["lon"])
        V[nm] = Dict{String,Any}("shape" => shp, "type" => "parameter", "units" => "1",
            "description" => "capacity-build lane input (capacity_chem.jl)")
        nsurg += 1
    end
    E = Any[eq for eq in E if !(lhs_name(eq) in drop)]
    M["equations"] = E

    # --- 6. prune to what the derivative equations can still reach ----------
    obsdef = Dict{String,Any}(); roots = Any[]
    for eq in E
        n = lhs_name(eq)
        if n === nothing
            push!(roots, eq)
        else
            haskey(obsdef, n) || (obsdef[n] = eq["rhs"])
        end
    end
    need = Set{String}(); stack = String[]
    for eq in roots
        for n in names_of(eq["rhs"]); push!(stack, n); end
        for n in names_of(get(eq, "lhs", nothing)); push!(stack, n); end
    end
    while !isempty(stack)
        n = pop!(stack)
        n in need && continue
        push!(need, n)
        haskey(obsdef, n) && for m in names_of(obsdef[n]); m in need || push!(stack, m); end
    end
    nbefore = length(E)
    M["equations"] = Any[eq for eq in E if (n = lhs_name(eq); n === nothing || n in need)]
    keep = Set{String}(k for k in keys(V) if k in need)
    for eq in M["equations"]; union!(keep, intersect(names_of(eq), keys(V))); end
    for k in collect(keys(V)); k in keep || delete!(V, k); end

    say(@sprintf("  capacity doc: %d surgery sites, %d/%d equations kept, %d variables, C=%d",
                 nsurg, length(M["equations"]), nbefore, length(V), C))
    return d, (; C, closure = need,
               lane_arrays = collect(LANE_VARS), emis_arrays = collect(EMIS_E),
               lonc = "NEIRegrid.lonc", variables = keep)
end

# --------------------------------------------------------------------------- #
# Runtime side: the gather.
# --------------------------------------------------------------------------- #
# A PADDING lane is not "off". It is a lane the compiled program evaluates like
# any other, and its result lands in the same vector the error norm reduces over
# -- so leaving its inputs at zero does not cost nothing, it returns NaN
# (`log(PS/Pc)` with both zero) and NaNs the whole step's `EEst`. Every padding
# lane therefore carries a DUPLICATE of a real cell: valid arithmetic, a
# discardable answer, and the only lane in the batch whose output nobody reads.
function _filled(cells::Vector{NTuple{3,Int}})
    fill_c = (0, 0, 0)
    for c in cells; c[1] != 0 && (fill_c = c; break); end
    fill_c[1] == 0 && error("_filled: every lane is padding; a batch needs at least one real cell")
    return NTuple{3,Int}[c[1] == 0 ? fill_c : c for c in cells]
end

"""
    lane_buffers(meta, refparam, refconst, C) -> (param_arrays, const_arrays)

Allocate the capacity build's runtime buffers. Every GEOS-FP forcing array is
re-shaped to its LANE form -- `(2, 1, 2, C+1)` for a 3-D field and `(2, 2, C+1)`
for a surface field -- because the capacity document, loaded at LON0=LAT0=1,
reads it at `[r, gk, 1+gj, 1+gi]` with `gj == gk == 1`, i.e. at slot
`[r, 1, 2, gi+1]`. The `+1` origin is the halo the load metaparameters bake in;
it costs one wasted lane and keeps the index expression untouched.
"""
function lane_buffers(meta, refparam::AbstractDict, C::Int)
    pa = Dict{String,Any}()
    for (k, v) in refparam
        ks = String(k); ks in meta.variables || continue
        a = v isa AbstractArray ? v : continue
        pa[ks] = ndims(a) == 4 ? zeros(Float64, size(a, 1), 1, 2, C + 1) :
                 ndims(a) == 3 ? zeros(Float64, size(a, 1), 2, C + 1) :
                 error("lane_buffers: $ks is $(ndims(a))-D; expected a 3-D or 4-D " *
                       "GEOS-FP forcing array")
    end
    for nm in meta.lane_arrays; pa[nm] = zeros(Float64, C); end
    for nm in meta.emis_arrays; pa[nm] = zeros(Float64, C); end
    pa[meta.lonc] = zeros(Float64, C)
    return pa
end

"""
    gather_forcing!(capbufs, refbufs, cells; lon0, lat0) -> capbufs

`cells` is a length-C vector of `(i, j, k)` REFERENCE grid indices (1-based,
local), one per lane; a lane with `(0,0,0)` is padding and is left at whatever
it held. `lon0`/`lat0` are the reference load's LON0/LAT0, so the native slot a
local cell reads is `(lat0 + j, lon0 + i)`.

`refparam` MUST be the `param_arrays` DICTIONARY the reference build was given,
not `forcing_buffers(f)`. Both alias the same memory, but the evaluator stores
the `vec` view (`_PGatherArray.flat`), so the buffers NamedTuple hands back
flat `Vector{Float64}`s with the grid shape gone. Gathering off the NamedTuple
silently matches no array at all, leaves every lane at zero, and the RHS then
returns NaN through `log(PS/Pc)` -- which is exactly what it did.

The index convention is Julia's: the evaluator linearizes a forcing subscript
with `LinearIndices(dims)`, i.e. COLUMN-major, so `src[r, k, lat, lon]` here
selects the same element the document's `[r, gk, lat0+gj, lon0+gi]` does.
"""
function gather_forcing!(capbufs, refparam::AbstractDict, cells::Vector{NTuple{3,Int}};
                        lon0::Int, lat0::Int)
    cs = _filled(cells)
    nd4 = 0; nd3 = 0
    for (rawk, src) in refparam
        ks = String(rawk); haskey(capbufs, ks) || continue
        src isa AbstractArray || continue
        dst = capbufs[ks]; nr = size(src, 1)
        if ndims(src) == 4
            nd4 += 1
            @inbounds for (l, (i, j, kk)) in enumerate(cs)
                for r in 1:nr; dst[r, 1, 2, l + 1] = src[r, kk, lat0 + j, lon0 + i]; end
            end
        elseif ndims(src) == 3
            nd3 += 1
            @inbounds for (l, (i, j, kk)) in enumerate(cs)
                for r in 1:nr; dst[r, 2, l + 1] = src[r, lat0 + j, lon0 + i]; end
            end
        else
            error("gather_forcing!: `$ks` is $(ndims(src))-D; expected 3-D or 4-D")
        end
    end
    (nd4 + nd3) > 0 || error("gather_forcing!: gathered NOTHING -- `refparam` must be " *
                             "the param_arrays DICT (shaped arrays), not forcing_buffers()")
    return capbufs
end

"""
    gather_geometry!(capbufs, meta, cells; Ap, Bp, latp, lonp, E, lonc)

Fill the eight index-derived lane inputs. `Ap`/`Bp` are the reference hybrid
tables, `latp`/`lonp` the reference latitude/longitude coordinate vectors,
`E` a name -> reference `E_*` vector map (flat `(j-1)*NLON + i`), and `lonc`
the reference target-cell centre longitudes.
"""
function gather_geometry!(capbufs, meta, cells::Vector{NTuple{3,Int}};
                          Ap, Bp, latp, lonp, E::AbstractDict, lonc, nlon::Int)
    for (l, (i, j, k)) in enumerate(_filled(cells))
        capbufs["Cap.apc"][l] = Ap[k] + Ap[k + 1]
        capbufs["Cap.bpc"][l] = Bp[k] + Bp[k + 1]
        capbufs["Cap.lat"][l] = latp[j]
        capbufs["Cap.lon"][l] = lonp[i]
        capbufs["Cap.lev"][l] = k
        capbufs[meta.lonc][l]  = lonc[i]
        for nm in meta.emis_arrays; capbufs[nm][l] = E[nm][(j - 1) * nlon + i]; end
    end
    return capbufs
end

end # module
