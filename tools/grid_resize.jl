# ===========================================================================
# grid_resize.jl -- programmatically resize the ReSEACT .esm grid.
# ===========================================================================
# The reseact.esm grid (NLON x NLAT x NLEV) is a NATIVE, no-regridding slice of
# the GEOS-FP 4x5 global array (72 lon x 46 lat x 72 lev). Its geometry is baked
# into the transport flux expressions as literals, so it is NOT a single knob.
# This helper does the coordinated multi-literal rewrite that a resize needs:
#
#   1. the 6 expression_template_imports args {NLON, NLAT, NLEV};
#   2. the level clamp  min(gke, 72)  ->  min(gke, NLEV);
#   3. the lat   clamp  min(gje, 7)   ->  min(gje, NLAT).
#
# There is NO lon clamp (open BC reads native offset 14+gi directly), and the
# lat/lev base offsets (14+gi, 29+..., 28+...) and the pole/edge floors
# (max(gje,2), max(gke-1,1)) are grid-size-independent, so they are left alone.
#
# BOUNDS (native GEOS-FP 4x5 array, no regridding):
#   NLON in 1..57   (east halo native 14+NLON+1 must be <= 72)
#   NLAT in 1..17   (top lat point native 29+NLAT must be <= 46)
#   NLEV in 1..72   (hybrid_coefs table length)
#
# dA/dB/Ap/Bp are supplied as const_arrays at build time; when NLEV < 72 the
# caller must slice them to NLEV (see `slice_hybrid_coefs`), because the model's
# `lev` axis is now shorter than the 72-entry table. This helper only rewrites
# the .esm; the coef slice happens in the runner's forcing wiring.
# ===========================================================================
module GridResize

using JSON3

const BASE_NLON = 7
const BASE_NLAT = 7
const BASE_NLEV = 72
const MAX_NLON = 57
const MAX_NLAT = 17
const MAX_NLEV = 72

# JSON3 parses to immutable views; convert to mutable Dict{String,Any}/Vector{Any}.
_mut(x::JSON3.Object) = Dict{String,Any}(String(k) => _mut(v) for (k, v) in x)
_mut(x::JSON3.Array)  = Any[_mut(v) for v in x]
_mut(x) = x

# Collect every `expression_template_imports` list found anywhere in the doc.
function _find_imports(x, acc = Any[])
    if x isa Dict
        haskey(x, "expression_template_imports") && append!(acc, x["expression_template_imports"])
        for v in values(x); _find_imports(v, acc); end
    elseif x isa Vector
        for v in x; _find_imports(v, acc); end
    end
    return acc
end

# Recursively rewrite a `min(var, lit_from)` node in place to `min(var, lit_to)`.
function _rewrite_min_clamp!(x, var::String, lit_from::Int, lit_to::Int)
    if x isa Dict
        if get(x, "op", nothing) == "min" && x["args"] isa Vector && length(x["args"]) == 2 &&
           x["args"][1] == var && x["args"][2] == lit_from
            x["args"][2] = lit_to
        end
        for v in values(x); _rewrite_min_clamp!(v, var, lit_from, lit_to); end
    elseif x isa Vector
        for v in x; _rewrite_min_clamp!(v, var, lit_from, lit_to); end
    end
    return x
end

"""
    resize_esm(base_path, out_path; NLON=7, NLAT=7, NLEV=72) -> out_path

Read the base .esm, resize its grid to (NLON, NLAT, NLEV), and write out_path.
Returns out_path. Validates the native-array bounds.
"""
function resize_esm(base_path::AbstractString, out_path::AbstractString;
                    NLON::Int = BASE_NLON, NLAT::Int = BASE_NLAT, NLEV::Int = BASE_NLEV)
    1 <= NLON <= MAX_NLON || error("NLON=$NLON out of native GEOS-FP bounds 1..$MAX_NLON")
    1 <= NLAT <= MAX_NLAT || error("NLAT=$NLAT out of native GEOS-FP bounds 1..$MAX_NLAT")
    1 <= NLEV <= MAX_NLEV || error("NLEV=$NLEV out of hybrid-table bounds 1..$MAX_NLEV")

    doc = _mut(JSON3.read(read(base_path, String)))

    # 1. the 6 template import args. `expression_template_imports` lives under
    #    whichever model carries the transport stencils (models.Transport3D); find
    #    it structurally rather than hard-coding the path.
    n_imports = 0
    for imp in _find_imports(doc)
        ov = get(imp, "bindings", nothing)
        ov === nothing && continue
        if haskey(ov, "NLON"); ov["NLON"] = NLON; n_imports += 1; end
        haskey(ov, "NLAT") && (ov["NLAT"] = NLAT)
        haskey(ov, "NLEV") && (ov["NLEV"] = NLEV)
    end
    n_imports == 6 || error("expected 6 grid template imports, rewrote $n_imports")

    # 2. + 3. clamp literals (only when shrinking below the baked base value)
    if NLEV != BASE_NLEV
        _rewrite_min_clamp!(doc, "gke", BASE_NLEV, NLEV)
    end
    if NLAT != BASE_NLAT
        _rewrite_min_clamp!(doc, "gje", BASE_NLAT, NLAT)
    end

    open(out_path, "w") do io
        JSON3.pretty(io, doc)
    end
    return out_path
end

"""
    slice_hybrid_coefs(const_arrays, NLEV) -> const_arrays'

Return a copy of const_arrays with the vertical hybrid tables sliced to the
first NLEV levels (dA/dB have length 72; Ap/Bp have length 73 = edges).
"""
function slice_hybrid_coefs(const_arrays::AbstractDict, NLEV::Int)
    NLEV == BASE_NLEV && return const_arrays
    out = Dict{String,Any}(const_arrays)
    for k in ("Transport3D.dA", "Transport3D.dB")
        haskey(out, k) && (out[k] = out[k][1:NLEV])
    end
    for k in ("Transport3D.Ap", "Transport3D.Bp")
        haskey(out, k) && (out[k] = out[k][1:(NLEV + 1)])
    end
    return out
end

end # module GridResize
