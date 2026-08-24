# NEI2016 providers for the 0-D box, built on EarthSciIO.
#
# EarthSciIO owns all the I/O: URL resolution (including the `file://` local mirror),
# caching, and NetCDF decoding. EarthSciAST's EarthSciIO extension already adapts
# `EarthSciIO.Provider` to the provider seam, so an EarthSciIO provider can be handed
# straight to `esm_problem(...; providers=...)`.
#
# EarthSciIO deliberately stops there — "Variable remap / unit conversion / regrid are NOT
# here" (EarthSciIO/julia/src/provider.jl:1-6). It returns the RAW native array. Two things
# therefore remain, and neither has a home in EarthSciIO:
#
#   1. UNIT CONVERSION. The loader .esm declares these variables as kg/m^2/s, but the raw
#      IOAPI file holds monthly TOTALS labelled "tons/day". The conversion is
#         raw / days_in_month * (907.185/86400) / (XCELL*YCELL)  ->  kg/m^2/s
#      (EarthSciData.jl src/nei2016monthly.jl `loadslice!`; the /days_in_month is issue #209).
#      The .esm schema HAS a `unit_conversion` field for exactly this, and the Python runtime
#      applies it — but the Julia runtime parses and ignores it, and this loader declares none.
#      Even if it did, a static factor cannot express this one: it depends on days-in-month.
#
#   2. CELL SELECTION. EarthSciIO has no spatial subsetting, so a sample is the full
#      459x299 12US1 grid. A 0-D box needs one cell (EarthSciAST const-folds a loader field
#      to a scalar only when the array has length 1).
#
# Both live in a composed provider below. That is the sanctioned extension point, not a
# workaround: EarthSciAST's provider protocol (src/data_refresh.jl) is a pair of generics
# whose "concrete implementations live outside EarthSciAST", and the EarthSciIO adapter is
# itself just one such implementation. We add another that wraps it.
#
# Every path and grid constant is read from the loader component — nothing is hard-coded here.

using EarthSciIO, Dates, JSON3
import EarthSciAST

const NEI_LOADER = normpath(joinpath(@__DIR__, "..", "..", "..", "EarthSciModels",
                                     "components", "earthsci_data", "nei2016_monthly_loader.esm"))

"""Expand a loader `url_template` / mirror: environment variables (EarthSciIO's FileTransport
does this too, but we need the resolved path to test which mirror actually exists), plus
`{sector}` and `{date:%m}`-style strftime fields."""
function expand_url(tmpl::AbstractString, sector::AbstractString, when::DateTime)
    s = replace(tmpl, "{sector}" => sector)
    s = replace(s, r"\{date:([^}]+)\}" => m -> Dates.format(when, _strftime_to_julia(m)))
    for (k, v) in ENV
        s = replace(s, "\${$k}" => v, "\$$k" => v)
    end
    # A trailing slash on e.g. EARTHSCIDATADIR leaves "//" in the path. Collapse duplicate
    # slashes in the PATH only — the scheme's own "//" must survive, and so must the leading
    # "/" of an absolute path (file:///abs/path).
    m = match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*://)(.*)$", s)
    m === nothing && return s
    return m.captures[1] * replace(m.captures[2], r"/{2,}" => "/")
end
# Only the fields these templates actually use.
_strftime_to_julia(m) = replace(match(r"\{date:([^}]+)\}", m).captures[1],
                                "%Y" => "yyyy", "%m" => "mm", "%d" => "dd")

"""Resolve the loader's source to a URL EarthSciIO can fetch: prefer a `file://` mirror whose
file is actually present locally, else fall back to the canonical remote `url_template`."""
function resolve_source(loader, sector::AbstractString, when::DateTime)
    src = loader["source"]
    for m in get(src, "mirrors", String[])
        u = expand_url(String(m), sector, when)
        startswith(u, "file://") || continue
        isfile(replace(u, "file://" => "")) && return u
    end
    return expand_url(String(src["url_template"]), sector, when)
end

# Forward Lambert Conformal Conic (spherical, two standard parallels), from the loader's own
# `lcc_parameters` — matching the file's GDTYP=2 projection.
function lcc_forward(lcc, lon_deg, lat_deg)
    φ  = deg2rad(lat_deg);            λ  = deg2rad(lon_deg)
    φ1 = deg2rad(lcc["truelat1"]);    φ2 = deg2rad(lcc["truelat2"])
    φ0 = deg2rad(lcc["cen_lat"]);     λ0 = deg2rad(lcc["stand_lon"])
    R  = lcc["sphere_radius_m"]
    n = log(cos(φ1)/cos(φ2)) / log(tan(π/4 + φ2/2) / tan(π/4 + φ1/2))
    F = cos(φ1) * tan(π/4 + φ1/2)^n / n
    ρ  = R * F / tan(π/4 + φ/2)^n
    ρ0 = R * F / tan(π/4 + φ0/2)^n
    θ = n * (λ - λ0)
    return (ρ*sin(θ), ρ0 - ρ*cos(θ))
end

"""
Composes an `EarthSciIO.Provider` (which does the fetch/cache/decode and returns the raw
459x299 native array) into the single-cell, unit-converted scalar a 0-D consumer needs.
"""
struct NEICellProvider{P}
    inner::P          # EarthSciIO.Provider, one variable
    col::Int          # 1-based (COL, ROW, LAY, TSTEP) index of the box's grid cell
    row::Int
    scale::Float64    # tons/month-total -> kg/m^2/s
end
EarthSciAST.provider_refresh_times(p::NEICellProvider) =
    EarthSciAST.provider_refresh_times(p.inner)
function EarthSciAST.provider_sample(p::NEICellProvider, t::Real)
    a = EarthSciAST.provider_sample(p.inner, Float64(t))   # raw native array, via the EarthSciIO adapter
    # EarthSciIO hands back the array in the FILE's declared dim order — the IOAPI variables
    # are float V(TSTEP, LAY, ROW, COL), so that is exactly the index order here. (NCDatasets
    # would have reversed it to column-major (COL, ROW, LAY, TSTEP); the EarthSciIO adapter
    # deliberately does not reorder.)
    return [Float64(a[1, 1, p.row, p.col]) * p.scale]      # length 1 => const-folds to a scalar
end

"""
    nei_providers(fs, owner; lat, lon, when) -> (providers, fluxes, cell)

Build one provider per `<owner>.NEI2016.<SP>` loader field in the flattened system `fs`,
sampling the 12US1 cell containing (`lat`, `lon`). Returns the provider dict keyed as
`esm_problem` expects, the converted kg/m^2/s fluxes by species, and the (row, col) cell.
"""
function nei_providers(fs, owner::AbstractString; lat::Float64, lon::Float64, when::DateTime)
    doc = JSON3.read(read(NEI_LOADER, String))
    loader = doc["data_loaders"]["NEI2016"]
    md = loader["metadata"]
    sector = String(md["default_sector"])

    url = resolve_source(loader, sector, when)
    @info "NEI source resolved by EarthSciIO" url

    # Grid geometry, all from the loader component.
    x0, y0 = md["grid_origin_m"]["x"], md["grid_origin_m"]["y"]
    dx, dy = md["native_resolution_m"]["x"], md["native_resolution_m"]["y"]
    ncols, nrows = md["nominal_dimensions"]["cols"], md["nominal_dimensions"]["rows"]

    x, y = lcc_forward(md["lcc_parameters"], lon, lat)
    col = Int(floor((x - x0) / dx)) + 1                    # 1-based
    row = Int(floor((y - y0) / dy)) + 1
    (1 <= row <= nrows && 1 <= col <= ncols) ||
        error("($(lat), $(lon)) is outside the 12US1 domain (row=$(row), col=$(col)))")

    # tons/day (really a monthly total) -> kg/m^2/s. See the header note.
    scale = (1 / daysinmonth(when)) * (907.185 / 86400) / (dx * dy)

    cache = EarthSciIO.Cache(; offline = false)            # local file:// mirror => no network
    prefix = owner * ".NEI2016."
    fields = sort([k for k in keys(fs.observed_variables) if startswith(k, prefix)])
    isempty(fields) && error("no NEI loader fields found under '$(prefix)'")

    providers = Dict{String,Any}()
    fluxes = Dict{String,Float64}()
    for name in fields
        sp = name[length(prefix)+1:end]
        inner = EarthSciIO.const_provider(cache, url; format = "netcdf", variables = [sp])
        p = NEICellProvider(inner, col, row, scale)
        providers[name] = p
        fluxes[sp] = first(EarthSciAST.provider_sample(p, 0.0))
    end
    return providers, fluxes, (row, col)
end
