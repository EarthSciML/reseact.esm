# ===========================================================================
# geosfp_grids.jl -- the GEOS-FP resolution table, and nothing else.
#
# Deliberately DEPENDENCY-FREE (no `using` at all): split_common.jl includes it
# for `native_slice`/`reseact_forcing`, and run_reseact_adjoint.jl includes it
# on its own, before any driver, to size the CONUS box for the chosen row. A
# second copy of these numbers anywhere is how the meteorology and the emissions
# end up on different grids, so there is exactly one.
# ===========================================================================

# --------------------------------------------------------------------------- #
# 1b. THE RESOLUTION TABLE: one entry per GEOS-FP grid, and the ONLY place a
#     grid spacing, a native extent or a URL token is written down.
#
# Every number in a row is a property of the ARCHIVE, not of this model, and the
# rows were verified against the files themselves (netCDF header read over HTTP
# range requests, 2016-01-01):
#
#   4x5           72 x 46 x 72     lon -180:5:175      lat -90:4:90
#   2x2.5        144 x 91 x 72     lon -180:2.5:177.5  lat -90:2:90
#   0.25x0.3125 1152 x 721 x 72    lon -180:0.3125:..  lat -90:0.25:90
#   0.25x0.3125_CH 225 x 161 x 72  lon  70:0.3125:140  lat  15:0.25:55
#
# All four carry the same 72 hybrid levels, so `hybrid_coefs.json`, dA/dB/Ap/Bp
# and NLEV are untouched by a horizontal switch; and all four archive the same
# six collections on the same cadences (I3 3-hourly instantaneous, A3 3-hourly
# means, A1 hourly means), so `reseact_forcing`'s phase/dt table carries over
# unchanged too. What DOES change is the URL, the cell spacing, and the native
# extent the halo bounds are checked against.
#
# `lon_first`/`lat_first` are the FIRST CELL CENTRE / LAT POINT, i.e. the origin
# of centre(i) = lon_first + dlon*(i-1). The polar rows are half-width cells in
# every global product; 4x5 and 2x2.5 report their centres AT the pole (-90) so
# the linear form is exact everywhere, while 0.25x0.3125 reports the half-cell
# centre (-89.9375) and the linear form is exact only for j >= 2. That matters
# for a slice touching a pole and for nothing else; CONUS is far from both.
#
# `conus` is the (lon0, lat0, nlon, nlat) index box that reproduces the SAME
# geographic footprint at each resolution -- lon centres -125..-65, lat points
# 26..50 -- so `native_slice(res = "2x2.5")` simulates the domain
# `native_slice()` does, at 3.6x the columns. It is the per-row default, not a
# constraint: any in-bounds box is still spelled by hand.
#
# NOT in the table, deliberately: the nested North America / Europe / Asia
# domains. They exist as directories on both mirrors but carry ONLY `soil`
# files -- no A1/A3*/I3 met -- at every year sampled (2016, 2018, 2020, 2023),
# so there is no nested CONUS product to point at. High resolution over CONUS
# means slicing the GLOBAL 0.25x0.3125 files, which is why the china row is the
# only nested one here (it is complete, and it is the shape a nested row takes).
# --------------------------------------------------------------------------- #
struct GEOSFPGrid
    dir::String          # bucket prefix:      GEOS_<dir>/GEOS_FP/YYYY/MM/
    suffix::String       # filename token:     GEOSFP.YYYYMMDD.<coll>.<suffix>.nc
    dlon::Float64        # zonal cell width, deg
    dlat::Float64        # meridional point spacing, deg
    lon_first::Float64   # centre of native lon cell 1
    lat_first::Float64   # native lat point 1
    nlon::Int            # native zonal extent
    nlat::Int            # native meridional extent
    conus::NTuple{4,Int} # (lon0, lat0, nlon, nlat) for the standard CONUS box
end

const GEOSFP_GRIDS = Dict{String,GEOSFPGrid}(
    "4x5"            => GEOSFPGrid("GEOS_4x5", "4x5", 5.0, 4.0,
                                   -180.0, -90.0, 72, 46, (11, 29, 13, 7)),
    "2x2.5"          => GEOSFPGrid("GEOS_2x2.5", "2x25", 2.5, 2.0,
                                   -180.0, -90.0, 144, 91, (22, 58, 25, 13)),
    "0.25x0.3125"    => GEOSFPGrid("GEOS_0.25x0.3125", "025x03125", 0.3125, 0.25,
                                   -180.0, -90.0, 1152, 721, (176, 464, 193, 97)),
    "0.25x0.3125_CH" => GEOSFPGrid("GEOS_0.25x0.3125_CH", "025x03125.CH", 0.3125, 0.25,
                                   70.0, 15.0, 225, 161, (32, 40, 161, 81)))

"""The [`GEOSFPGrid`] row for `res`, or an error naming the rows that exist."""
function geosfp_grid(res::AbstractString)
    haskey(GEOSFP_GRIDS, res) || throw(ArgumentError(
        "unknown GEOS-FP resolution $(repr(res)); the table has " *
        join(sort(collect(keys(GEOSFP_GRIDS))), ", ") *
        ". Nested NA/EU/AS domains are deliberately absent: the mirrors carry " *
        "only their `soil` files, no meteorology."))
    return GEOSFP_GRIDS[res]
end

