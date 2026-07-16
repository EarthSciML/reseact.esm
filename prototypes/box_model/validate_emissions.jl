# Validation of the NEI2016 -> SuperFast emissions coupling.
#
# Recomputes each species' expected source term from first principles — straight from the
# NEI NetCDF flux, the delp table, the temporal scale factors and the ppb conversion — and
# compares it against what the assembled model's RHS actually produces.
#
#   expected d[X]/dt [ppb/s] = flux_X / (g0_100 * delp) / Δz * scale * temporal_X
#                              * 1e9 * MW_Air / MW_X
#
# The model RHS is evaluated at u = 0, where every chemical production/loss term (each a
# product of species concentrations) vanishes, leaving only the emission source.
import Pkg; Pkg.activate("/Users/ctessum/code/earthsciml/reseact.esm/run-model-jl"; io=devnull)
using EarthSciAST, Dates, Printf
const EA = EarthSciAST
HERE = @__DIR__
include(joinpath(HERE, "nei_provider.jl"))

const LAT, LON = 40.0, -97.0
const T0 = 1462345200.0                      # 2016-05-04T07:00:00Z = local midnight at lon -97
const WHEN = unix2datetime(T0)

# Reference constants, from GasChem.jl ext/EarthSciDataExt.jl couple2(SuperFast, NEI2016).
const MW_AIR = 28.97e-3
const MW = Dict("NO"=>30.01e-3, "NO2"=>46.0055e-3, "CO"=>28.0101e-3,
                "ISOP"=>68.12e-3, "FORM"=>30.026e-3)
const NEI2SF = Dict("NO"=>"NO", "NO2"=>"NO2", "CO"=>"CO", "ISOP"=>"ISOP", "FORM"=>"CH2O")

# The temporal profiles, transcribed from EarthSciData.jl src/nei2016monthly.jl.
const DIURNAL       = [0.45,0.45,0.6,0.6,0.6,0.6,1.45,1.45,1.45,1.45,1.4,1.4,
                       1.4,1.4,1.45,1.45,1.45,1.45,0.65,0.65,0.65,0.65,0.45,0.45]
const DIURNAL_NOX   = [0.39598674,0.31852847,0.30128068,0.29590213,0.33177775,0.43871498,
                       0.9094625,1.5850095,1.6223788,1.3429453,1.2265036,1.1937649,1.254314,
                       1.3282939,1.331211,1.4135737,1.6848333,1.710925,1.3491899,1.0586671,
                       0.84439224,0.761263,0.72693235,0.5741503]
const DIURNAL_ISOP  = [0,0,0,0,0,0,0.2376,0.7224,1.2048,1.656,2.0496,2.3616,2.5728,2.6616,
                       2.6184,2.4408,2.1288,1.6896,1.1448,0.5136,0,0,0,0]
const DOW_CO        = [1.076,1.1076,1.0706,1.0706,1.0706,0.779,0.683]
const DOW_NOX       = [1.0706,1.0706,1.0706,1.0706,1.0706,0.863,0.784]

# local_hour / day_of_week exactly as the .esm computes them.
tz_offset(lon_deg) = floor(lon_deg / 15)
function local_hour(t, lon_deg)
    t_local = t + T0 + tz_offset(lon_deg) * 3600
    Dates.hour(unix2datetime(t_local)) + 1            # 1..24
end
function day_of_week(t, lon_deg)
    t_local = t + T0 + tz_offset(lon_deg) * 3600
    D = floor(t_local / 86400)
    Int(mod(D + 3, 7)) + 1                            # 1=Mon .. 7=Sun
end
function temporal(sp, t, lon_deg)
    h, d = local_hour(t, lon_deg), day_of_week(t, lon_deg)
    sp in ("NO", "NO2") && return DOW_NOX[d] * DIURNAL_NOX[h]
    sp == "CO"          && return DOW_CO[d]  * DIURNAL[h]
    sp == "FORM"        && return DIURNAL[h]
    sp == "ISOP"        && return DIURNAL_ISOP[h]
    error("no profile for $(sp)")
end

fs = EA.flatten(EA.load(joinpath(HERE, "box.esm")); base_path=HERE)
providers, fluxes, cell = nei_providers(fs, "NEI2016Emis"; lat=LAT, lon=LON, when=WHEN)
# The providers are CONST (length-1 samples), so materialize them the way simulate would.
const_arrays = Dict{String,Any}(k => EA.provider_sample(v, 0.0) for (k,v) in providers)

p_over = Dict(
  "SuperFast.T"=>290.0, "SuperFast.P"=>101325.0, "FastJX.T"=>290.0, "FastJX.P"=>101325.0,
  "FastJX.H2O"=>1.8e7, "Clock.t_utc0"=>T0, "Solar.lat"=>LAT, "Solar.lon"=>LON,
  "NEI2016Emis.t_ref"=>T0, "NEI2016Emis.lat"=>deg2rad(LAT), "NEI2016Emis.lon"=>deg2rad(LON),
  "NEI2016Emis.lev"=>1.0, "NEI2016Emis.scale"=>1.0,
  "SuperFastDepositionSink.k_O3"=>0.0, "SuperFastDepositionSink.k_NO2"=>0.0,
  "SuperFastDepositionSink.k_HNO3"=>0.0, "SuperFastDepositionSink.k_H2O2"=>0.0,
  "SuperFastDepositionSink.k_HCHO"=>0.0, "SuperFastDepositionSink.k_wet"=>0.0)

f!, u0, p, tspan, vm = EA.build_evaluator(fs; parameter_overrides=p_over,
                                          const_arrays=const_arrays)
sidx(sp) = vm[first(filter(k -> string(k)=="SuperFast.$sp", collect(keys(vm))))]
du = zeros(length(u0)); uz = zeros(length(u0))

# delp_dry_surface at the site, bilinearly interpolated from the component's own table.
# (lon = -97 does NOT land on a grid node: the axis is -125 + 0.625k, and (-97+125)/0.625
# = 44.8, so a nearest-node lookup here would be off by ~0.4%.)
e = fs.observed_variables["NEI2016Emis.delp_dry_surface"].expression
tbl = e.args[1].value
ax, ay = Float64.(e.args[2].value), Float64.(e.args[3].value)
function bilinear(tbl, ax, ay, x, y)
    x = clamp(x, ax[1], ax[end]); y = clamp(y, ay[1], ay[end])
    i = clamp(searchsortedlast(ax, x), 1, length(ax)-1)
    j = clamp(searchsortedlast(ay, y), 1, length(ay)-1)
    wx = (x - ax[i]) / (ax[i+1] - ax[i])
    wy = (y - ay[j]) / (ay[j+1] - ay[j])
    z(a, b) = Float64(tbl[a][b])
    (1-wx)*(1-wy)*z(i,j) + wx*(1-wy)*z(i+1,j) + (1-wx)*wy*z(i,j+1) + wx*wy*z(i+1,j+1)
end
delp = bilinear(tbl, ax, ay, LON, LAT)
g0_100 = 10.197162129779283

println("site: lat=", LAT, " lon=", LON, "  NEI cell=(row=", cell[1], ", col=", cell[2],
        ")  delp=", round(delp, digits=4), " hPa  layer mass=", round(g0_100*delp, digits=1), " kg/m^2")
println("start ", unix2datetime(T0), " UTC (local midnight, ", Dates.dayname(unix2datetime(T0 + tz_offset(LON)*3600)), ")\n")

@printf("%-6s %-4s %-6s %-9s %-13s %-13s %-9s\n",
        "hour", "spc", "->SF", "temporal", "expected", "model", "rel.err")
maxerr = 0.0
for hr in (0, 6, 8, 12, 17, 22)
    t = hr * 3600.0
    f!(du, uz, p, t)
    for sp in ("NO", "NO2", "CO", "ISOP", "FORM")
        tf = temporal(sp, t, LON)
        expected = fluxes[sp] / (g0_100 * delp) * tf * 1e9 * MW_AIR / MW[sp]
        model = du[sidx(NEI2SF[sp])]
        rel = expected == 0 ? abs(model) : abs(model - expected) / abs(expected)
        global maxerr = max(maxerr, rel)
        @printf("%-6d %-4s %-6s %-9.4f %-13.5e %-13.5e %-9.2e\n",
                hr, sp, NEI2SF[sp], tf, expected, model, rel)
    end
    println()
end
@printf("max relative error across all species/times: %.3e  ->  %s\n", maxerr,
        maxerr < 1e-10 ? "PASS (emissions coupling reproduces the reference exactly)" : "FAIL")
