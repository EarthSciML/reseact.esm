# ReSEACT 0-D box, stage 2: 24 h run with NEI2016 emissions, FastJX photolysis and
# deposition. Usage:  julia box_run.jl [site]    site = kansas (default) | chicago
#
# The site's lat/lon drives three things at once and they must stay consistent:
#   * FastJX's solar zenith angle (degrees),
#   * NEI's timezone offset + delp_dry_surface interpolation (radians),
#   * which NEI grid cell the emission fluxes are read from.
import Pkg
Pkg.activate("/Users/ctessum/code/earthsciml/reseact.esm/run-model-jl"; io=devnull)
try; @eval import OrdinaryDiffEqRosenbrock; catch; Pkg.add("OrdinaryDiffEqRosenbrock"; io=devnull); @eval import OrdinaryDiffEqRosenbrock; end
using EarthSciAST, Dates, Printf
const EA = EarthSciAST

HERE = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/box_model"
include(joinpath(HERE, "nei_provider.jl"))

site = length(ARGS) >= 1 ? lowercase(ARGS[1]) : "kansas"
LAT, LON, LABEL = site == "chicago" ? (41.881, -87.628, "Chicago (urban)") :
                  site == "kansas"  ? (40.000, -97.000, "central Kansas (rural)") :
                  error("unknown site '$(site)'")

# 2016-05-04T07:00:00Z. The NEI file is May 2016; at lon=-97 the NEI timezone offset is
# floor(-97/15) = -7 h, so 07:00 UTC is local midnight -> the 24 h window is one full
# local day. 2016-05-04 is a Wednesday, so the day-of-week scaling is a weekday value.
const T0 = 1462345200.0
@assert unix2datetime(T0) == DateTime(2016, 5, 4, 7, 0, 0)
const WHEN = unix2datetime(T0)          # picks the NEI monthly file + days-in-month

path = joinpath(HERE, "box.esm")
fs = EA.flatten(EA.load(path); base_path=HERE)
println("── ", LABEL, "  lat=", LAT, "  lon=", LON, "  start=", unix2datetime(T0), " UTC")
println("states=", length(fs.state_variables), "  observeds=", length(fs.observed_variables))

# Bind the 69 pure-I/O NEI loader fields from the real EPA 12US1 NetCDF.
providers, fluxes, cell = nei_providers(fs, "NEI2016Emis"; lat=LAT, lon=LON, when=WHEN)
@printf("NEI cell (row=%d, col=%d), %d loader fields bound via EarthSciIO (%s)\n",
        cell[1], cell[2], length(providers), Dates.format(WHEN, "yyyy-mm"))
for sp in ("NO", "NO2", "CO", "ISOP", "FORM")
    @printf("   flux %-5s = %11.4e kg/m^2/s\n", sp, fluxes[sp])
end

# Emissions and deposition have to act on the SAME layer, and that layer has to be deep
# enough to be a defensible well-mixed box. NEI dilutes its flux into its own surface layer,
# whose column mass is g0_100*delp [kg/m^2] (here ~146 kg/m^2, i.e. only ~122 m). A 0-D box
# has no entrainment from aloft, so depositing O3 out of a 122 m layer (8.5 h lifetime)
# simply drains it — O3 falls by 12 ppb overnight before any photochemistry runs.
#
# So make the box a 1 km well-mixed boundary layer: NEI's own Δz parameter rescales the
# emission depth (emission = flux / (g0_100*delp) / Δz), and deposition uses k = vdep/H_BL.
e = fs.observed_variables["NEI2016Emis.delp_dry_surface"].expression
tbl, ax, ay = e.args[1].value, Float64.(e.args[2].value), Float64.(e.args[3].value)
let x = clamp(LON, ax[1], ax[end]), y = clamp(LAT, ay[1], ay[end])
    i = clamp(searchsortedlast(ax, x), 1, length(ax)-1)
    j = clamp(searchsortedlast(ay, y), 1, length(ay)-1)
    wx = (x-ax[i])/(ax[i+1]-ax[i]); wy = (y-ay[j])/(ay[j+1]-ay[j])
    z(a,b) = Float64(tbl[a][b])
    global DELP = (1-wx)*(1-wy)*z(i,j) + wx*(1-wy)*z(i+1,j) + (1-wx)*wy*z(i,j+1) + wx*wy*z(i+1,j+1)
end
const RHO_AIR = 1.2                                     # kg/m^3
const H_NEI = 10.197162129779283 * DELP / RHO_AIR       # m — NEI's own surface layer
const H_BL  = 1000.0                                    # m — well-mixed boundary layer
const DZ    = H_BL / H_NEI                              # NEI Δz: dilute emissions into H_BL
@printf("delp = %.2f hPa -> NEI surface layer %.0f m; box = %.0f m well-mixed BL (Δz = %.2f)\n",
        DELP, H_NEI, H_BL, DZ)

# Typical daytime deposition velocities [m/s] -> first-order loss k = vdep/H_BL [1/s].
vdep = Dict("O3"=>0.004, "NO2"=>0.003, "HNO3"=>0.020, "H2O2"=>0.010, "HCHO"=>0.005)
kdep = Dict(sp => v / H_BL for (sp, v) in vdep)
for sp in ("O3","NO2","HNO3","H2O2","HCHO")
    @printf("   k_%-5s = %9.3e 1/s  (vdep=%.1f cm/s, lifetime %.0f h)\n",
            sp, kdep[sp], 100*vdep[sp], 1/kdep[sp]/3600)
end

Tsec = 24 * 3600.0
save = collect(0.0:3600.0:Tsec)
params = Dict(
  "SuperFast.T" => 290.0, "SuperFast.P" => 101325.0,
  "FastJX.T" => 290.0, "FastJX.P" => 101325.0,
  "FastJX.H2O" => 1.8e7,                             # ~1.8% H2O (ppb); default 450 ppb is ~1e4x too low
  "Clock.t_utc0" => T0,
  "Solar.lat" => LAT, "Solar.lon" => LON,            # degrees; drives FastJX.cos_sza
  # NEI: same absolute clock as FastJX, lat/lon in RADIANS, surface layer (lev<2).
  "NEI2016Emis.t_ref" => T0,
  "NEI2016Emis.lat" => deg2rad(LAT), "NEI2016Emis.lon" => deg2rad(LON),
  "NEI2016Emis.lev" => 1.0, "NEI2016Emis.scale" => 1.0, "NEI2016Emis.Δz" => DZ,
  # deposition: first-order loss k = vdep/H out of the same layer NEI emits into; no rain
  "SuperFastDepositionSink.k_O3"   => kdep["O3"],
  "SuperFastDepositionSink.k_NO2"  => kdep["NO2"],
  "SuperFastDepositionSink.k_HNO3" => kdep["HNO3"],
  "SuperFastDepositionSink.k_H2O2" => kdep["H2O2"],
  "SuperFastDepositionSink.k_HCHO" => kdep["HCHO"],
  "SuperFastDepositionSink.k_wet"  => 0.0,
)
ics = Dict("SuperFast.O3"=>30.0, "SuperFast.NO2"=>1.0, "SuperFast.NO"=>0.5)

el = @elapsed sim = EA.simulate(fs, (0.0, Tsec);
        alg = OrdinaryDiffEqRosenbrock.Rosenbrock23(autodiff=false),
        parameters = params, initial_conditions = ics, providers = providers,
        reltol = 1e-7, abstol = 1e-10, saveat = save)
println("\nsuccess=", sim.success, "  retcode=", sim.retcode, "  nsaved=", length(sim.t),
        @sprintf("  (%.1f s)", el))

vm = sim.var_map
idx(sp) = vm[first(filter(k -> string(k) == "SuperFast.$sp", collect(keys(vm))))]
species = ["O3","NO2","NO","OH","HO2","CH2O","CO","ISOP"]
cols = Dict(sp => idx(sp) for sp in species)

println("\nlocal hr  " * join([rpad(sp, 11) for sp in species]))
for (k, tt) in enumerate(sim.t)
    vals = [sim.u[k][cols[sp]] for sp in species]
    println(rpad(round(Int, tt/3600), 9), join([rpad(round(v, sigdigits=4), 11) for v in vals]))
end

println("\ndiurnal swing over 24 h:")
for sp in species
    s = [sim.u[k][cols[sp]] for k in 1:length(sim.t)]
    lo, hi = minimum(s), maximum(s)
    @printf("  %-5s min=%-11.4g max=%-11.4g swing=%.1f%%\n", sp, lo, hi,
            hi > 0 ? 100*(hi-lo)/hi : 0.0)
end
oh = [sim.u[k][cols["OH"]] for k in 1:length(sim.t)]
o3 = [sim.u[k][cols["O3"]] for k in 1:length(sim.t)]
@printf("\nOH peaks at local hour %d (solar noon ~12) | O3: %.1f -> %.1f ppb (%+.1f)\n",
        argmax(oh)-1, o3[1], o3[end], o3[end]-o3[1])
println(maximum(oh) > 3*minimum(oh) ? "DIURNAL CYCLE PRESENT" : "little variation")
