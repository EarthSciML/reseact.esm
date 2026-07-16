#!/usr/bin/env python3
"""Generate reseact_3d.esm: the verified 7x7x7 Transport3D driven by a NATIVE
GEOS-FP 4x5 CONUS slice (no regridding).

Built by PATCHING prototypes/transport_3d/transport_3d.esm — itself generated from
the verified regional-inflow exemplar — so the proven horizontal/vertical rule stack
stays byte-identical and only the FORCING changes: the analytic Mx/My/Mz/PS are
replaced by real GEOS-FP fields sliced out of the native grid at
  lon: i_nat = 14 + i   (local cells 1..7 -> centres -110..-80)
  lat: j_nat = 29 + j   (local points 1..7 -> 26..50)
  lev: k               (1..7, k=1 = surface, no flip)
See memory geosfp-native-slice-facts for the measured ground truth behind every
constant here.
"""
import json, collections, os

SRC = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/transport_3d/transport_3d.esm"
OUT = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/reseact_3d/reseact_3d.esm"
GEOSFP_REF = "../../../EarthSciModels/components/earthsci_data/geosfp.esm"

NLON = NLAT = NLEV = 7
A_EARTH = 6371000.0          # IUGG mean radius (m); the grid metrics are unit-sphere, so `a` only scales Mx/My.
HPA_TO_PA = 100.0            # PS is hPa in file, declared Pa by the loader. MEASURED.
LON_OFF, LAT_OFF = 14, 29    # native index shifts (verified against the real coord arrays)

# ---- expression helpers ----------------------------------------------------
def op(o, *a):        return {"op": o, "args": list(a)}
def idx(a, *ix):      return {"op": "index", "args": [a, *ix]}
def add(*a):          return op("+", *a)
def sub(a, b):        return op("-", a, b)
def mul(*a):          return op("*", *a)
def div(a, b):        return op("/", a, b)
def agg(out, ranges, expr, args):
    return {"op": "aggregate", "output_idx": out, "args": args,
            "ranges": ranges, "expr": expr}
def rng(*names):      return {n: {"from": src} for n, src in names}

def blend(w, f0, f1):
    """(1-w)*f0 + w*f1 — the two-record bracket blend (era5 w_time pattern)."""
    return add(mul(sub(1.0, w), f0), mul(w, f1))

def ps_at(nj, ni):
    """Surface pressure in Pa at NATIVE (lat=nj, lon=ni), time-blended on the I3 cadence."""
    return mul(HPA_TO_PA, blend("w_I3", idx("F_PS", 1, nj, ni), idx("F_PS", 2, nj, ni)))

def dp_at(gk, nj, ni):
    """Hybrid pressure thickness dp = dA[k] + dB[k]*PS at NATIVE (nj, ni)."""
    return add(idx("dA", gk), mul(idx("dB", gk), ps_at(nj, ni)))

def wind_at(F, gk, nj, ni):
    """A3dyn wind at NATIVE (lev=gk, lat=nj, lon=ni), time-blended on the A3 cadence."""
    return blend("w_A3", idx(F, 1, gk, nj, ni), idx(F, 2, gk, nj, ni))

# ---- load the verified Stage-B model ---------------------------------------
d = json.load(open(SRC), object_pairs_hook=collections.OrderedDict)
M = d["models"]["Transport3D"]
V = M["variables"]

d["metadata"]["name"] = "ReSEACT3D"
d["metadata"]["description"] = (
    "ReSEACT 3-D transport on a 7x7x7 NATIVE slice of the GEOS-FP 4x5 grid centred over the "
    "central US, driven by real GEOS-FP meteorology. NO REGRIDDING: the local grid IS the native "
    "grid (lon cells 15..21 -> centres -110..-80; lat points 30..36 -> 26..50; levels 1..7, "
    "k=1 = surface), so the whole conservative-overlap chain collapses to an index shift.\n\n"
    "Transport is the monotone flux-form PPM stack imported BY REFERENCE from "
    "EarthSciDiscretizations: regional open-wall inflow BCs on lon/lat, and the HYBRID "
    "sigma-pressure massform vertical (dp = dA[k] + dB[k]*PS(i,j,t)) on lev. The horizontal and "
    "vertical rule stack is byte-identical to the verified prototypes/transport_3d/transport_3d.esm; "
    "only the FORCING differs — the analytic test winds and constant PS are replaced by GEOS-FP "
    "U/V/OMEGA/PS, each supplied by an EarthSciIO discrete_provider returning the TWO records "
    "bracketing the solver time and blended continuously in time (records_per_sample=2 + w_time, "
    "the era5 pattern).\n\n"
    "GEOS-FP's collections are not co-phased, so three independent weights are used: w_I3 "
    "(PS, 3-hourly instantaneous at 00:00Z), w_A3 (U/V/OMEGA, 3-hourly means at 01:30Z) and w_A1 "
    "(PBLH, hourly means at 00:30Z).\n\n"
    "Face fluxes follow the metric the ESD divergence stencils imply on the unit sphere: "
    "Mx = dp_face*U_face/a and My = dp_face*V_face/a (Pa/s, matching m in Pa), with the A-grid "
    "GEOS-FP winds averaged to faces. Mz = -OMEGA (OMEGA is dP/dt, +ve DOWNWARD; Mz is +ve UPWARD "
    "at each cell's LOWER edge).\n\n"
    "CAVEATS. (1) PS is read in hPa and scaled by 100 — the loader declares Pa but EarthSciIO "
    "returns raw file values. (2) The lev no-flux BC is baked into the stencils (they never read "
    "Mz[1] or Mz[NLEV+1]), so the 7-level slab is CLOSED at its ~906 hPa top even though real air "
    "flows through it — a truncated-column artifact, negligible over the 1 s first-light window but "
    "not for long runs. (3) The two end lat rows are half-width (lat_edge clamps them) while native "
    "lat cells are a full 4 deg. (4) Using OMEGA directly means m drifts from dp: the mass fluxes "
    "are not the ones that reproduce the observed surface-pressure tendency (Jockel et al. 2001). "
    "CWC still holds EXACTLY — both continuity and tracer use the same fluxes — which is what the "
    "q == 1 gate tests.")

# ---- 1. index sets for the native GEOS-FP arrays ---------------------------
# The provider does NO spatial subsetting: it returns the FULL native field with a
# leading size-2 bracket axis, and the model indexes into it.
d["index_sets"] = collections.OrderedDict([
    ("gf_t",   {"kind": "interval", "size": 2}),
    ("gf_lev", {"kind": "interval", "size": 72}),
    ("gf_lat", {"kind": "interval", "size": 46}),
    ("gf_lon", {"kind": "interval", "size": 72}),
])

# ---- 2. loader-field parameters + cadence weights ---------------------------
def P(units, shape, desc):
    v = collections.OrderedDict([("type", "parameter"), ("units", units)])
    if shape: v["shape"] = shape
    v["default"] = 0.0
    v["description"] = desc
    return v

new_params = collections.OrderedDict([
    ("F_PS", P("hPa", ["gf_t", "gf_lat", "gf_lon"],
        "GEOS-FP I3 surface pressure on the NATIVE 4x5 grid, the two records bracketing the "
        "current time: [time,lat,lon] = [gf_t,gf_lat,gf_lon]. Coupled from GEOSFP.GEOSFP_I3.PS. "
        "UNITS ARE hPa, not the Pa the loader declares — the file says hPa and EarthSciIO returns "
        "raw values, so every consumer here scales by 100.")),
    ("F_U", P("m/s", ["gf_t", "gf_lev", "gf_lat", "gf_lon"],
        "GEOS-FP A3dyn eastward wind, two bracketing records, NATIVE [time,lev,lat,lon]. Coupled "
        "from GEOSFP.GEOSFP_A3dyn.U. CELL-CENTRED (A-grid) despite the loader's 'x-edge-staggered' "
        "note — the archived GEOS-Chem files are A-grid — so it is averaged to lon faces here.")),
    ("F_V", P("m/s", ["gf_t", "gf_lev", "gf_lat", "gf_lon"],
        "GEOS-FP A3dyn northward wind, two bracketing records, NATIVE [time,lev,lat,lon]. Coupled "
        "from GEOSFP.GEOSFP_A3dyn.V. Cell-centred; averaged to lat faces here.")),
    ("F_OM", P("Pa/s", ["gf_t", "gf_lev", "gf_lat", "gf_lon"],
        "GEOS-FP A3dyn vertical pressure velocity OMEGA = dP/dt, +ve DOWNWARD, two bracketing "
        "records, NATIVE [time,lev,lat,lon]. Coupled from GEOSFP.GEOSFP_A3dyn.OMEGA. Cell-centred; "
        "averaged to lev edges and NEGATED to give Mz (+ve upward).")),
    ("F_PBLH", P("m", ["gf_t", "gf_lat", "gf_lon"],
        "GEOS-FP A1 planetary boundary layer height, two bracketing records, NATIVE [time,lat,lon]. "
        "Coupled from GEOSFP.GEOSFP_A1.PBLH. Carried for the Kzz parameterization.")),
    ("w_I3", P("1", None,
        "Time-interpolation weight for the I3 cadence (PS): 3-hourly INSTANTANEOUS records at "
        "00:00, 03:00, ... UTC. Coupled from GEOSFP.w_time_I3.")),
    ("w_A3", P("1", None,
        "Time-interpolation weight for the A3 cadence (U/V/OMEGA): 3-hourly time-AVERAGED records "
        "centred 01:30, 04:30, ... UTC. Coupled from GEOSFP.w_time_A3. Distinct from w_I3 because "
        "GEOS-FP's collections are not co-phased.")),
    ("w_A1", P("1", None,
        "Time-interpolation weight for the A1 cadence (PBLH): hourly time-AVERAGED records centred "
        "00:30, 01:30, ... UTC. Coupled from GEOSFP.w_time_A1.")),
])
for k, v in new_params.items():
    V[k] = v

# ---- 3. PS: native slice, hPa -> Pa, time-blended ---------------------------
V["PS"]["units"] = "Pa"
V["PS"]["description"] = (
    "Surface pressure (Pa) on the local grid: the GEOS-FP I3 field at NATIVE (lat=29+j, lon=14+i), "
    "blended across its two bracketing records with w_I3 and scaled hPa->Pa. Replaces the Stage-B "
    "constant 101325. Drives dp = dA[k] + dB[k]*PS, so the hybrid vertical breathes with the real "
    "surface pressure.")
V["PS"]["expression"] = agg(
    ["gi", "gj"], rng(("gi", "lon"), ("gj", "lat")),
    ps_at(add(LAT_OFF, "gj"), add(LON_OFF, "gi")), ["F_PS"])

# ---- 4. PBLH: native slice, time-blended (for Kzz) --------------------------
V["PBLH"] = collections.OrderedDict([
    ("type", "observed"), ("units", "m"), ("shape", ["lon", "lat"]),
    ("description",
     "Planetary boundary layer height (m) on the local grid: the GEOS-FP A1 field at NATIVE "
     "(lat=29+j, lon=14+i), blended across its two bracketing records with w_A1. Carried for the "
     "Kzz parameterization; not yet consumed by the transport RHS."),
    ("expression", agg(["gi", "gj"], rng(("gi", "lon"), ("gj", "lat")),
        blend("w_A1", idx("F_PBLH", 1, add(LAT_OFF, "gj"), add(LON_OFF, "gi")),
                      idx("F_PBLH", 2, add(LAT_OFF, "gj"), add(LON_OFF, "gi"))), ["F_PBLH"])),
])

# ---- 5. Mx: zonal air-mass flux at lon faces --------------------------------
# Face ie (ie = 1..NLON+1) lies exactly on the midpoint of native lon cells
# (13+ie) and (14+ie) — the local cell edges land on native cell midpoints — so the
# A-grid wind and the cell thickness are both simple 2-point averages. No clamping:
# lon is cell-based and the halo cells 14 and 22 exist in the native array.
nL, nR = add(13, "gie"), add(14, "gie")
nj    = add(LAT_OFF, "gj")
V["Mx"]["units"] = "Pa/s"
V["Mx"]["description"] = (
    "Zonal face-normal AIR-MASS flux (Pa/s) at each lon face: Mx = dp_face * U_face / a, the metric "
    "the ESD divergence stencil implies on the unit sphere (cell area dlon_rad*dS_lat, zonal face "
    "length dphi_lat). dp_face and U_face are 2-point averages of the two native cells flanking the "
    "face (13+ie, 14+ie) — exact, because each local lon cell has the same footprint as native cell "
    "14+i and the local edges sit on native cell midpoints. Replaces the Stage-B analytic wind. The "
    "west/east wall faces (ie=1, ie=NLON+1) are GENUINE faces on an open domain, and their halo "
    "cells (native 14 and 22) exist in the native array — no extrapolation needed.")
V["Mx"]["expression"] = agg(
    ["gie", "gj", "gk"], rng(("gie", "lon_nodes"), ("gj", "lat"), ("gk", "lev")),
    div(mul(mul(0.5, add(dp_at("gk", nj, nL), dp_at("gk", nj, nR))),
            mul(0.5, add(wind_at("F_U", "gk", nj, nL), wind_at("F_U", "gk", nj, nR)))),
        A_EARTH),
    ["F_PS", "F_U", "dA", "dB"])

# ---- 6. My: meridional air-mass flux at lat faces ---------------------------
# lat is POINT-based with CLAMPED end edges (lat_edge): face je=1 sits ON lat point 1
# and face je=NLAT+1 sits ON lat point NLAT, while faces 2..NLAT bisect points je-1,je.
# The clamp is expressed as an index clamp, which the evaluator supports:
#   jL = 28 + max(je,2)   jR = 29 + min(je,NLAT)
# je=1 -> (30,30) = point 1;  je=8 -> (36,36) = point 7;  je=4 -> (32,33) bisects points 3,4.
jL = add(LAT_OFF - 1, op("max", "gje", 2))
jR = add(LAT_OFF,     op("min", "gje", NLAT))
ni = add(LON_OFF, "gi")
V["My"]["units"] = "Pa/s"
V["My"]["description"] = (
    "Meridional face-normal AIR-MASS flux (Pa/s) at each lat face: My = dp_face * V_face / a, the "
    "metric the ESD divergence stencil implies on the unit sphere (meridional face length "
    "dlon_rad*cos(phi), supplied by the stencil's coslat_e factor). Because lat is POINT-based with "
    "lat_edge CLAMPING the two end edges onto the end points, the flanking native points are "
    "jL = 28+max(je,2) and jR = 29+min(je,7): the walls (je=1, je=8) collapse to the single native "
    "point that the face sits on, and interior faces bisect their two neighbours. Replaces the "
    "Stage-B analytic wind.")
V["My"]["expression"] = agg(
    ["gi", "gje", "gk"], rng(("gi", "lon"), ("gje", "lat_nodes"), ("gk", "lev")),
    div(mul(mul(0.5, add(dp_at("gk", jL, ni), dp_at("gk", jR, ni))),
            mul(0.5, add(wind_at("F_V", "gk", jL, ni), wind_at("F_V", "gk", jR, ni)))),
        A_EARTH),
    ["F_PS", "F_V", "dA", "dB"])

# ---- 7. Mz: vertical air-mass flux at lev edges -----------------------------
# Mz[k] = upward mass flux at the LOWER edge of cell k = -OMEGA there.
# OMEGA is cell-centred, so the edge value bisects cells k-1 and k; the clamp
# kL = max(ke-1,1), kR = min(ke,NLEV) keeps the gather in range at the walls. The
# wall values are IRRELEVANT: facediv_lev_mass_nf_k1/_kN hardcode Mz[1] = Mz[NLEV+1] = 0
# and never read them (that IS the no-flux BC), so the slab is closed at its top.
kL = op("max", sub("gke", 1), 1)
kR = op("min", "gke", NLEV)
V["Mz"]["units"] = "Pa/s"
V["Mz"]["description"] = (
    "Vertical face-normal AIR-MASS flux (Pa/s) at each cell's LOWER edge, +ve UPWARD: Mz = -OMEGA "
    "(GEOS-FP OMEGA is dP/dt, +ve DOWNWARD). OMEGA is CELL-CENTRED in the archived files, so the "
    "edge value averages the two flanking cells, clamped at the walls. The values at ke=1 and "
    "ke=NLEV+1 are NEVER READ: facediv_lev_mass_nf_k1/_kN hardcode the no-flux wall to 0, which is "
    "what makes the 7-level slab closed at its ~906 hPa top. Replaces the Stage-B analytic profile.")
V["Mz"]["expression"] = agg(
    ["gi", "gj", "gke"], rng(("gi", "lon"), ("gj", "lat"), ("gke", "lev_nodes")),
    op("-", mul(0.5, add(wind_at("F_OM", kL, add(LAT_OFF, "gj"), add(LON_OFF, "gi")),
                         wind_at("F_OM", kR, add(LAT_OFF, "gj"), add(LON_OFF, "gi"))))),
    ["F_OM"])

# ---- 8. mount GEOSFP + wire the coupling edges ------------------------------
models = collections.OrderedDict()
models["GEOSFP"] = collections.OrderedDict([("ref", GEOSFP_REF)])
models["Transport3D"] = M
d["models"] = models

def vm(frm, to, desc):
    return collections.OrderedDict([("type", "variable_map"), ("from", frm), ("to", to),
                                    ("transform", "param_to_var"), ("description", desc)])

d["coupling"] = [
    vm("GEOSFP.GEOSFP_I3.PS", "Transport3D.F_PS",
       "GEOS-FP I3 surface pressure (hPa in file) -> the native-slice PS. Note the subsystem key is "
       "GEOSFP_I3, not I3."),
    vm("GEOSFP.GEOSFP_A3dyn.U", "Transport3D.F_U", "GEOS-FP A3dyn eastward wind -> the zonal face flux Mx."),
    vm("GEOSFP.GEOSFP_A3dyn.V", "Transport3D.F_V", "GEOS-FP A3dyn northward wind -> the meridional face flux My."),
    vm("GEOSFP.GEOSFP_A3dyn.OMEGA", "Transport3D.F_OM", "GEOS-FP A3dyn OMEGA -> the vertical face flux Mz = -OMEGA."),
    vm("GEOSFP.GEOSFP_A1.PBLH", "Transport3D.F_PBLH", "GEOS-FP A1 boundary-layer height -> carried for Kzz."),
    vm("GEOSFP.w_time_I3", "Transport3D.w_I3", "I3 cadence weight -> the PS two-record blend."),
    vm("GEOSFP.w_time_A3", "Transport3D.w_A3", "A3 cadence weight -> the U/V/OMEGA two-record blend."),
    vm("GEOSFP.w_time_A1", "Transport3D.w_A1", "A1 cadence weight -> the PBLH two-record blend."),
]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("wrote", OUT)
print("  models:", list(d["models"].keys()))
print("  coupling edges:", len(d["coupling"]))
print("  imports:", len(M["expression_template_imports"]), "(all by reference)")
