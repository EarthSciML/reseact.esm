import json, copy
SRC = "/Users/ctessum/code/earthsciml/EarthSciDiscretizations/problems/latlon3d_transport_cwc_regional_inflow.esm"
OUT = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/transport_3d/transport_3d.esm"
RULES = "../../../EarthSciDiscretizations/grids/latlon3d/rules/"

NLON=NLAT=NLEV=7
# GEOS-FP hybrid coefficients, edges k=1..8 (k=1 = surface: Ap=0, Bp=1). Source: EarthSciModels geosfp.esm P observed.
Ap=[0.0,4.804826,659.3752,1313.48,1961.311,2609.201,3257.081,3898.201]
Bp=[1.0,0.984952,0.963406,0.941865,0.920387,0.898908,0.877429,0.856018]
dA=[Ap[k]-Ap[k+1] for k in range(7)]   # dA[k] = Ap[k]-Ap[k+1], k=1..7 (0-based here)
dB=[Bp[k]-Bp[k+1] for k in range(7)]

d=json.load(open(SRC))
name=list(d['models'])[0]
m=d['models'][name]

# 1. rebind every existing import to 7x7x7, and ADD the hybrid vertical pair
B={"NLON":NLON,"NLAT":NLAT,"NLEV":NLEV}
for i in m['expression_template_imports']:
    i['bindings']=dict(B)
    i['ref']=RULES+i['ref'].split('/')[-1]
m['expression_template_imports'] += [
  {"ref":RULES+"ppm_flux_D_lev_mono_hybrid_noflux_bc.esm","bindings":dict(B)},
  {"ref":RULES+"face_flux_divergence_lev_massform_noflux_bc.esm","bindings":dict(B)},
]
v=m['variables']

# 2. CONUS-centered native GEOS-FP 4x5 slice (lands exactly on native cell boundaries)
v['lon0_deg']={"type":"parameter","units":"deg","default":-112.5,
  "description":"West EDGE of the slice. GEOS-FP 4x5 lon cell edges are -182.5+5*(i-1); -112.5 is the edge of native cell i=15, so cells i=15..21 (centers -110..-80) map to lon_local 1..7 with NO interpolation."}
v['lat0_deg']={"type":"parameter","units":"deg","default":26.0,
  "description":"Southern-most lat POINT of the slice. GEOS-FP 4x5 lat points are -90+4*(j-1); 26 is native j=30, so points j=30..36 (26..50) map to lat_local 1..7 with NO interpolation."}
v['dlon_deg']={"type":"observed","units":"deg","expression":5.0,"description":"GEOS-FP 4x5 native zonal spacing. Open west/east walls (regional, not periodic)."}
v['dlat_deg']={"type":"observed","units":"deg","expression":4.0,"description":"GEOS-FP 4x5 native meridional spacing. Open south/north walls (regional, not polar)."}

def agg(idx, ranges, args, expr):
    return {"op":"aggregate","output_idx":idx,"ranges":{k:{"from":r} for k,r in ranges.items()},"args":args,"expr":expr}
def ix(a,*s): return {"op":"index","args":[a,*s]}
def const(vals): return {"op":"const","args":[],"value":vals}

# 3a. Hybrid coefficient differences as SHAPED PARAMETERS, supplied via const_arrays at simulate().
#     They cannot be inline `const` arrays: index(const([...]), k) is rejected as a "non-scalar const
#     op outside an array-consuming position". Routing them through interp.linear WOULD work (the
#     fastjx pattern) but would drag interp.* into the transport RHS and forfeit autodiff.
v['dA']={"type":"parameter","units":"Pa","shape":["lev"],"default":0.0,
  "description":"dA[k] = Ap[k]-Ap[k+1] from the native GEOS-FP hybrid table at integer edges k=1..8. Supplied via const_arrays."}
v['dB']={"type":"parameter","units":"1","shape":["lev"],"default":0.0,
  "description":"dB[k] = Bp[k]-Bp[k+1] from the native GEOS-FP hybrid table at integer edges k=1..8. Supplied via const_arrays."}

# 3. PS (constant for this stage; replaced by the GEOS-FP slice in Stage A)
PS0 = 101325.0
v['PS']={"type":"observed","units":"Pa","shape":["lon","lat"],
  "description":"Surface pressure. STAGE B: constant 101325 Pa. Stage A replaces this with the GEOS-FP I3.PS native slice (i=14+lon, j=29+lat).",
  "expression":agg(["gi","gj"],{"gi":"lon","gj":"lat"},[],PS0)}

# 4. dp[lon,lat,lev] -- the hybrid stencil's free name. dp = dA[k] + dB[k]*PS(i,j,t); 3-D and
#    time-varying, terrain-following. Ap/Bp are read at INTEGER k, so no interp.* enters the RHS.
v['dp']={"type":"observed","units":"Pa","shape":["lon","lat","lev"],
  "description":"Physical pressure thickness dp = dA[k] + dB[k]*Ps(i,j,t) on the GEOS-FP hybrid sigma-pressure grid; the CW84 parabola is built on these (ppmflux_D_lev_hyb_*). dA/dB are differences of the native GEOS-FP Ap(Pa)/Bp tables at integer edges k=1..8 -- read directly, NOT via interp.linear, which keeps the transport RHS pure arithmetic and therefore autodiff-capable.",
  "expression":agg(["gi","gj","gk"],{"gi":"lon","gj":"lat","gk":"lev"},["PS","dA","dB"],
      {"op":"+","args":[ix("dA","gk"),{"op":"*","args":[ix("dB","gk"),ix("PS","gi","gj")]}]})}

# 5. INITIAL CONDITIONS: keep the regional exemplar's own analytic m/mq ICs, unchanged.
#    Rationale: this stage gates COMPOSITION + CWC, and CWC is insensitive to the scale of m --
#    q = mq/m stays 1 for ANY m(0) == mq(0) driven by the same fluxes, and `dp` only sets the PPM
#    reconstruction geometry (at q == 1 every CW84 slope is exactly 0.0). An ic() RHS additionally
#    may NOT reference an observed NOR an inline const array -- it must be a const-array field, a
#    constant, or a per-cell coordinate expression -- and the exemplar's ICs already satisfy that.
#    Stage A supplies the physical air mass m(0) = dp via const_arrays/providers.
# 6. Mz[lon,lat,lev_nodes]: vertical air-mass flux at the LOWER edge of each cell, +ve UPWARD
#    (ESD convention: pe increases upward, k=1 = surface). Analytic polynomial in s=(k-1)/NLEV that
#    vanishes EXACTLY at both walls (s=0,1) and reverses sign mid-column -- same shape as the
#    verified global 3-D PPM exemplar. Stage A replaces this with -OMEGA from the GEOS-FP slice.
s = {"op":"/","args":[{"op":"-","args":["ck",1]},float(NLEV)]}
v['Mz']={"type":"observed","units":"Pa/s","shape":["lon","lat","lev_nodes"],
  "description":"Vertical face-normal air-mass flux at the LOWER edge of each cell, positive UPWARD. Vanishes exactly at k=1 (surface) and k=NLEV+1 (lid), consistent with the no-flux hybrid BC. STAGE B: analytic. Stage A: Mz = -OMEGA (OMEGA is dP/dt, +ve DOWNWARD, and is already lev-interface staggered in GEOS-FP).",
  "expression":agg(["ci","cj","ck"],{"ci":"lon","cj":"lat","ck":"lev_nodes"},[],
     {"op":"*","args":[0.05,{"op":"*","args":[4.0,s,{"op":"-","args":[1.0,s]},{"op":"-","args":[1.0,{"op":"*","args":[2.0,s]}]}]}]})}

# 7. add the vertical term to the three tendency equations
def Dlev(a): return {"op":"D","args":[a],"wrt":"lev"}
for e in m['equations']:
    l=e['lhs']
    if l.get('op')!='D' or l.get('wrt')!='t': continue
    tgt=l['args'][0]
    inner = Dlev("Mz") if tgt=='m' else Dlev({"op":"*","args":["Mz","q"]})
    if tgt=='dev':
        # dev = (flux-form tracer divergence) - (mass divergence); add BOTH lev terms
        e['rhs']['args'][0]['args'][0]['args'].append(Dlev({"op":"*","args":["Mz","q"]}))
        e['rhs']['args'][0]['args'][1]['args'].append(Dlev("Mz"))
    else:
        e['rhs']['args'][0]['args'].append(inner)

d['metadata']['name']="ReSEACT_Transport3D"
d['metadata']['description']=("ReSEACT 3-D transport foundation: monotone PPM on the latlon3d grid over a 7x7x7 "
  "CONUS-centered NATIVE slice of GEOS-FP 4x5 (no regridding). Horizontal = regional open-wall inflow BCs "
  "(ppm_flux_D_{lon,lat}_mono_inflow_bc + face_flux_divergence_{lon,lat}_open_bc); vertical = hybrid "
  "sigma-pressure massform (ppm_flux_D_lev_mono_hybrid_noflux_bc + face_flux_divergence_lev_massform_noflux_bc). "
  "All six rules imported BY REFERENCE. Gate: CWC -- q=mq/m must stay EXACTLY 1 and dev EXACTLY 0.")
d['models']={"Transport3D":m}
json.dump(d, open(OUT,'w'), indent=1)
print("wrote", OUT)
print("dA =", [round(x,4) for x in dA])
print("dB =", [round(x,6) for x in dB])
print("dp @ PS=101325 (Pa):", [round(dA[k]+dB[k]*101325.0,1) for k in range(7)])
print("column top pressure (Pa):", round(101325.0-sum(dA[k]+dB[k]*101325.0 for k in range(7)),1))
print("imports:", len(m['expression_template_imports']))
json.dump({"dA":dA,"dB":dB}, open("hybrid_coefs.json","w"))
