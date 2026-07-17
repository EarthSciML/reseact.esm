#!/usr/bin/env python3
"""Stage C1: SuperFast chemistry lifted onto the 7x7x7 GEOS-FP-driven grid.

Patches the VERIFIED Stage-A prototypes/reseact_3d/reseact_3d.esm so the whole
GEOS-FP forcing chain (verified to 4.1e-10 against the netCDF) stays byte-identical
and only chemistry is added:

  - SuperFast mounted BY REFERENCE (scalar species, as authored);
  - a per-species PPM advection contribution on the Transport model, referencing the
    species by SCOPED name, in CWC mixing-ratio form  dq/dt = [-div(M q) + q div(M)]/m;
  - operator_compose(SuperFast, Transport3D) with lifting:"pointwise", which merges the
    two tendencies and lifts each species onto the grid;
  - real GEOS-FP T and hybrid mid-layer pressure P coupled into SuperFast's scalar
    T/P params (param_to_var carries the grid shape, so the lift indexes them per cell).

Only the 12 NON-constant species are transported: O2/CH4/H2O are `constant: true`
reservoir species with no ODE, so they neither advect nor lift (correct — they stay
uniform and simply broadcast into the rate expressions).
"""
import json, collections, os

SRC = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/reseact_3d/reseact_3d.esm"
OUT = "/private/tmp/claude-501/-Users-ctessum-code-earthsciml-reseact-esm/4186e9d5-d275-4da9-b4c7-5a5f7a721d5a/scratchpad/c1mini.esm"
SF  = "../../../EarthSciModels/components/gaschem/superfast.esm"
SF_ABS = "/Users/ctessum/code/earthsciml/EarthSciModels/components/gaschem/superfast.esm"
LON_OFF, LAT_OFF = 14, 29
HPA_TO_PA = 100.0

def op(o,*a): return {"op":o,"args":list(a)}
def idx(a,*ix): return {"op":"index","args":[a,*ix]}
def D(x,wrt): return {"op":"D","args":[x],"wrt":wrt}
def agg(out, rngs, expr, args):
    return {"op":"aggregate","output_idx":out,"args":args,
            "ranges":{n:{"from":s} for n,s in rngs},"expr":expr}
def blend(w,f0,f1): return op("+", op("*", op("-",1.0,w), f0), op("*", w, f1))

d = json.load(open(SRC), object_pairs_hook=collections.OrderedDict)
T = d["models"]["Transport3D"]; V = T["variables"]
d["metadata"]["name"] = "ReSEACT3DChem"

# --- 1. drop the CWC diagnostic pair -----------------------------------------
# mq/dev are the consistency-with-continuity gate; CWC is already proven bit-exact
# in Stage A and chemistry cannot affect it (they are independent states). Dropping
# them saves 6 of 45 PPM rule instantiations (~15% of the build). `m` STAYS — the
# CWC mixing-ratio form divides by it.
for k in ["mq","dev","q"]:
    V.pop(k, None)

# --- 1b. inflow boundary fields ------------------------------------------------
# qbc_{w,e,s,n} are NOT part of the CWC pair: the regional inflow PPM rule consumes
# them as FREE NAMES inside its boundary stencils, so they must stay.
#
# LIMITATION worth knowing: `qbc_*` is a free name, not a rule PARAM, so ALL 12
# species advected by this rule SHARE ONE inflow field — there is no way to give O3
# a 40 ppb inflow and NO a 0.0004 ppb inflow. (The EarthSciAST fixture's `grad` rule
# avoids this by taking the inflow as an explicit ARG: grad(Chemistry.O3, O3_inflow).)
# So set a CLEAN-AIR inflow (0 ppb) for first light: honest and interpretable, rather
# than the exemplar's q == 1 which is meaningless as a concentration. Making the
# inflow per-species needs an ESD rule change (qbc as a param).
#
# Authored with the CORRECT output_idx: the exemplar's own qbc_w declares
# shape ["lat","lev"] but has output_idx ["gl","gl"] over only a `lev` range — a
# latent bug masked because q == 1 everywhere. Do not copy that.
for nm, ax in [("qbc_w",["lat","lev"]), ("qbc_e",["lat","lev"]),
               ("qbc_s",["lon","lev"]), ("qbc_n",["lon","lev"])]:
    V[nm] = collections.OrderedDict([
        ("type","observed"), ("units","ppb"), ("shape",list(ax)),
        ("description", f"Inflow boundary mixing ratio on the {nm[-1]} wall, over [{ax[0]},{ax[1]}]. "
                        "CLEAN AIR (0 ppb): consumed as a FREE NAME by the regional inflow PPM "
                        "stencils, so every advected species SHARES this one field — a per-species "
                        "inflow would need qbc to be a rule param."),
        ("expression", agg(["ga","gb"], [("ga",ax[0]),("gb",ax[1])], 0.0, []))])
T["equations"] = [e for e in T["equations"]
                  if not any(f'"{n}"' in json.dumps(e["lhs"]) for n in ("mq","dev"))]
T.pop("tests", None)

# --- 2. GEOS-FP temperature on the local slice --------------------------------
V["F_T"] = collections.OrderedDict([
    ("type","parameter"), ("units","K"), ("shape",["gf_t","gf_lev","gf_lat","gf_lon"]),
    ("default",0.0),
    ("description","GEOS-FP I3 temperature, two bracketing records, NATIVE [time,lev,lat,lon]. "
                   "Coupled from GEOSFP.GEOSFP_I3.T. Declared K and delivered K — unlike PS, no "
                   "unit gap here (verified against the file).")])
V["Tc"] = collections.OrderedDict([
    ("type","observed"), ("units","K"), ("shape",["lon","lat","lev"]),
    ("description","Air temperature (K) on the local grid: the GEOS-FP I3 field at NATIVE "
                   "(lev=k, lat=29+j, lon=14+i), blended across its two bracketing records with "
                   "w_I3. Coupled into SuperFast.T; param_to_var carries this grid shape onto the "
                   "scalar param so the pointwise lift indexes it per cell."),
    ("expression", agg(["gi","gj","gk"], [("gi","lon"),("gj","lat"),("gk","lev")],
        blend("w_I3", idx("F_T",1,"gk",op("+",LAT_OFF,"gj"),op("+",LON_OFF,"gi")),
                      idx("F_T",2,"gk",op("+",LAT_OFF,"gj"),op("+",LON_OFF,"gi"))), ["F_T"]))])

# --- 3. hybrid mid-layer pressure --------------------------------------------
# P_edge(k) = Ap[k] + Bp[k]*PS ; the layer's representative pressure is the mean of
# its two bounding edges. Ap/Bp are read at INTEGER k, so they are shaped parameters
# supplied via const_arrays -- `index(const([...]), k)` is rejected anywhere.
for nm, un, desc in [("Ap","Pa","GEOS-FP hybrid sigma-pressure Ap coefficients at the NLEV+1 layer "
                                "edges (Pa). k=1 is the surface (Ap[1]=0, Bp[1]=1 => P_edge(1)=PS). "
                                "Supplied via const_arrays."),
                     ("Bp","1", "GEOS-FP hybrid sigma-pressure Bp coefficients at the NLEV+1 layer "
                                "edges (dimensionless). Supplied via const_arrays.")]:
    V[nm] = collections.OrderedDict([("type","parameter"),("units",un),("shape",["lev_nodes"]),
                                     ("default",0.0),("description",desc)])
def pedge(k): return op("+", idx("Ap",k), op("*", idx("Bp",k), idx("PS","gi","gj")))
V["Pc"] = collections.OrderedDict([
    ("type","observed"), ("units","Pa"), ("shape",["lon","lat","lev"]),
    ("description","Air pressure (Pa) at each layer's mid-point: the mean of its two hybrid "
                   "sigma-pressure edges, P_edge(k) = Ap[k] + Bp[k]*PS(i,j,t). Breathes with the "
                   "real surface pressure. Coupled into SuperFast.P."),
    ("expression", agg(["gi","gj","gk"], [("gi","lon"),("gj","lat"),("gk","lev")],
        op("*", 0.5, op("+", pedge("gk"), pedge(op("+","gk",1)))), ["Ap","Bp","PS"]))])

# --- 4. mount SuperFast + per-species PPM advection ---------------------------
sf = json.load(open(SF_ABS))
species = sf["reaction_systems"]["SuperFast"]["species"]
advected = [s for s,v in species.items() if not v.get("constant", False)][:2]  # MINI: 2 species to bisect fast
d["reaction_systems"] = {"SuperFast": {"ref": SF}}

def Dbc(x, lo, hi, wrt):
    """Lateral flux divergence carrying its inflow halo as operands 2 and 3.

    ESD binds `qbc_*` as rule PARAMS, so a 1-operand `D(Mx*q, wrt:lon)` matches
    nothing and dies with `unlowered_operator` -- which would make this probe
    fail for a reason that has nothing to do with what it is bisecting.
    """
    return {"op":"D","args":[x, lo, hi],"wrt":wrt}

def adv_eq(sp):
    s = f"SuperFast.{sp}"
    divMq = op("+", Dbc(op("*","Mx",s),"qbc_w","qbc_e","lon"),
                    Dbc(op("*","My",s),"qbc_s","qbc_n","lat"),
                    D(op("*","Mz",s),"lev"))
    divM  = op("+", D("Mx","lon"), D("My","lat"), D("Mz","lev"))
    return collections.OrderedDict([
        ("_comment", f"d({sp})/dt advection contribution, flux-form PPM in CWC mixing-ratio form: "
                     f"[-div(M q) + q div(M)]/m. operator_compose merges this with {sp}'s reaction "
                     f"tendency and the pointwise lift array-ifies the result."),
        ("lhs", D(s,"t")),
        ("rhs", op("/", op("+", op("-", divMq), op("*", s, divM)), "m"))])
T["equations"] += [adv_eq(s) for s in advected]

# --- 5. coupling --------------------------------------------------------------
def vm(f,t,desc):
    return collections.OrderedDict([("type","variable_map"),("from",f),("to",t),
                                    ("transform","param_to_var"),("description",desc)])
d["coupling"] = list(d["coupling"]) + [
    vm("GEOSFP.GEOSFP_I3.T", "Transport3D.F_T", "GEOS-FP I3 temperature -> the native-slice Tc."),
    vm("Transport3D.Tc", "SuperFast.T",
       "Per-cell GEOS-FP temperature -> SuperFast's scalar T param. param_to_var carries the "
       "[lon,lat,lev] shape so the pointwise lift indexes it per cell; the rate constants then "
       "vary with the real 3-D temperature field."),
    vm("Transport3D.Pc", "SuperFast.P",
       "Per-cell hybrid mid-layer pressure -> SuperFast's scalar P param."),
    collections.OrderedDict([
        ("type","operator_compose"), ("systems",["SuperFast","Transport3D"]),
        ("lifting","pointwise"),
        ("description","Lift the 0-D SuperFast mechanism onto the 7x7x7 grid and add each species' "
                       "PPM advection contribution to its reaction tendency. pointwise lifting "
                       "evaluates the network independently per cell; the merged reaction+advection "
                       "state ODEs are array-ified by the flattener's lift. Only the 12 "
                       "non-constant species lift — O2/CH4/H2O are constant:true reservoirs with "
                       "no ODE, so they broadcast into the rates instead.")]),
]

# MINI lives outside prototypes/, so make every relative ref absolute.
ROOT = "/Users/ctessum/code/earthsciml"
d["models"]["GEOSFP"]["ref"] = ROOT + "/EarthSciModels/components/earthsci_data/geosfp.esm"
d["reaction_systems"]["SuperFast"]["ref"] = SF_ABS
for imp in T.get("expression_template_imports", []):
    imp["ref"] = imp["ref"].replace("../../../EarthSciDiscretizations", ROOT + "/EarthSciDiscretizations")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,"w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False); f.write("\n")
print("wrote", OUT)
print("  advected species (", len(advected), "):", ", ".join(advected))
print("  constant (not advected):", ", ".join(s for s in species if s not in advected))
print("  PPM rule instantiations:", len(advected)*3, "+ 3 facediv =", len(advected)*3+3)
print("  coupling edges:", len(d["coupling"]))
