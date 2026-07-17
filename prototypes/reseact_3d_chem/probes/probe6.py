"""Stage-C GATE: does `operator_compose` + lifting:"pointwise" lift a REACTION SYSTEM's
scalar species onto the grid when the spatial contribution comes from the ESD PPM rules?

Cheap: 2 toy species + the Stage-B ANALYTIC winds (the GEOS-FP forcing is verified
separately by probe3), so this isolates the ONE unknown -- the lift.
Built by patching the verified transport_3d.esm so the PPM stack stays byte-identical.
"""
import json, collections
SRC = "/Users/ctessum/code/earthsciml/reseact.esm/prototypes/transport_3d/transport_3d.esm"
d = json.load(open(SRC), object_pairs_hook=collections.OrderedDict)
T = d["models"]["Transport3D"]; V = T["variables"]
d["metadata"]["name"] = "probe_pointwise_lift"

def op(o,*a): return {"op":o,"args":list(a)}
def D(x,wrt): return {"op":"D","args":[x],"wrt":wrt}

# Drop the CWC diagnostic pair; keep `m` (air mass) + the PPM/facediv stack.
for k in ["mq","dev","q"]: V.pop(k, None)
T["equations"] = [e for e in T["equations"]
                  if json.dumps(e["lhs"]).find('"mq"')<0 and json.dumps(e["lhs"]).find('"dev"')<0]
T.pop("tests", None)

# A 2-species toy reaction system: A -> B at rate k. Scalar species (0-D), as authored.
d["reaction_systems"] = {"Chem": {
   "species": {"A": {"units":"ppb","default":10.0,"description":"toy tracer A"},
               "B": {"units":"ppb","default": 0.0,"description":"toy tracer B"}},
   "parameters": {"k": {"units":"1/s","default":0.01,"description":"A->B rate"}},
   "reactions": [{"id":"R1","name":"A_to_B",
                  "substrates":[{"species":"A","stoichiometry":1}],
                  "products":[{"species":"B","stoichiometry":1}],
                  "rate":"k"}],
}}

# The Transport model supplies each species' advection contribution, referencing the
# reaction system's species by SCOPED name -- the advection_reaction_loaded_ic_bc.esm
# pattern, with the ESD PPM rules (which emit `makearray`) instead of `grad`.
# Mixing-ratio CWC form: dq/dt = [ -div(M q) + q*div(M) ] / m
def Dbc(x, lo, hi, wrt):
    """A lateral flux divergence carrying its inflow halo as the 2nd/3rd OPERANDS.

    ESD's regional-inflow rule binds `qbc_*` as rule PARAMS, so a 1-operand
    `D(Mx*q, wrt:lon)` matches NOTHING -- facediv's `where` guard on the
    staggered wind rejects the product -- and dies with `unlowered_operator` at
    build, which `validate` CANNOT see. `lev` is no-flux and stays 1-operand.
    """
    return {"op":"D","args":[x, lo, hi],"wrt":wrt}

def adv(sp):
    s = f"Chem.{sp}"
    divMq = op("+", Dbc(op("*","Mx",s),"qbc_w","qbc_e","lon"),
                    Dbc(op("*","My",s),"qbc_s","qbc_n","lat"),
                    D(op("*","Mz",s),"lev"))
    divM  = op("+", D("Mx","lon"), D("My","lat"), D("Mz","lev"))
    return {"_comment": f"d({sp})/dt advection contribution, CWC mixing-ratio form; "
                        f"merged with the reaction tendency by operator_compose.",
            "lhs": D(s,"t"),
            "rhs": op("/", op("+", op("-", divMq), op("*", s, divM)), "m")}
T["equations"] += [adv("A"), adv("B")]

d["coupling"] = [{"type":"operator_compose","systems":["Chem","Transport3D"],
                  "lifting":"pointwise",
                  "description":"Lift the 0-D toy chemistry onto the 7x7x7 grid and add per-species "
                                "PPM advection to each species tendency."}]
# Absolute rule refs: this probe lives outside prototypes/, so the ../../.. refs would miss.
for imp in T.get("expression_template_imports", []):
    imp["ref"] = imp["ref"].replace("../../../EarthSciDiscretizations",
                                    "/Users/ctessum/code/earthsciml/EarthSciDiscretizations")
json.dump(d, open("probe6.esm","w"), indent=1)
print("wrote probe6.esm: 2 toy species + PPM advection + operator_compose/pointwise")
