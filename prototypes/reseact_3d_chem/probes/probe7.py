"""Bisect the C1 failure (E_TREEWALK_UNBOUND_LOOP_VAR: gj) on the FAST probe6 base.
probe6 (2 toy species + PPM + pointwise lift) WORKS. The only new thing in C1 is
coupling a grid-shaped observed into the reaction system's SCALAR param. Add exactly
that: a `Tc[lon,lat,lev]` observed -> the toy system's scalar `T` param, with the rate
depending on T so it actually enters the lifted equation."""
import json, collections
exec(open("probe6.py").read().split('json.dump(d, open("probe6.esm"')[0])  # reuse probe6's build

def agg2(out, rngs, expr, args):
    return {"op":"aggregate","output_idx":out,"args":args,
            "ranges":{n:{"from":s} for n,s in rngs},"expr":expr}

# A grid-shaped temperature observed, built the same way C1's Tc is (an aggregate
# over gi/gj/gk — the loop names the C1 error names).
V["Tc"] = collections.OrderedDict([
    ("type","observed"), ("units","K"), ("shape",["lon","lat","lev"]),
    ("description","Gridded temperature; coupled into the toy system's scalar T param."),
    ("expression", agg2(["gi","gj","gk"], [("gi","lon"),("gj","lat"),("gk","lev")],
        {"op":"+","args":[280.0, {"op":"*","args":[1.0,"gi"]}]}, []))])

# Toy system gains a scalar T param, and the rate uses it.
d["reaction_systems"]["Chem"]["parameters"]["T"] = {
    "units":"K","default":280.0,"description":"temperature (coupled from Transport3D.Tc)"}
d["reaction_systems"]["Chem"]["reactions"][0]["rate"] = {"op":"*","args":["k",{"op":"/","args":["T",280.0]}]}

d["coupling"].append(collections.OrderedDict([
    ("type","variable_map"),("from","Transport3D.Tc"),("to","Chem.T"),
    ("transform","param_to_var"),
    ("description","Per-cell temperature -> the scalar T param; param_to_var carries the grid "
                   "shape so the pointwise lift indexes it per cell.")]))

for imp in T.get("expression_template_imports", []):
    imp["ref"] = imp["ref"].replace("../../../EarthSciDiscretizations",
                                    "/Users/ctessum/code/earthsciml/EarthSciDiscretizations")
json.dump(d, open("probe7.esm","w"), indent=1)
print("wrote probe7.esm (probe6 + a gridded observed coupled into a reaction-system scalar param)")
