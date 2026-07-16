"""probe9 + a RANK-4 loader gather (F_T[1, gk, 29+gj, 14+gi]) behind the observed that
feeds the reaction system's scalar T param -- exactly C1's Tc. probes 7/8 used an
arithmetic Tc; probe9's loader gather was rank-3. This is the untested structure, and
it is valid (both toy species advect, so both lift)."""
import json, collections
exec(open("probe9.py").read().split('json.dump(d, open("probe9.esm"')[0])

d["index_sets"]["gf_lev"] = {"kind":"interval","size":72}
V["F_T"] = collections.OrderedDict([
    ("type","parameter"),("units","K"),("shape",["gf_t","gf_lev","gf_lat","gf_lon"]),("default",0.0),
    ("description","GEOS-FP I3 temperature, two bracketing records, native [time,lev,lat,lon].")])
# Tc now reads the RANK-4 loader — C1's Tc verbatim in structure.
V["Tc"]["expression"] = agg2(["gi","gj","gk"], [("gi","lon"),("gj","lat"),("gk","lev")],
    {"op":"+","args":[
        {"op":"*","args":[{"op":"-","args":[1.0,"w_I3"]},
            {"op":"index","args":["F_T",1,"gk",{"op":"+","args":[29,"gj"]},{"op":"+","args":[14,"gi"]}]}]},
        {"op":"*","args":["w_I3",
            {"op":"index","args":["F_T",2,"gk",{"op":"+","args":[29,"gj"]},{"op":"+","args":[14,"gi"]}]}]}]},
    ["F_T"])
d["coupling"].append(collections.OrderedDict([
    ("type","variable_map"),("from","GEOSFP.GEOSFP_I3.T"),("to","Transport3D.F_T"),
    ("transform","param_to_var"),("description","GEOS-FP I3 temperature -> the native-slice Tc.")]))
for imp in T.get("expression_template_imports", []):
    imp["ref"] = imp["ref"].replace("../../../EarthSciDiscretizations",
                                    "/Users/ctessum/code/earthsciml/EarthSciDiscretizations")
json.dump(d, open("probe10.esm","w"), indent=1)
print("wrote probe10.esm (probe9 + a RANK-4 loader gather feeding a lifted scalar param)")
