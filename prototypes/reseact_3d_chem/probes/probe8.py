"""probe7 + a Pc-style observed: an aggregate over [gi,gj,gk] that INDEXES ANOTHER
aggregate (PS, over [gi,gj]) reusing the same loop names. This is the one structure C1
has that probe7 does not, and `gj` is exactly the name the C1 error reports."""
import json, collections
exec(open("probe7.py").read().split('json.dump(d, open("probe7.esm"')[0])

# A PS-like [lon,lat] aggregate...
V["PSx"] = collections.OrderedDict([
    ("type","observed"), ("units","Pa"), ("shape",["lon","lat"]),
    ("description","surface-pressure-like [lon,lat] aggregate"),
    ("expression", agg2(["gi","gj"], [("gi","lon"),("gj","lat")],
        {"op":"+","args":[101325.0,{"op":"*","args":[10.0,"gi"]}]}, []))])
# ...indexed from inside a [gi,gj,gk] aggregate, reusing gi/gj — exactly C1's Pc.
V["Pcx"] = collections.OrderedDict([
    ("type","observed"), ("units","Pa"), ("shape",["lon","lat","lev"]),
    ("description","pressure-like: indexes the PSx aggregate from within its own aggregate"),
    ("expression", agg2(["gi","gj","gk"], [("gi","lon"),("gj","lat"),("gk","lev")],
        {"op":"*","args":[{"op":"index","args":["PSx","gi","gj"]},
                          {"op":"/","args":["gk",7.0]}]}, ["PSx"]))])

d["reaction_systems"]["Chem"]["parameters"]["P"] = {
    "units":"Pa","default":101325.0,"description":"pressure (coupled from Transport3D.Pcx)"}
d["reaction_systems"]["Chem"]["reactions"][0]["rate"] = {"op":"*","args":[
    "k", {"op":"/","args":["T",280.0]}, {"op":"/","args":["P",101325.0]}]}
d["coupling"].append(collections.OrderedDict([
    ("type","variable_map"),("from","Transport3D.Pcx"),("to","Chem.P"),
    ("transform","param_to_var"),("description","Per-cell pressure -> the scalar P param.")]))

for imp in T.get("expression_template_imports", []):
    imp["ref"] = imp["ref"].replace("../../../EarthSciDiscretizations",
                                    "/Users/ctessum/code/earthsciml/EarthSciDiscretizations")
json.dump(d, open("probe8.esm","w"), indent=1)
print("wrote probe8.esm (probe7 + a nested-aggregate Pc-style observed)")
