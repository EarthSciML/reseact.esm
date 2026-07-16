"""probe8 + the LAST difference: back the PS-like observed with a LIVE LOADER gather
(index(F_PS, 1, 29+gj, 14+gi) on a provider-backed param) instead of arithmetic.
That routes it through the _PGatherRef path, which pre-linearizes the index — and
`gj` is exactly the loop var the C1 error names."""
import json, collections
exec(open("probe8.py").read().split('json.dump(d, open("probe8.esm"')[0])

d["models"]["GEOSFP"] = collections.OrderedDict([
    ("ref","/Users/ctessum/code/earthsciml/EarthSciModels/components/earthsci_data/geosfp.esm")])
d.setdefault("index_sets", collections.OrderedDict())
for nm,sz in [("gf_t",2),("gf_lat",46),("gf_lon",72)]:
    d["index_sets"][nm] = {"kind":"interval","size":sz}
V["F_PS"] = collections.OrderedDict([
    ("type","parameter"),("units","hPa"),("shape",["gf_t","gf_lat","gf_lon"]),("default",0.0),
    ("description","GEOS-FP I3 surface pressure, two bracketing records, native [time,lat,lon].")])
V["w_I3"] = collections.OrderedDict([
    ("type","parameter"),("units","1"),("default",0.0),
    ("description","I3 time-interp weight, coupled from GEOSFP.w_time_I3.")])
# PSx now reads the LOADER (this is C1's PS, verbatim in structure).
V["PSx"]["expression"] = agg2(["gi","gj"], [("gi","lon"),("gj","lat")],
    {"op":"*","args":[100.0, {"op":"+","args":[
        {"op":"*","args":[{"op":"-","args":[1.0,"w_I3"]},
            {"op":"index","args":["F_PS",1,{"op":"+","args":[29,"gj"]},{"op":"+","args":[14,"gi"]}]}]},
        {"op":"*","args":["w_I3",
            {"op":"index","args":["F_PS",2,{"op":"+","args":[29,"gj"]},{"op":"+","args":[14,"gi"]}]}]}]}]},
    ["F_PS"])
d["coupling"] += [
    collections.OrderedDict([("type","variable_map"),("from","GEOSFP.GEOSFP_I3.PS"),
        ("to","Transport3D.F_PS"),("transform","param_to_var"),
        ("description","GEOS-FP I3 surface pressure -> the native-slice gather.")]),
    collections.OrderedDict([("type","variable_map"),("from","GEOSFP.w_time_I3"),
        ("to","Transport3D.w_I3"),("transform","param_to_var"),
        ("description","I3 cadence weight -> the two-record blend.")])]
for imp in T.get("expression_template_imports", []):
    imp["ref"] = imp["ref"].replace("../../../EarthSciDiscretizations",
                                    "/Users/ctessum/code/earthsciml/EarthSciDiscretizations")
json.dump(d, open("probe9.esm","w"), indent=1)
print("wrote probe9.esm (probe8 + a LIVE LOADER gather behind the PS-like observed)")
