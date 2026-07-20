"""Probe the Mx/My/Mz forcing expressions ALONE (no PPM rules -> fast build), so
their values can be checked against an independent computation from the netCDF."""
import json, collections
import os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reseact_3d.esm")
d = json.load(open(SRC), object_pairs_hook=collections.OrderedDict)
M = d["models"]["Transport3D"]; V = M["variables"]

# Keep the forcing observeds; drop everything PPM so the build is seconds, not minutes.
M.pop("expression_template_imports", None)
M.pop("tests", None)
for k in ["m","mq","dev","q","qbc_w","qbc_e","qbc_s","qbc_n",
          "phie","coslat_e","dS_lat","dphi_lat"]:
    V.pop(k, None)

def idx(a,*ix): return {"op":"index","args":[a,*ix]}
# Probe states: dy/dt = <forcing at a fixed cell>, so y(1s)-y(0s) reads the value out.
PROBES = [
    ("pMx", "Mx", [3, 4, 1]),   # lon face 3, lat 4, lev 1
    ("pMy", "My", [3, 4, 1]),   # lon 3, lat face 4, lev 1
    ("pMz", "Mz", [3, 4, 2]),   # lon 3, lat 4, lev edge 2 (edge 1 is the ignored wall)
    ("pPS", "PS", [3, 4]),
    ("pdp", "dp", [3, 4, 1]),
    ("pBL", "PBLH", [3, 4]),
    ("pMxw","Mx", [1, 4, 1]),   # the WEST WALL face -> exercises the halo native cell 14
    ("pMys","My", [3, 1, 1]),   # the SOUTH WALL face -> exercises the lat index clamp
]
V["y"] = {"type":"state","units":"1","shape":["probe"],"description":"probe accumulators"}
d["index_sets"]["probe"] = {"kind":"interval","size":len(PROBES)}
# The grid index sets normally arrive with the (stripped) rule imports; declare them here.
for nm,sz in [("lon",7),("lat",7),("lev",7),("lon_nodes",8),("lat_nodes",8),("lev_nodes",8)]:
    d["index_sets"][nm] = {"kind":"interval","size":sz}

# One arm per probe, selected by equality on the probe index.
expr = 0.0
for n,(nm,var,ix) in enumerate(PROBES, start=1):
    expr = {"op":"+","args":[expr, {"op":"*","args":[
        {"op":"ifelse","args":[{"op":"==","args":["p",n]},1.0,0.0]}, idx(var,*ix)]}]}
M["equations"] = [{"lhs":{"op":"aggregate","output_idx":["p"],"args":[],
                          "ranges":{"p":{"from":"probe"}},
                          "expr":{"op":"D","args":[idx("y","p")],"wrt":"t"}},
                   "rhs":{"op":"aggregate","output_idx":["p"],
                          "args":[v for _,v,_ in PROBES],
                          "ranges":{"p":{"from":"probe"}}, "expr":expr}}]
d["models"]["GEOSFP"]["ref"] = "../../../EarthSciModels/components/earthsci_data/geosfp.esm"
json.dump(d, open("probe3.esm","w"), indent=1)
print("wrote probe3.esm with", len(PROBES), "probes:", [p[0] for p in PROBES])
