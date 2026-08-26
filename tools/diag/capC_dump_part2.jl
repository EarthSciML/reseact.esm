#!/usr/bin/env julia
# Dump the split CHEMISTRY doc (part 2) as JSON so its geometry dependence can
# be analysed offline, without paying the 80 s load each time.
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "_env.jl"))
using EarthSciAST, JSON3
const EA = EarthSciAST
include(joinpath(REPO, "prototypes", "reseact_3d_chem", "split_common.jl"))
using EarthSciASTSplitter: split_system
using Printf
NLON = parse(Int, get(ENV, "RESEACT_NLON", "6")); NLAT = parse(Int, get(ENV, "RESEACT_NLAT", "6"))
NLEV = parse(Int, get(ENV, "RESEACT_NLEV", "8"))
LON0 = parse(Int, get(ENV, "RESEACT_LON0", "11")); LAT0 = parse(Int, get(ENV, "RESEACT_LAT0", "29"))
OUT  = get(ENV, "RESEACT_DUMP_OUT", "/tmp/part2.json")
SLICE = native_slice(lon0 = LON0, lat0 = LAT0, nlon = NLON, nlat = NLAT, nlev = NLEV)
t0 = time()
file = EA.load_path(joinpath(REPO, "reseact.esm"); metaparameters = SLICE.metaparameters)
flat = EA.flatten(file)
pre  = EA.algebraic_states_to_observeds(flat)
flat = EA.promote_downstream_shapes(pre)
promoted = EA.promoted_array_names(pre, flat)
parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
open(OUT, "w") do io; JSON3.write(io, docs[2]); end
open(replace(OUT, ".json" => "_part1.json"), "w") do io; JSON3.write(io, docs[1]); end
println("wrote $OUT  ($(round(time()-t0, digits=1)) s)")
