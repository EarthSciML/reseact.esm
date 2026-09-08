#!/usr/bin/env julia
# Runtime cost of the tier change: build part 1 once and time the RHS.
# A build-time win that cost RHS time would be a bad trade, so price it.
#   AE_GRID  NLONxNLATxNLEV (default 12x12x8)   AE_N  reps (default 200)
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using Printf, Logging, SHA, Statistics
using EarthSciAST, EarthSciIO, JSON3
using EarthSciASTSplitter: split_system
const EA = EarthSciAST
include(joinpath(@__DIR__, "_env.jl"))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
say(s) = (println(s); flush(stdout))
const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const NLON, NLAT, NLEV = parse.(Int, split(get(ENV, "AE_GRID", "12x12x8"), 'x'))
const NREP = parse(Int, get(ENV, "AE_N", "200"))
say("EarthSciAST from: " * String(pkgdir(EarthSciAST)))
F0 = reseact_forcing(CHEMDIR; ndays = 1)
slice = native_slice(lon0 = 11, lat0 = 29, nlon = NLON, nlat = NLAT, nlev = NLEV)
mp = slice.metaparameters
# NOTE: `local fi, ...` + assignment inside the `do` closure made the build's
# results locals of the CLOSURE (`UndefVarError: fi`); return them instead.
fi, u0i, pi = Logging.with_logger(Logging.NullLogger()) do
    file = EA.load_path(MODEL; metaparameters = mp)
    flat = EA.flatten(file); pre = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(p), promoted) for p in parts]
    ff = merge(F0, (; const_arrays = GridResize.slice_hybrid_coefs(F0.const_arrays, mp["NLEV"])))
    mc = Dict{String,Any}(String(k) => v for (k,v) in ff.const_arrays); mpar = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, 5400.0), k)
        EA.provider_is_const(prov) ? (mc[k] = fld) : (mpar[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k,v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k,v) in slice.parameters))
    EA._reset_cascade_tally!()
    fi, u0i, pi, _, _ = EA.build_evaluator(docs[1]; form = :oop, parameter_overrides = ov,
        const_arrays = mc, param_arrays = mpar, materialize_out = EA.DiscreteMaterializer())
    say("tally: " * join(["$(k)=$(v)" for (k,v) in sort(collect(EA._CASCADE_TALLY), by=x->String(x[1]))], "  "))
    (fi, u0i, pi)
end
du = fi(u0i, pi, 5400.0)
say("sha " * bytes2hex(sha256(reinterpret(UInt8, vec(du)))))
for _ in 1:20; fi(u0i, pi, 5400.0); end               # warm
ts = Float64[]
for _ in 1:NREP; push!(ts, @elapsed fi(u0i, pi, 5400.0)); end
@printf("RHS %s  n=%d  min %.4f ms  median %.4f ms  mean %.4f ms\n",
        get(ENV,"AE_GRID","12x12x8"), NREP, 1e3*minimum(ts), 1e3*median(ts), 1e3*mean(ts))
