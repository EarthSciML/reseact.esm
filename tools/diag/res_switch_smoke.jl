# ===========================================================================
# res_switch_smoke.jl -- the resolution switch, gated.
# ===========================================================================
# WHAT IT ASSERTS. Building the model on a small box at a given GEOS-FP
# resolution (a) resolves the four degree parameters from the resolution row,
# (b) pulls the native forcing arrays at that grid's native extent, and (c)
# produces a finite transport RHS. It prints a SHA-256 of that RHS, which is
# what makes the 4x5 arm a REGRESSION gate rather than a smoke test.
#
# MEASURED 2026-09-05, 6x6x8 box, t = 64800 s (18:00Z 2016-01-01):
#
#   4x5    RHSSHA 4f820583fe4038b95f9b19825e0997c1ac721f0960938c202b2949cfd691aceb
#          sum -5.668493200231e-01  norm 8.632791319959e-02
#          BIT-IDENTICAL to the pre-resolution-switch model (reseact.esm at
#          47b603a-era HEAD, where dlon_deg/dlat_deg were literal EQUATIONS and
#          native_slice passed no spacing overrides). That is the whole claim of
#          the refactor: promoting the two spacings to parameters and driving
#          them from the table changes nothing at 4x5.
#          Native arrays: PS (2,46,72), U (2,72,46,72) -- the global 4x5 grid.
#
#   2x2.5  sum -6.212141248915e-01  norm 9.334668797662e-02, finite
#          Native arrays: PS (2,91,144), U (2,72,91,144).
#          Parameters resolved to dlon 2.5 / dlat 2.0 / lon0 -126.25 / lat0 26.0.
#          Values DIFFER from 4x5 by construction -- a 6x6 box at 2.5 deg is a
#          different piece of atmosphere than a 6x6 box at 5 deg. Only the 4x5
#          arm is an equality gate.
#
# HOW TO RE-GATE after touching split_common.jl / geosfp_grids.jl / the .esm's
# geometry parameters:
#
#   julia --project=run-model-jl tools/diag/res_switch_smoke.jl            # 4x5
#   SMOKE_RES=2x2.5 julia --project=run-model-jl tools/diag/res_switch_smoke.jl
#
# and compare the 4x5 SHA against the line above. To re-derive the reference
# from git rather than trusting it, check an old reseact.esm out INTO THE REPO
# (its subsystem refs resolve relative to its own directory, so a copy in /tmp
# cannot find ../EarthSciModels) and point SMOKE_MODEL at it with
# SMOKE_DROP_SPACING=1:
#
#   git show <rev>:reseact.esm > .head_reference.esm
#   SMOKE_MODEL=$PWD/.head_reference.esm SMOKE_DROP_SPACING=1 \
#     julia --project=run-model-jl tools/diag/res_switch_smoke.jl
#
# Runs in ~6 min on a login node; the build is small enough for an interactive
# cgroup, unlike anything at CONUS scale.
# ===========================================================================
# Resolution-switch smoke test: build a TINY box at each resolution, evaluate the
# transport RHS once, and print the geometry the model actually resolved.
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
using Logging, Printf, SHA
const EA = EarthSciAST

res = get(ENV, "SMOKE_RES", "4x5")
slice = native_slice(res = res, nlon = 6, nlat = 6, nlev = 8)
model = get(ENV, "SMOKE_MODEL", joinpath(REPO, "reseact.esm"))
drop  = get(ENV, "SMOKE_DROP_SPACING", "0") == "1"
println("res=$res  mp=", slice.metaparameters)
println("  params: ", sort(collect(slice.parameters), by = first))
println("  lon ", slice.lon_deg, "  lat ", slice.lat_deg)
mp = slice.metaparameters
t0 = 64800.0
u0 = p = nothing; f1 = nothing
Logging.with_logger(Logging.NullLogger()) do
    global u0, p, f1
    file = EA.load_path(model; metaparameters = mp)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in parts]
    f0 = reseact_forcing(CHEMDIR; ndays = 1, res = res)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, mp["NLEV"])))
    mc = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    mpar = Dict{String,Any}()
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, t0), k)
        EA.provider_is_const(prov) ? (mc[k] = fld) : (mpar[k] = fld)
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters
                                if !(drop && endswith(String(k), "dlon_deg") || drop && endswith(String(k), "dlat_deg"))))
    fi, u0i, pi, _, vmi = EA.build_evaluator(docs[1]; form = :oop,
        parameter_overrides = ov, const_arrays = mc, param_arrays = mpar)
    f1 = fi; u0 = u0i; p = pi
    for (k, v) in mpar
        println(@sprintf("  forcing %-22s size %s", k, string(size(v))))
    end
end
du = f1(u0, p, t0)
println("  RHSSHA ", bytes2hex(sha256(reinterpret(UInt8, vec(du)))))
@printf("  RHS: n=%d  sum=%.12e  norm=%.12e  finite=%s\n",
        length(du), sum(du), sqrt(sum(abs2, du)), all(isfinite, du))
