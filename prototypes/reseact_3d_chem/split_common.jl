# Shared glue for the operator-split / IMEX Stage-C runs.
#
# The reseact model is TOO STIFF for a monolithic dense solve (4459 states => a
# dense finite-difference Jacobian is O(N) RHS evals + O(N^3) LU per step). The
# fix is operator splitting: the PDE (transport) terms and the pointwise
# (chemistry) terms go to two sub-systems that share ONE state vector, so
#   f_full(u) = f_transport(u) + f_chemistry(u)   (exactly).
# Transport is non-stiff => an explicit SSP-RK solver; chemistry is stiff but
# BLOCK-DIAGONAL (each grid cell's chemistry couples only its own species) => an
# implicit solver whose Jacobian is a BlockDiagonal (343 blocks of 13x13).
#
# The split itself is EarthSciASTSplitter's `split_system` with the
# `stencil_vs_pointwise` rule (a term is transport iff it carries a `makearray`
# stencil). The forcing (GEOS-FP winds/T/pressure) is wired exactly as
# `EarthSciAST.simulate` does — both parts are built over the SAME live forcing
# buffers so one refresh callback drives both.

using EarthSciAST
using EarthSciASTSplitter
using EarthSciASTSplitter: split_system, stencil_vs_pointwise
using EarthSciIO, JSON3
const EA = EarthSciAST

# --------------------------------------------------------------------------- #
# 1. Split the reseact document into [transport, chemistry] run docs.
#    Replicates EarthSciAST._prepare_run_doc(preserve_refs=true) up to the
#    transformed FlattenedSystem, then splits and re-emits each part.
# --------------------------------------------------------------------------- #
function prepare_split_docs(model_path; rule = stencil_vs_pointwise, nparts = 2,
                            preserve_refs::Bool = true)
    file = EA.load(model_path)
    flat = EA.flatten(file)                                # carries refs by default now (tier at build)
    preserve_refs || (flat = EA.expand_flattened_refs(flat))
    flat = EA.promote_downstream_shapes(EA.algebraic_states_to_observeds(flat))
    parts = split_system(flat, rule; nparts = nparts)     # [transport, chemistry]
    return [EA.flattened_to_esm(p) for p in parts]
end

# --------------------------------------------------------------------------- #
# 2. The GEOS-FP forcing chain (identical to run_c1.jl / Stage A).
# --------------------------------------------------------------------------- #
function reseact_forcing(dir)
    BASE = "https://geos-chem.s3-us-west-2.amazonaws.com/GEOS_4x5/GEOS_FP/2013/01/GEOSFP.20130101"
    cache = EarthSciIO.Cache()
    mk(coll, var, phase, dt, n) = EarthSciIO.discrete_provider(
        cache, "$BASE.$coll.4x5.nc", [phase + dt * k for k in 0:(n-1)];
        format = "netcdf", variables = [var], time_dim = "time", records_per_sample = 2)
    providers = Dict(
        "GEOSFP.GEOSFP_I3.PS"       => mk("I3", "PS", 0.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3dyn.U"     => mk("A3dyn", "U", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3dyn.V"     => mk("A3dyn", "V", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3dyn.OMEGA" => mk("A3dyn", "OMEGA", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A1.PBLH"     => mk("A1", "PBLH", 1800.0, 3600.0, 24),
        "GEOSFP.GEOSFP_I3.T"        => mk("I3", "T", 0.0, 10800.0, 8))
    params = Dict(
        "GEOSFP.t_interp_ref_I3" => 0.0, "GEOSFP.dt_interp_I3" => 10800.0,
        "GEOSFP.t_interp_ref_A3" => 5400.0, "GEOSFP.dt_interp_A3" => 10800.0,
        "GEOSFP.t_interp_ref_A1" => 1800.0, "GEOSFP.dt_interp_A1" => 3600.0)
    co = JSON3.read(read(joinpath(dir, "hybrid_coefs.json"), String))
    const_arrays = Dict{String,Any}(
        "Transport3D.dA" => Float64.(co.dA), "Transport3D.dB" => Float64.(co.dB),
        "Transport3D.Ap" => Float64.(co.Ap), "Transport3D.Bp" => Float64.(co.Bp))
    return (; providers, parameters = params, const_arrays)
end

# --------------------------------------------------------------------------- #
# 3. Build one forcing-wired RHS closure PER part, sharing u0/p/var_map and the
#    live forcing buffers. Mirrors EarthSciAST.simulate's provider wiring, minus
#    the solve. Returns funcs=(f_transport!, f_chem!, …), u0, p, var_map, and the
#    single refresh callback (cb, tstops) that drives every part's forcing.
# --------------------------------------------------------------------------- #
function build_split_run(docs, tspan; providers = nothing,
                         parameters::AbstractDict = Dict{String,Float64}(),
                         const_arrays::AbstractDict = Dict{String,Any}(),
                         param_arrays::AbstractDict = Dict{String,Any}(),
                         initial_conditions::AbstractDict = Dict{String,Float64}(),
                         model_name = nothing, inspect = nothing)
    t0 = Float64(tspan[1])
    overrides = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in parameters)

    # --- forcing wiring (once): const providers -> const_arrays; discrete -> live
    #     buffers in param_arrays that a single refresh callback rewrites in place.
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in const_arrays)
    merged_param = Dict{String,Any}(String(k) => v for (k, v) in param_arrays)
    discrete_providers = Dict{String,Any}()
    if providers !== nothing
        for (rawk, prov) in providers
            k = String(rawk)
            fld = EA._provider_const_field(EA.provider_sample(prov, t0), k)
            if EA.provider_is_const(prov)
                merged_const[k] = fld
            else
                merged_param[k] = fld
                discrete_providers[k] = prov
            end
        end
    end

    # --- build each part over the shared buffers, each with its own discrete
    #     materializer (state-free derived caches differ per part). Per-part timing +
    #     compile-once tier bench counters so a slow build is diagnosable (not blind).
    funcs = Function[]; dms = EA.DiscreteMaterializer[]
    u0 = nothing; p = nothing; var_map = nothing
    for (i, doc) in enumerate(docs)
        dm = EA.DiscreteMaterializer()
        EA._BENCH_ON[] = true; EA._bench_reset!()
        tb = time()
        f!, u0i, pi, _tspan, vmi = EA.build_evaluator(doc;
            model_name = model_name, parameter_overrides = overrides,
            const_arrays = merged_const, param_arrays = merged_param,
            materialize_out = dm, inspect = (i == 1 ? inspect : nothing))
        bt = time() - tb
        bv = EA._BENCH_BODY_VARIANTS[]; cc = EA._BENCH_COMPILE_CALLS[]
        EA._BENCH_ON[] = false
        println("  part[$i] build: $(round(bt, digits=1)) s  tier body_variants=$bv compile_calls=$cc  (tier $(bv>0 ? "ON" : "OFF/fused"))"); flush(stdout)
        push!(funcs, f!); push!(dms, dm)
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part $i var_map != part 1 (state sets must match)")
        end
    end

    isempty(initial_conditions) || EA._apply_initial_conditions!(u0, var_map, initial_conditions)

    # --- ONE refresh callback for all parts; post_refresh rebuilds every part's
    #     discrete caches at each cadence boundary.
    cb = nothing; tstops = Float64[]
    if !isempty(discrete_providers)
        # New API (EarthSciASTDataRefreshExt): zero-positional keyword method; the
        # callback is a pure function of the provider/buffer registries and never
        # reads the model, so the old positional `model` arg is gone.
        post = () -> (for d in dms; d.materialize!(); end)
        cb, tstops = EA.build_refresh_callback(; providers = discrete_providers,
            buffers = EA.RefreshBuffers(merged_param), post_refresh = post)
    end

    return (; funcs = Tuple(funcs), u0, p, tspan = (t0, Float64(tspan[2])),
            var_map, cb, tstops, merged_param, dms)
end
