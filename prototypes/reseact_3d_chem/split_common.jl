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
using EarthSciASTSplitter: split_system, stencil_vs_pointwise, spatially_coupled,
                           contains_op, references
using EarthSciIO, JSON3, Dates
const EA = EarthSciAST

# --------------------------------------------------------------------------- #
# 0. The splitting rule: stencil_vs_pointwise, but following OBSERVEDS.
#
# `stencil_vs_pointwise` asks a purely SYNTACTIC question of each additive term:
# does the term itself contain a materialized `makearray`? That is exactly right
# for a term spelled `D(Mx, wrt: lon)`, which lowers to one in place. It is wrong
# for a term spelled as a bare reference to an OBSERVED that is itself a stencil.
# The term is then just a name, the rule sees no makearray, and the term lands in
# the POINTWISE half.
#
# The air-mass equation is exactly that case. Written
#     dm/dt = -(divh_fix + D(Mz, wrt: lev))
# the splitter distributes over the `+` and sends `-D(Mz,lev)` to transport and
# `-divh_fix` to CHEMISTRY -- so the horizontal mass divergence gets evaluated at
# every Rosenbrock stage of the stiff solve (part 2's compile_calls went 1.7e3 ->
# 9.7e4), and no single part's RHS is the air-mass tendency any more, which makes
# the continuity diagnostic in tools/continuity_residual.jl silently measure half
# an equation.
#
# It is not a CORRECTNESS bug -- Lie-Trotter integrates f1+f2 and the term is pure
# forcing (Mx/My/Mz/dp all come from GEOS-FP, not from the state, so its Jacobian
# contribution is exactly zero and the block-diagonal chemistry Jacobian stays
# exact) -- but it is the wrong operator, and it costs.
#
# So: mark an observed as "stencil-valued" if its own definition contains a
# makearray, transitively through the observeds it reads, and treat any term that
# references one as transport. Aggregates list their array operands in `args`, so
# the shipped `references` walk (args-only) sees the dependency edges. The result
# reduces to `stencil_vs_pointwise` on a system whose observeds are all pointwise.
function stencil_following_rule(flat)
    obs = flat.observed_variables
    # Names an expression reads, collected once per observed (the same args-only
    # walk `references` does, inverted so the dependency scan is O(E) not O(E*V)).
    function readnames!(acc::Set{String}, e)
        e isa EA.VarExpr && (push!(acc, e.name); return acc)
        e isa EA.OpExpr && for a in e.args; readnames!(acc, a); end
        return acc
    end
    deps = Dict{String,Set{String}}()
    direct = Dict{String,Bool}()
    for (n, v) in obs
        e = v.expression
        direct[n] = e !== nothing && contains_op(e, "makearray")
        deps[n] = e === nothing ? Set{String}() : readnames!(Set{String}(), e)
    end
    memo = Dict{String,Bool}()
    function is_stencil(name::AbstractString)::Bool
        haskey(memo, name) && return memo[name]
        haskey(direct, name) || return false
        memo[name] = false                                  # cycle guard
        r = direct[name] || any(d -> d != name && is_stencil(d), deps[name])
        memo[name] = r
        return r
    end
    stencil_obs = Set{String}(n for n in keys(obs) if is_stencil(n))
    return function (term, ctx)
        (contains_op(term, "makearray") || spatially_coupled(term, ctx) ||
         references(term, stencil_obs)) ? 1 : 2
    end
end

# --------------------------------------------------------------------------- #
# 1. Split the reseact document into [transport, chemistry] run docs.
#    Replicates EarthSciAST._prepare_run_doc(preserve_refs=true) up to the
#    transformed FlattenedSystem, then splits and re-emits each part.
# --------------------------------------------------------------------------- #
function prepare_split_docs(model_path; rule = nothing, nparts = 2,
                            preserve_refs::Bool = true,
                            metaparameters::AbstractDict = Dict{String,Int}())
    file = isempty(metaparameters) ? EA.load(model_path) :
           EA.load(model_path; metaparameters = metaparameters)
    flat = EA.flatten(file)                                # carries refs by default now (tier at build)
    preserve_refs || (flat = EA.expand_flattened_refs(flat))
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)         # scalar chains lifted to the grid
    r = rule === nothing ? stencil_following_rule(flat) : rule
    parts = split_system(flat, r; nparts = nparts)        # [transport, chemistry]
    return [index_promoted_refs_by_loop!(EA.flattened_to_esm(p), promoted) for p in parts]
end

# --------------------------------------------------------------------------- #
# 1b. Gather-ify bare references to shape-PROMOTED variables inside equations
#     that were ALREADY array-shaped before the promotion.
#
# `promote_downstream_shapes` lifts a scalar 0-D physics chain fed by array
# sources into the grid shape (FastJX: lat/lon/T/P come in as [lon,lat,lev], so
# every j-rate becomes [lon,lat,lev]). It rewrites each promoted variable's OWN
# defining equation into an `arrayop` with its leaves indexed — but it leaves its
# CONSUMERS alone, and returns the promoted-name set for the caller to finish the
# job (`promoted_array_names` -> `index_promoted_refs!`, its documented companion).
#
# For reseact the consumers that matter are the pointwise-lifted species ODEs.
# Those were array-ified earlier, inside `flatten`'s pointwise lift, which indexes
# only the operands that carried a shape AT THAT MOMENT — the promotion has not
# run yet, so `FastJX.j_NO2` is still scalar there and survives as a bare name
# inside a per-cell `aggregate` body. The evaluator then fails to bind it
# (E_TREEWALK_UNBOUND_VARIABLE: FastJX.j_NO2) because an array has no scalar slot.
#
# `index_promoted_refs!` takes ONE `spatial_loops` list, but our equations do not
# share one: the lifted species ODEs iterate the lift's `["i","j","k"]` while a
# promoted definition's own `arrayop` iterates `["_p0","_p1","_p2"]`. So drive it
# per equation with that equation's own `output_idx`, which is exactly the loop
# context its body is evaluated in. Scalar equations (no `output_idx`) are skipped:
# nothing there can legally reference an array bare.
#
# Belongs upstream — `promote_downstream_shapes` knows the promoted set and the
# consumer equations' loop names, so it could close this itself (and `simulate`
# would then get it for free; today it has the same hole).
function index_promoted_refs_by_loop!(doc::AbstractDict, promoted)
    isempty(promoted) && return doc
    models = get(doc, "models", nothing)
    models isa AbstractDict || return doc
    for (_, m) in models
        eqs = get(m, "equations", nothing)
        eqs isa AbstractVector || continue
        for eq in eqs
            rhs = get(eq, "rhs", nothing)
            rhs isa AbstractDict || continue
            oi = get(rhs, "output_idx", nothing)
            oi isa AbstractVector || continue
            # All-symbolic output_idx only. A literal `1` entry (esm-spec §4.3.1
            # singleton dim) has no loop name, so dropping it would emit a gather
            # of the wrong rank -- skip the equation rather than mis-index it.
            all(x -> x isa AbstractString, oi) || continue
            loops = String[String(x) for x in oi]
            isempty(loops) && continue
            # One-equation view; `index_promoted_refs!` mutates eq["rhs"] in place.
            EA.index_promoted_refs!(
                Dict{String,Any}("models" => Dict{String,Any}(
                    "_" => Dict{String,Any}("equations" => Any[eq]))),
                promoted; spatial_loops = loops)
        end
    end
    return doc
end

# --------------------------------------------------------------------------- #
# 2. The GEOS-FP forcing chain (identical to run_c1.jl / Stage A).
#
# `ndays` is how many DAILY GEOS-FP files the cadence spans. GEOS-FP publishes
# one file per collection per day, so a run longer than a day needs more than
# one, and the provider gets a `t -> url` resolver instead of a fixed string:
# every tick names the day it falls in and EarthSciIO locates the record INSIDE
# that day's file (`_file_record`; before that existed, tick 9 of a 3-hourly
# cadence asked an 8-record file for record 9 and threw).
#
# Size the span from the RUN, not the window: the last tick's bracket wants a
# SUCCESSOR, and if there is none the bracket degenerates to [last, last] and
# the forcing silently freezes for the rest of the run. `ndays` must therefore
# cover the run end plus one more tick -- `forcing_days_for` does that.
#
# Model time is seconds since 2016-01-01T00:00:00Z (the epoch the .esm's solar
# chain also decomposes), so day d is EPOCH_DATE + d.
#
# 2016, not 2013, so the meteorology year matches the NEI2016 emissions
# inventory -- there is only one year of NEI, and running 2016 emissions under
# 2013 weather would be a silent inconsistency of exactly the kind the
# LON0/LAT0-vs-lon0_deg seam was made to prevent. The .esm's solar chain needs
# NO change for this: it encodes the epoch as the literals doy0=1, hour0=0 and
# then integrates `gamma_s = k*((doy0-1) + (hour0-12)/24 + t/86400)`, i.e.
# elapsed days from Jan 1, never re-consulting the calendar -- and 2016-01-01 is
# a Jan 1 like any other. The leap day is likewise irrelevant to the sun; it
# only shifts which GEOS-FP FILE a given day index names, which the URL builder
# below derives from the real calendar date and therefore gets right.
# --------------------------------------------------------------------------- #
const EPOCH_DATE = Dates.Date(2016, 1, 1)

"""Daily GEOS-FP files needed to cover `[t0, tf]` in model seconds, including
the one extra day the last bracket's successor may land in."""
forcing_days_for(t0, tf) = Int(floor(tf / 86400)) + 2

function reseact_forcing(dir; ndays::Integer = 1)
    ndays >= 1 || throw(ArgumentError("ndays must be >= 1, got $ndays"))
    cache = EarthSciIO.Cache()
    function url(coll, t)
        d = EPOCH_DATE + Dates.Day(floor(Int, t / 86400))
        return string("https://geos-chem.s3-us-west-2.amazonaws.com/GEOS_4x5/GEOS_FP/",
                      Dates.format(d, "yyyy/mm"), "/GEOSFP.", Dates.format(d, "yyyymmdd"),
                      ".", coll, ".4x5.nc")
    end
    # `n` is the record count PER DAY; the cadence repeats it for each day.
    mk(coll, var, phase, dt, n) = EarthSciIO.discrete_provider(
        cache, t -> url(coll, t), [phase + dt * k for k in 0:(n * ndays - 1)];
        format = "netcdf", variables = [var], time_dim = "time", records_per_sample = 2)
    providers = Dict(
        "GEOSFP.GEOSFP_I3.PS"       => mk("I3", "PS", 0.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3dyn.U"     => mk("A3dyn", "U", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3dyn.V"     => mk("A3dyn", "V", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3dyn.OMEGA" => mk("A3dyn", "OMEGA", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A1.PBLH"     => mk("A1", "PBLH", 1800.0, 3600.0, 24),
        "GEOSFP.GEOSFP_I3.T"        => mk("I3", "T", 0.0, 10800.0, 8),
        # Relative humidity, A3dyn collection/cadence (NOT I3 -- so it blends with
        # w_A3). Feeds Transport3D.H2Oc -> FastJX.H2O and SuperFast.H2O. RH is what
        # the reference uses: GasChem.jl ext/EarthSciDataExt.jl derives H2O from
        # `water_concentration_ppb(g.A3dyn_RH, g.P, g.I3_T)` for BOTH systems.
        "GEOSFP.GEOSFP_A3dyn.RH"    => mk("A3dyn", "RH", 5400.0, 10800.0, 8),
        # --- deposition drivers -------------------------------------------------
        # Wesley (1989) dry deposition + EMEP wet deposition, wired as
        # AtmosphericDeposition.jl ext/EarthSciDataExt.jl does:
        #   dry: Ts<-A1.TS, z0<-A1.Z0M, u_star<-A1.USTAR, G<-A1.SWGDN,
        #        L<-MoninObhukov(rho,TS,USTAR,A1.HFLUX), rho_A<-P/(R T) MW_air,
        #        del_P<-first_level_pressure_thickness(PS), z<-Z_agl
        #   wet: cloudFrac<-A3cld.CLOUD, qrain<-(A3mstE.PFLCU+PFLLSAN)/Vdr/rho_air
        "GEOSFP.GEOSFP_A1.TS"       => mk("A1", "TS", 1800.0, 3600.0, 24),
        "GEOSFP.GEOSFP_A1.Z0M"      => mk("A1", "Z0M", 1800.0, 3600.0, 24),
        "GEOSFP.GEOSFP_A1.USTAR"    => mk("A1", "USTAR", 1800.0, 3600.0, 24),
        "GEOSFP.GEOSFP_A1.SWGDN"    => mk("A1", "SWGDN", 1800.0, 3600.0, 24),
        "GEOSFP.GEOSFP_A1.HFLUX"    => mk("A1", "HFLUX", 1800.0, 3600.0, 24),
        "GEOSFP.GEOSFP_A3cld.CLOUD" => mk("A3cld", "CLOUD", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3mstE.PFLCU"   => mk("A3mstE", "PFLCU", 5400.0, 10800.0, 8),
        "GEOSFP.GEOSFP_A3mstE.PFLLSAN" => mk("A3mstE", "PFLLSAN", 5400.0, 10800.0, 8))
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
# 2a. The native slice: ONE origin, four places that have to agree.
#
# The slice is described twice over, in two different currencies, and neither
# derives the other:
#   * LON0 / LAT0    -- METAPARAMETERS, folded into the index expressions that
#                       read the GEOS-FP arrays (local i -> native LON0+i).
#   * lon0_deg /
#     lat0_deg       -- PARAMETERS in degrees, which the model's own solar chain
#                       uses to place the sun over each cell.
# An .esm parameter default cannot be an expression over a metaparameter, so the
# .esm cannot tie them together; a caller that sets only the metaparameters gets
# meteorology from one place and SUNLIGHT from another, with nothing to complain.
# `hydrostatic_dp` (below) is a third consumer of the same origin, and the .esm's
# own bounds a fourth.
#
# So derive all of them here, from one (lon0, lat0), and never write the
# literals at a call site again.
#
#   native_slice(lon0 = 11, nlon = 13)   -> lon -125..-65, the CONUS span
#   native_slice()                       -> the 7x7x72 central-US default
#
# GEOS-FP 4x5 geometry: lon CENTRES are -180 + 5*(i-1) with cell edges 2.5 deg
# either side; lat POINTS are -90 + 4*(j-1) (the two polar rows are half cells,
# which is caveat (3) in the model description, not something this fixes).
function native_slice(; lon0::Integer = 14, lat0::Integer = 29,
                        nlon::Integer = 7, nlat::Integer = 7, nlev::Integer = 72)
    lon0 >= 1 || throw(ArgumentError("lon0 >= 1: the west halo reads native cell lon0"))
    lat0 >= 1 || throw(ArgumentError("lat0 >= 1: the south flank reads native point lat0"))
    lon0 + nlon + 1 <= 72 || throw(ArgumentError(
        "lon0+nlon+1 = $(lon0+nlon+1) > 72: the east halo runs off the native grid"))
    lat0 + nlat <= 46 || throw(ArgumentError(
        "lat0+nlat = $(lat0+nlat) > 46: the top lat point runs off the native grid"))
    1 <= nlev <= 72 || throw(ArgumentError("nlev in 1..72 (hybrid-coef table length)"))
    return (; lon0 = Int(lon0), lat0 = Int(lat0),
            metaparameters = Dict("NLON" => Int(nlon), "NLAT" => Int(nlat),
                                  "NLEV" => Int(nlev),
                                  "LON0" => Int(lon0), "LAT0" => Int(lat0)),
            parameters = Dict("Transport3D.lon0_deg" => -182.5 + 5.0 * lon0,
                              "Transport3D.lat0_deg" => -90.0 + 4.0 * lat0),
            # human-readable extent, for logging a run's actual footprint
            lon_deg = (-180.0 + 5.0 * lon0, -180.0 + 5.0 * (lon0 + nlon - 1)),
            lat_deg = (-90.0 + 4.0 * lat0, -90.0 + 4.0 * (lat0 + nlat - 1)))
end

# --------------------------------------------------------------------------- #
# 3. Build one forcing-wired RHS closure PER part, sharing u0/p/var_map and the
#    live forcing buffers. Mirrors EarthSciAST.simulate's provider wiring, minus
#    the solve. Returns funcs=(f_transport!, f_chem!, …), u0, p, var_map, and the
#    single refresh callback (cb, tstops) that drives every part's forcing.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 2b. Hydrostatic air mass at time t, on the local grid.
#
# The prognostic state `Transport3D.m` IS the pressure thickness of its cell:
#   dp(i,j,k,t) = dA[k] + dB[k] * PS(i,j,t)
# so m(0) MUST be seeded from the real GEOS-FP surface pressure. The runners used
# to seed it from a CONSTANT `PS_REF = 101325 Pa` -- a leftover from the Stage-B
# configuration where PS really was constant (see the `PS` variable's own
# description, "Replaces the Stage-B constant 1013..."). With real GEOS-FP
# meteorology that seed is wrong by the terrain: measured 16% RMS and 48% in the
# worst column (lon -105, lat 38 -- the Colorado Rockies, PS ~ 80 kPa not 101.3),
# BEFORE a single step is taken. The tracer/air-mass consistency the whole scheme
# rests on starts violated, and any measurement of continuity DRIFT is swamped by
# that constant offset.
#
# Mirrors Transport3D.PS exactly: the I3 field at NATIVE (lat = lat0+j,
# lon = lon0+i), blended across its two bracketing records with w_I3, scaled
# hPa -> Pa. `slice` MUST be the one the model was built with -- pass
# `run.slice`, which `build_split_run` echoes back for exactly this. A mismatch
# does not throw: it seeds every column from the wrong terrain, which then reads
# downstream as a continuity violation the model never committed.
function hydrostatic_dp(merged_param::AbstractDict, const_arrays::AbstractDict, t::Real;
                        slice = native_slice())
    lon0, lat0 = slice.lon0, slice.lat0
    F  = merged_param["GEOSFP.GEOSFP_I3.PS"]
    dA = Float64.(const_arrays["Transport3D.dA"])
    dB = Float64.(const_arrays["Transport3D.dB"])
    w  = (t - 10800.0 * floor(t / 10800.0)) / 10800.0     # I3: 3-hourly, anchored 00:00Z
    return function (i::Integer, j::Integer, k::Integer)
        ps = 100.0 * ((1 - w) * F[1, lat0 + j, lon0 + i] + w * F[2, lat0 + j, lon0 + i])
        return dA[k] + dB[k] * ps
    end
end

function build_split_run(docs, tspan; providers = nothing,
                         parameters::AbstractDict = Dict{String,Float64}(),
                         const_arrays::AbstractDict = Dict{String,Any}(),
                         param_arrays::AbstractDict = Dict{String,Any}(),
                         initial_conditions::AbstractDict = Dict{String,Float64}(),
                         slice = native_slice(),
                         model_name = nothing, inspect = nothing)
    t0 = Float64(tspan[1])
    overrides = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in parameters)
    # The slice's degree-space parameters are applied HERE rather than left to
    # the caller, because forgetting them is silent: LON0/LAT0 move where the
    # meteorology is read from, lon0_deg/lat0_deg move where the SUN is, and a
    # run with the two disagreeing produces a complete, plausible trajectory of
    # the wrong place. They win over `parameters` for the same reason.
    merge!(overrides, Dict{String,Float64}(k => Float64(v) for (k, v) in slice.parameters))

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
            var_map, cb, tstops, merged_param, dms, slice)
end
