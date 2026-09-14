#!/usr/bin/env julia
# ===========================================================================
# direct_rhs_agree.jl -- DOES THE DIRECT StableHLO PROGRAM COMPUTE WHAT THE
# TRACED ONE DOES, ON THIS MODEL, AT A REAL STATE?
# ===========================================================================
# The two lanes cannot meet in one process. ReSEACT's production environment
# pins Reactant 0.2.280 and the direct emitter was built against 0.2.285, and on
# 0.2.285 the traced `:oop` lane does not compile at all (see DIRECT_RHS_CENSUS.md
# section 4). So the comparison goes through FILES: the traced lane writes the
# inputs it used and the `du` it produced, and the direct lane reads those exact
# inputs back and writes its own `du`.
#
# That is stronger than rebuilding the same point twice. Every input the two
# programs see -- the state vector, the parameter values, `t`, and the contents
# of every live forcing buffer -- comes out of ONE file, so a difference in the
# table below can only be the program.
#
#   RESEACT_AGREE_MODE=traced    the production env: build, compile
#                                `rhs_with_buffers`, write inputs.bin and
#                                traced_<half>.bin
#   RESEACT_AGREE_MODE=direct    the emitter's env: read inputs.bin, compile
#                                `direct_rhs_with_buffers`, write direct_<half>.bin
#   RESEACT_AGREE_MODE=compare   no Reactant at all: read the files, print the
#                                agreement table
#
# ONE HALF PER PROCESS. The chemistry half's XLA compile has been OOM-killed in
# this cgroup beside another Julia, so `RESEACT_AGREE_HALF=1` / `=2` builds both
# halves (the split is one build) but compiles and runs only the one asked for.
#
# Env:
#   RESEACT_AGREE_MODE   traced | direct | compare      (required)
#   RESEACT_AGREE_HALF   1 (transport) | 2 (chemistry) | both   (default both)
#   RESEACT_AGREE_DIR    where the .bin files live
#   RESEACT_NLON/NLAT/NLEV, RESEACT_RES, RESEACT_T0, RESEACT_ADJ_UJITTER
#   ESS_OOP_SSA          read at BUILD time (production sets 1)
#   RESEACT_RXENV        the Julia environment to activate
# ===========================================================================
const MODE = get(ENV, "RESEACT_AGREE_MODE", "")
MODE in ("traced", "direct", "compare") ||
    error("set RESEACT_AGREE_MODE to traced, direct or compare")
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const AGDIR = get(ENV, "RESEACT_AGREE_DIR",
                  joinpath("/scratch", get(ENV, "USER", "ctessum"),
                           "oopretire-logs", "agree"))
mkpath(AGDIR)
using Printf, Statistics, Serialization
say(s) = (println(s); flush(stdout))

const HALVES = let h = get(ENV, "RESEACT_AGREE_HALF", "both")
    h == "both" ? [1, 2] : [parse(Int, h)]
end
const HNAME = Dict(1 => "transport", 2 => "chemistry")

# --------------------------------------------------------------------------- #
# compare: the only mode that loads nothing.
# --------------------------------------------------------------------------- #
# Blocks are the model-qualified variable NAME with its subscript stripped, so a
# per-block row is per FIELD (SuperFast.O3, Transport3D.m, …) and names the
# variable a violation belongs to rather than a flat index.
_block(nm::AbstractString) = (i = findfirst('[', nm); i === nothing ? nm : nm[1:i-1])

function compare_mode()
    inp = deserialize(joinpath(AGDIR, "inputs.bin"))
    vm = inp["var_map"]::Dict{String,Int}
    tpts = inp["tpts"]::Vector{Float64}
    blocks = Dict{String,Vector{Int}}()
    for (nm, ix) in vm
        push!(get!(blocks, _block(nm), Int[]), ix)
    end
    say("="^96)
    say("AGREEMENT  direct (Reactant $(inp["direct_rxver"])) vs traced " *
        "(Reactant $(inp["traced_rxver"]))   n_states=$(length(inp["u"]))")
    say("  grid $(inp["grid"])  ESS_OOP_SSA(traced build)=$(inp["ssa"])  " *
        "jitter=$(inp["ujitter"]) seed 31337")
    say("="^96)
    say(rpad("half", 11) * rpad("t (s)", 10) * rpad("max |Δ|", 13) *
        rpad("max rel", 13) * rpad("max |traced|", 13) * rpad("nonfinite", 11) *
        "worst variable")
    worst = Dict{String,Tuple{Float64,String}}()
    for h in HALVES
        nm = HNAME[h]
        tf = joinpath(AGDIR, "traced_$(nm).bin")
        df = joinpath(AGDIR, "direct_$(nm).bin")
        if !(isfile(tf) && isfile(df))
            say("  $nm: MISSING " * (isfile(tf) ? "" : "traced_$(nm).bin ") *
                (isfile(df) ? "" : "direct_$(nm).bin"))
            continue
        end
        T = deserialize(tf)::Dict{String,Any}
        D = deserialize(df)::Dict{String,Any}
        for (k, t) in enumerate(tpts)
            du_t = T["du_$k"]::Vector{Float64}
            du_d = D["du_$k"]::Vector{Float64}
            length(du_t) == length(du_d) || error("$nm t=$t: width mismatch")
            nnf = count(!isfinite, du_d) + count(!isfinite, du_t)
            ad = abs.(du_d .- du_t)
            # Relative to the traced value with an ABSOLUTE FLOOR of 1e-14: a
            # component whose traced value is ~0 has no meaningful relative
            # error, and dividing by it is what turns a 1e-20 difference into a
            # spurious "1e6 relative".
            rd = ad ./ max.(abs.(du_t), 1e-14)
            ia = argmax(ad); ir = argmax(rd)
            bname = "?"
            for (n2, ix) in vm; ix == ir && (bname = n2; break); end
            say(rpad(nm, 11) * rpad(@sprintf("%.0f", t), 10) *
                rpad(@sprintf("%.3e", ad[ia]), 13) *
                rpad(@sprintf("%.3e", rd[ir]), 13) *
                rpad(@sprintf("%.3e", maximum(abs, du_t)), 13) *
                rpad(string(nnf), 11) * bname)
            cur = get(worst, nm, (0.0, ""))
            rd[ir] > cur[1] && (worst[nm] = (rd[ir], bname))
        end
        # Per-variable-block worst relative difference over all three times.
        say("  per-variable-block worst relative difference ($nm):")
        rows = Tuple{Float64,String,Float64}[]
        for (b, idxs) in blocks
            r = 0.0; a = 0.0
            for k in eachindex(tpts)
                du_t = T["du_$k"]::Vector{Float64}; du_d = D["du_$k"]::Vector{Float64}
                for ix in idxs
                    d = abs(du_d[ix] - du_t[ix])
                    a = max(a, d)
                    r = max(r, d / max(abs(du_t[ix]), 1e-14))
                end
            end
            push!(rows, (r, b, a))
        end
        sort!(rows; rev = true)
        for (r, b, a) in rows
            say(@sprintf("    %-28s rel %.3e   abs %.3e", b, r, a))
        end
    end
    say("="^96)
    for h in HALVES
        nm = HNAME[h]
        haskey(worst, nm) || continue
        r, b = worst[nm]
        cls = r <= 1e-13 ? "ALGEBRAIC (<=1e-13)" :
              r <= 1e-12 ? "TRANSCENDENTAL (<=1e-12)" :
              r <= 1e-11 ? "REDUCTION (<=1e-11)" : "OUTSIDE every class"
        say(@sprintf("%-10s worst relative %.3e on %s -> %s", nm, r, b, cls))
    end
    say("="^96)
end

if MODE == "compare"
    compare_mode()
    exit(0)
end

# --------------------------------------------------------------------------- #
# Both compiling modes share the census probe's build block.
# --------------------------------------------------------------------------- #
import Pkg
Pkg.activate(get(ENV, "RESEACT_RXENV", joinpath(REPO, "run-model-jl")); io = devnull)
using LinearAlgebra, Logging, Random
using EarthSciAST, EarthSciIO
using EarthSciASTSplitter: split_system
using Reactant
const EA = EarthSciAST
const RX = Reactant
try; RX.set_default_backend("cpu"); catch; end

const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
include(joinpath(CHEMDIR, "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize

get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
get!(ENV, "ESS_OOP_SSA", "1")
const SSA   = ENV["ESS_OOP_SSA"]
const MODEL = get(ENV, "RESEACT_MODEL", joinpath(REPO, "reseact.esm"))
const T0    = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const UJIT  = parse(Float64, get(ENV, "RESEACT_ADJ_UJITTER", "1e-1"))
_envi(k) = haskey(ENV, "RESEACT_$k") ? parse(Int, ENV["RESEACT_$k"]) : nothing
const RES   = get(ENV, "RESEACT_RES", "4x5")
const SLICE = native_slice(res = RES, lon0 = _envi("LON0"), lat0 = _envi("LAT0"),
                           nlon = _envi("NLON"), nlat = _envi("NLAT"),
                           nlev = _envi("NLEV"))
const GRID_MP  = SLICE.metaparameters
const NLEV_EFF = GRID_MP["NLEV"]
const GRIDSTR  = "$(GRID_MP["NLON"])x$(GRID_MP["NLAT"])x$(NLEV_EFF)"
const TPTS  = Float64[T0, T0 + 8 * 3600, T0 + 16 * 3600]
const NDAYS = forcing_days_for(T0, last(TPTS))

# The adjoint driver's production compile options, applied to BOTH lanes: a
# comparison compiled under different options would be measuring the options.
const _EXCL_RAW = get(ENV, "RESEACT_CENSUS_EXCL", "dynamic_update_to_concat,sub_const_prop")
const EXCLP = _EXCL_RAW == "none" ? String[] :
    String.(filter(!isempty, strip.(split(_EXCL_RAW, ','))))
const COPTS = RX.CompileOptions(; sync = true,
                                xla_debug_options = (; xla_cpu_prefer_vector_width = 128),
                                (isempty(EXCLP) ? (;) : (; excluded_passes = EXCLP))...)

say("="^78)
say(@sprintf("DIRECT/TRACED AGREEMENT  mode=%s  halves=%s  grid=%s  Reactant %s  ESS_OOP_SSA=%s",
             MODE, join(HALVES, ","), GRIDSTR, pkgversion(RX), SSA))
say("  files: $AGDIR")
say("="^78)

validate_reseact(MODEL; metaparameters = GRID_MP, say = say)
fo = Vector{Any}(undef, 2); dms = Vector{Any}(undef, 2)
u0 = p = var_map = nothing
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
ff = nothing; merged_const = nothing; ov = nothing
tb = time()
Logging.with_logger(Logging.NullLogger()) do
    global fo, dms, u0, p, var_map, merged_param, discrete, ff, merged_const, ov
    file = EA.load_path(MODEL; metaparameters = GRID_MP)
    flat = EA.flatten(file)
    pre  = EA.algebraic_states_to_observeds(flat)
    flat = EA.promote_downstream_shapes(pre)
    promoted = EA.promoted_array_names(pre, flat)
    splitparts = split_system(flat, stencil_following_rule(flat); nparts = 2)
    docs  = [index_promoted_refs_by_loop!(EA.flattened_to_esm(pt), promoted) for pt in splitparts]
    f0 = reseact_forcing(CHEMDIR; ndays = NDAYS, res = SLICE.res)
    ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV_EFF)))
    merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
    for (rawk, prov) in ff.providers
        k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
        if EA.provider_is_const(prov)
            merged_const[k] = fld
        else
            merged_param[k] = fld; discrete[k] = prov
        end
    end
    ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
    merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))
    for i in 1:2
        dms[i] = EA.DiscreteMaterializer()
        fi, u0i, pi, _, vmi = EA.build_evaluator(docs[i]; form = :oop,
            parameter_overrides = ov, const_arrays = merged_const,
            param_arrays = merged_param, materialize_out = dms[i])
        fo[i] = fi
        if i == 1
            u0, p, var_map = u0i, pi, vmi
        else
            vmi == var_map || error("split part 2 var_map != part 1")
        end
    end
end
foreach(d -> d.materialize!(), dms)
say(@sprintf("BUILD (two :oop halves) %.2f s   nstates=%d  nparams=%d",
             time() - tb, length(u0), length(p)))

# `Transport3D.m` seeded from the real GEOS-FP surface pressure, then the
# harnesses' jitter — never the uniform default, which sits on the PPM limiters'
# switching surfaces.
let dp0 = hydrostatic_dp(merged_param, ff.const_arrays, T0; slice = SLICE)
    for (nm, idx) in var_map
        mm = match(r"^Transport3D\.m\[(\d+),(\d+),(\d+)\]$", nm)
        mm === nothing && continue
        u0[idx] = dp0(parse(Int, mm.captures[1]), parse(Int, mm.captures[2]),
                      parse(Int, mm.captures[3]))
    end
end
const N = length(u0)
const UJITTERED = let u = copy(u0)
    UJIT > 0 && (u .*= (1 .+ UJIT .* randn(Random.MersenneTwister(31337), N)))
    u
end

const HOSTBUFS = [EA.forcing_buffers(fo[i]) for i in 1:2]
const BUFNAMES = [String[String(k) for k in keys(HOSTBUFS[i])] for i in 1:2]
const DEVBUFS  = [map(RX.ConcreteRArray, hb) for hb in HOSTBUFS]
function refresh_forcing(t)
    for (k, prov) in discrete
        merged_param[k] .= EA._provider_const_field(EA.provider_sample(prov, t), k)
    end
    foreach(d -> d.materialize!(), dms)
    for i in 1:2
        EA.sync_forcing!(DEVBUFS[i], EA.forcing_buffers(fo[i]))
    end
    return nothing
end

compile4(g, u, pp, t, b) = RX.@compile compile_options = COPTS g(u, pp, t, b)
_devp(pp::NamedTuple) = NamedTuple{keys(pp)}(map(RX.ConcreteRNumber, values(pp)))

# --------------------------------------------------------------------------- #
# traced: the lane production runs. Writes the inputs AND its own `du`.
# --------------------------------------------------------------------------- #
if MODE == "traced"
    inp = Dict{String,Any}("var_map" => Dict{String,Int}(var_map),
                           "u" => copy(UJITTERED),
                           "p_names" => String[String(k) for k in keys(p)],
                           "p_vals" => Float64[Float64(v) for v in values(p)],
                           "tpts" => copy(TPTS), "grid" => GRIDSTR,
                           "ssa" => SSA, "ujitter" => UJIT,
                           "traced_rxver" => string(pkgversion(RX)),
                           "direct_rxver" => "(pending)",
                           "buf_names" => BUFNAMES)
    u_dev = RX.ConcreteRArray(copy(UJITTERED))
    PR = _devp(p)
    for h in HALVES
        nm = HNAME[h]
        say("\n" * "-"^78)
        say("  $nm: TRACED `rhs_with_buffers`")
        g = EA.rhs_with_buffers(fo[h])
        t0 = time()
        c = compile4(g, u_dev, PR, RX.ConcreteRNumber(T0), DEVBUFS[h])
        say(@sprintf("    @compile %.1f s", time() - t0))
        out = Dict{String,Any}()
        for (k, t) in enumerate(TPTS)
            refresh_forcing(t)
            # The buffers as the program will actually see them, captured from
            # THIS half's host arrays after the refresh.
            inp["bufs_$(nm)_$k"] = Vector{Float64}[copy(b) for b in HOSTBUFS[h]]
            du = Array(c(u_dev, PR, RX.ConcreteRNumber(t), DEVBUFS[h]))
            out["du_$k"] = Vector{Float64}(du)
            say(@sprintf("    t=%.0f  |du|_inf=%.6e  nonfinite=%d",
                         t, maximum(abs, du), count(!isfinite, du)))
        end
        serialize(joinpath(AGDIR, "traced_$(nm).bin"), out)
    end
    # Merge into any inputs.bin a sibling process wrote for the other half.
    f = joinpath(AGDIR, "inputs.bin")
    if isfile(f)
        old = deserialize(f)::Dict{String,Any}
        old["u"] == inp["u"] || error("inputs.bin on disk used a different state")
        merge!(old, inp); inp = old
    end
    serialize(f, inp)
    say("\n  wrote $(f) and traced_*.bin")
end

# --------------------------------------------------------------------------- #
# direct: the emitter's lane, on the traced lane's OWN inputs.
# --------------------------------------------------------------------------- #
if MODE == "direct"
    inp = deserialize(joinpath(AGDIR, "inputs.bin"))::Dict{String,Any}
    inp["var_map"] == Dict{String,Int}(var_map) ||
        error("the state layout differs between the two environments; the " *
              "comparison would be between different variables")
    u_saved = inp["u"]::Vector{Float64}
    length(u_saved) == N || error("saved state width $(length(u_saved)) != $N")
    pn = inp["p_names"]::Vector{String}; pv = inp["p_vals"]::Vector{Float64}
    pn == String[String(k) for k in keys(p)] ||
        error("the parameter NamedTuple's keys differ between environments")
    tpts = inp["tpts"]::Vector{Float64}
    inp["direct_rxver"] = string(pkgversion(RX))
    EXT = Base.get_extension(EarthSciAST, :EarthSciASTReactantExt)
    EXT === nothing && error("the Reactant extension did not load")
    u_dev = RX.ConcreteRArray(copy(u_saved))
    PR = NamedTuple{Tuple(Symbol.(pn))}(Tuple(RX.ConcreteRNumber(v) for v in pv))
    for h in HALVES
        nm = HNAME[h]
        say("\n" * "-"^78)
        say("  $nm: DIRECT emission")
        inp["buf_names"][h] == BUFNAMES[h] ||
            error("$nm: the forcing-buffer order differs between environments")
        d = EXT.direct_rhs_with_buffers(fo[h]; var_map = var_map)
        t0 = time()
        c = compile4(d, u_dev, PR, RX.ConcreteRNumber(T0), DEVBUFS[h])
        say(@sprintf("    EMITTED and COMPILED in %.1f s   stats=%s", time() - t0, d.d.stats))
        mod = repr(RX.@code_hlo compile_options = COPTS optimize = true d(
            u_dev, PR, RX.ConcreteRNumber(T0), DEVBUFS[h]))
        say(@sprintf("    optimized module: %d lines", count(==('\n'), mod) + 1))
        out = Dict{String,Any}("compile_s" => time() - t0,
                               "module_lines" => count(==('\n'), mod) + 1)
        # Per-call cost, on the same program, for the census table.
        c(u_dev, PR, RX.ConcreteRNumber(T0), DEVBUFS[h])
        ts = Float64[]
        for _ in 1:20
            tt = time(); c(u_dev, PR, RX.ConcreteRNumber(T0), DEVBUFS[h])
            push!(ts, time() - tt)
        end
        out["percall_median_s"] = median(ts)
        say(@sprintf("    per-call median %.3f ms  (min %.3f, max %.3f)",
                     1e3 * median(ts), 1e3 * minimum(ts), 1e3 * maximum(ts)))
        for (k, t) in enumerate(tpts)
            # The traced lane's OWN buffer contents, not a re-derived refresh.
            saved = inp["bufs_$(nm)_$k"]::Vector{Vector{Float64}}
            hb = HOSTBUFS[h]
            length(saved) == length(hb) || error("$nm t=$t: buffer count mismatch")
            for (j, b) in enumerate(hb)
                length(b) == length(saved[j]) || error("$nm t=$t: buffer $j width")
                copyto!(b, saved[j])
            end
            EA.sync_forcing!(DEVBUFS[h], hb)
            du = Array(c(u_dev, PR, RX.ConcreteRNumber(t), DEVBUFS[h]))
            out["du_$k"] = Vector{Float64}(du)
            say(@sprintf("    t=%.0f  |du|_inf=%.6e  nonfinite=%d",
                         t, maximum(abs, du), count(!isfinite, du)))
        end
        serialize(joinpath(AGDIR, "direct_$(nm).bin"), out)
    end
    serialize(joinpath(AGDIR, "inputs.bin"), inp)
    say("\n  wrote direct_*.bin")
end
