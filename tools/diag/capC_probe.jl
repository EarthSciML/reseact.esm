#!/usr/bin/env julia
# ===========================================================================
# capC_probe.jl -- DOES A FIXED-LANE-CAPACITY CHEMISTRY BUILD WORK AT ALL?
# ===========================================================================
# Everything in the variable-per-cell-timestep plan rests on one question: can
# the pointwise (chemistry) RHS be evaluated on a COMPACTED SUB-BATCH of cells?
# The built evaluator's slot and scatter vectors are computed for the full NC
# (`_oop_gather(u, slots)` with slot arithmetic `(s-1)*NC + c`), so a batch of C
# cells needs its OWN build at capacity C -- not a slice of the NC build.
#
# This builds the chemistry half TWICE at a small grid: once normally, once at a
# `C x 1 x 1` lane grid through `tools/capacity_chem.jl`, and checks that the
# capacity RHS reproduces the reference RHS CELL FOR CELL under a RANDOM cell
# permutation. A random permutation is the point: an identity map would pass on
# a build that had silently kept some geometry tied to the grid index.
#
# It also runs a PADDED capacity (C > NC) to check that padding lanes cannot
# leak into a real lane's result, and reports both builds' wall cost, which is
# the second thing the plan wants to know (a capacity build's cost should depend
# on C, not on the real grid).
#
#   RESEACT_NLON/NLAT/NLEV   reference grid (default 6x6x8 -> NC = 288)
#   RESEACT_CAPC             lane capacity (default NC; also runs NC + 37)
# ===========================================================================
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
get!(ENV, "RESEACT_NLON", "6"); get!(ENV, "RESEACT_NLAT", "6"); get!(ENV, "RESEACT_NLEV", "8")
include(joinpath(@__DIR__, "_env.jl"))
using EarthSciAST, EarthSciIO, JSON3, Logging, Printf, Random, Statistics
const EA = EarthSciAST
include(joinpath(REPO, "prototypes", "reseact_3d_chem", "split_common.jl"))
include(joinpath(REPO, "tools", "grid_resize.jl")); using .GridResize
include(joinpath(REPO, "tools", "capacity_chem.jl")); using .CapacityChem
using EarthSciASTSplitter: split_system
say(s) = (println(s); flush(stdout))

const MODEL = joinpath(REPO, "reseact.esm")
const CHEMDIR = joinpath(REPO, "prototypes", "reseact_3d_chem")
_env(k, d) = parse(Int, get(ENV, "RESEACT_$k", string(d)))
const T0 = parse(Float64, get(ENV, "RESEACT_T0", "5400"))
const SLICE = native_slice(lon0 = _env("LON0", 11), lat0 = _env("LAT0", 29),
                           nlon = _env("NLON", 6), nlat = _env("NLAT", 6),
                           nlev = _env("NLEV", 8))
const GRID = SLICE.metaparameters
const NLON, NLAT, NLEV = GRID["NLON"], GRID["NLAT"], GRID["NLEV"]
const LON0, LAT0 = GRID["LON0"], GRID["LAT0"]
const NC = NLON * NLAT * NLEV
const NDAYS = forcing_days_for(T0, T0 + 600)

say("="^78)
say(@sprintf("CAPACITY-C CHEMISTRY PROBE   reference grid %dx%dx%d (NC=%d)  T0=%.0f",
             NLON, NLAT, NLEV, NC, T0))
say("="^78)

# --------------------------------------------------------------------------- #
# 1. The reference chemistry build -- the same pipeline adjoint_gradient.jl uses.
# --------------------------------------------------------------------------- #
function split_chem(mp)
    Logging.with_logger(Logging.NullLogger()) do
        file = EA.load_path(MODEL; metaparameters = mp)
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
        return index_promoted_refs_by_loop!(EA.flattened_to_esm(parts[2]), promoted)
    end
end

t = time(); docC = split_chem(GRID); say(@sprintf("load+split (reference) %.1f s", time() - t))

f0 = reseact_forcing(CHEMDIR; ndays = NDAYS)
ff = merge(f0, (; const_arrays = GridResize.slice_hybrid_coefs(f0.const_arrays, NLEV)))
merged_const = Dict{String,Any}(String(k) => v for (k, v) in ff.const_arrays)
merged_param = Dict{String,Any}(); discrete = Dict{String,Any}()
for (rawk, prov) in ff.providers
    k = String(rawk); fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
    if EA.provider_is_const(prov); merged_const[k] = fld
    else; merged_param[k] = fld; discrete[k] = prov; end
end
ov = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in ff.parameters)
merge!(ov, Dict{String,Float64}(k => Float64(v) for (k, v) in SLICE.parameters))

insp = EA.BuildInspection()
t = time()
fR, u0R, pR, _, vmR = Logging.with_logger(Logging.NullLogger()) do
    EA.build_evaluator(docC; form = :oop, parameter_overrides = ov,
                       const_arrays = merged_const, param_arrays = merged_param,
                       inspect = insp)
end
const TBUILD_REF = time() - t
say(@sprintf("BUILD reference %.1f s   nstates=%d  nparams=%d  forcing buffers=%d",
             TBUILD_REF, length(u0R), length(pR), length(EA.forcing_buffers(fR))))

# The NEI emission map: const-folded at build time (137,241 source cells), so
# `inspect` is the only way to read the values a capacity build has to be GIVEN.
function insp_array(insp, name)
    for reg in (insp.setup_arrays, insp.const_arrays)
        haskey(reg, name) && return vec(reg[name])
    end
    hits = String[]
    for reg in (insp.setup_arrays, insp.const_arrays)
        append!(hits, [k for k in keys(reg) if endswith(k, "." * split(name, '.')[end])])
    end
    error("capC_probe: `$name` is not in the build inspection " *
          "(near misses: $(first(sort(unique(hits)), min(5, length(hits)))))")
end
EREF = Dict{String,Vector{Float64}}(nm => insp_array(insp, nm) for nm in CapacityChem.EMIS_E)
for (nm, v) in EREF
    length(v) == NLON * NLAT || error("capC_probe: $nm has length $(length(v)), expected $(NLON*NLAT)")
end
say("  NEI emission maps read from the build inspection: " *
    join([@sprintf("%s(%d)", split(k,'.')[2], length(v)) for (k, v) in sort(collect(EREF), by=first)], " "))

# The index-derived geometry, from the templates' own formulas (lat_coord /
# lon_coord / NEIRegrid.lonc, all verified against the document).
pget(nm, dflt) = haskey(pR, Symbol(nm)) ? Float64(getfield(pR, Symbol(nm))) : dflt
const LAT0D = pget("Transport3D.lat0_deg", -90.0 + 4.0 * LAT0)
const LON0D = pget("Transport3D.lon0_deg", -182.5 + 5.0 * LON0)
const DLAT  = pget("Transport3D.dlat_deg", 4.0)
const DLON  = pget("Transport3D.dlon_deg", 5.0)
const NLON0D = pget("NEIRegrid.lon0_deg", LON0D)
const NDLON  = pget("NEIRegrid.dlon_deg", DLON)
LATP = [LAT0D + (j - 1) * DLAT for j in 1:NLAT]
LONP = [LON0D + (i - 0.5) * DLON for i in 1:NLON]
LONC = [NLON0D + (i - 0.5) * NDLON for i in 1:NLON]
AP = Float64.(merged_const["Transport3D.Ap"]); BP = Float64.(merged_const["Transport3D.Bp"])

# --------------------------------------------------------------------------- #
# 2. The capacity build.
# --------------------------------------------------------------------------- #
# Loaded at LON0=LAT0=1 (so the native forcing reads land at `[r, gk, 1+gj, 1+gi]`)
# and at a legal NLEV (>= 6, or the TRANSPORT half's vertical stencil regions
# invert at load); the index-set sizes are rewritten to the lane grid afterwards.
const CAPMP = Dict("NLON" => 70, "NLAT" => 45, "NLEV" => 6, "LON0" => 1, "LAT0" => 1)
t = time(); docCAP0 = split_chem(CAPMP); say(@sprintf("load+split (capacity)  %.1f s", time() - t))

function build_capacity(C::Int)
    cd, meta = CapacityChem.capacity_doc(docCAP0, C; say = say)
    pa = CapacityChem.lane_buffers(meta, merged_param, C)
    ca = Dict{String,Any}(k => v for (k, v) in merged_const if k in meta.variables)
    ovc = Dict{String,Float64}(k => v for (k, v) in ov if k in meta.variables)
    tb = time()
    f, u0, p, _, vm = Logging.with_logger(Logging.NullLogger()) do
        EA.build_evaluator(cd; form = :oop, parameter_overrides = ovc,
                           const_arrays = ca, param_arrays = pa)
    end
    say(@sprintf("BUILD capacity C=%d %.1f s   nstates=%d  nparams=%d  forcing buffers=%d",
                 C, time() - tb, length(u0), length(p), length(EA.forcing_buffers(f))))
    return (; f, u0, p, vm, pa, meta, C, tbuild = time() - tb)
end

# --------------------------------------------------------------------------- #
# 3. Gather + compare.
# --------------------------------------------------------------------------- #
const SPECIES = sort(unique([String(match(r"^(.*)\[\d+,\d+,\d+\]$", n).captures[1])
                             for n in keys(vmR) if occursin(r"\[\d+,\d+,\d+\]$", n)]))
say("  state fields: " * join(SPECIES, ", "))

"reference state index of (species, i, j, k)"
ridx(sp, i, j, k) = vmR[@sprintf("%s[%d,%d,%d]", sp, i, j, k)]

function run_case(B, cells::Vector{NTuple{3,Int}}, uR::Vector{Float64}, t::Float64;
                  gcells::Vector{NTuple{3,Int}} = cells)
    C = B.C
    # `gcells` is what the LANE INPUTS are gathered from; `cells` is what the
    # state is mapped by and what the answer is compared against. They differ
    # only in the negative control, where a deliberately wrong gather has to
    # make the check fail -- otherwise the check proves nothing.
    CapacityChem.gather_forcing!(B.pa, merged_param, gcells; lon0 = LON0, lat0 = LAT0)
    CapacityChem.gather_geometry!(B.pa, B.meta, gcells; Ap = AP, Bp = BP, latp = LATP,
                                  lonp = LONP, E = EREF, lonc = LONC, nlon = NLON)
    uC = copy(B.u0)
    for (l, (i, j, k)) in enumerate(cells)
        i == 0 && continue
        for sp in SPECIES
            uC[B.vm[@sprintf("%s[%d,1,1]", sp, l)]] = uR[ridx(sp, i, j, k)]
        end
    end
    duR = EA.rhs_with_buffers(fR)(uR, pR, t, EA.forcing_buffers(fR))
    duC = EA.rhs_with_buffers(B.f)(uC, B.p, t, EA.forcing_buffers(B.f))
    nfR = count(!isfinite, duR); nfC = count(!isfinite, duC)
    (nfR == 0 && nfC == 0) || say(@sprintf("    !! non-finite: %d of %d reference slots, %d of %d capacity slots",
                                           nfR, length(duR), nfC, length(duC)))
    worst = 0.0; worst_at = ("", 0)
    scale = Dict(sp => maximum(abs, filter(isfinite,
                    [duR[ridx(sp, i, j, k)] for i in 1:NLON, j in 1:NLAT, k in 1:NLEV]))
                 for sp in SPECIES)
    badlanes = Tuple{String,Int,NTuple{3,Int},Float64,Float64}[]
    for (l, (i, j, k)) in enumerate(cells)
        i == 0 && continue
        for sp in SPECIES
            a = duR[ridx(sp, i, j, k)]; b = duC[B.vm[@sprintf("%s[%d,1,1]", sp, l)]]
            den = max(abs(a), scale[sp] * 1e-12, 1e-300)
            # a NON-FINITE difference is the WORST outcome, not a skipped one:
            # `NaN > worst` is false, so an unguarded max silently reports 0.
            r = (isfinite(a) && isfinite(b)) ? abs(a - b) / den : Inf
            if r > worst || (r == Inf && worst < Inf)
                worst = r; worst_at = (sp, l)
            end
            r > 1e-8 && length(badlanes) < 12 && push!(badlanes, (sp, l, (i, j, k), a, b))
        end
    end
    if !isempty(badlanes)
        say("    first mismatching (species, lane, ref cell, du_ref, du_cap):")
        for (sp, l, c, a, b) in badlanes
            say(@sprintf("      %-22s lane %4d  cell %-12s  %.6e  %.6e", sp, l, string(c), a, b))
        end
        bl = unique([x[3] for x in badlanes])
        say(@sprintf("    bad reference cells: %s", join(string.(bl), " ")))
    end
    return worst, worst_at, duR, duC
end

const RNG = MersenneTwister(20260824)
allcells = [(i, j, k) for i in 1:NLON, j in 1:NLAT, k in 1:NLEV][:]
uR = copy(u0R)
uR .*= (1 .+ 0.3 .* rand(RNG, length(uR)))            # a non-degenerate base point

results = Any[]

say("\n---- capacity build at C = NC ----")
B1 = build_capacity(NC)
for (label, cells) in (("identity", copy(allcells)), ("shuffled", shuffle(RNG, copy(allcells))))
    w, at, duR, duC = run_case(B1, cells, uR, T0)
    say(@sprintf("  %-9s |du_ref|: min %.3e  median %.3e  max %.3e over %d slots",
                 label, minimum(abs, duR), median(abs.(duR)), maximum(abs, duR), length(duR)))
    say(@sprintf("  %-9s worst relative |du_cap - du_ref| = %.3e  (at %s lane %d)",
                 label, w, at[1], at[2]))
    push!(results, (label, NC, w, B1.tbuild))
end

# --------------------------------------------------------------------------- #
# 3b. SENSITIVITY DIAGNOSTICS -- does the capacity RHS actually READ its lane
#     buffers? A comparison that passes because nothing is connected proves
#     nothing, so establish the connection explicitly before believing it.
# --------------------------------------------------------------------------- #
let cells = shuffle(RNG, copy(allcells))
    w, at, duR, duC = run_case(B1, cells, uR, T0)
    fb = EA.forcing_buffers(B1.f)
    say(@sprintf("  buffer aliasing: Cap.lat %s   GEOSFP.T %s",
                 getfield(fb, Symbol("Cap.lat")) === B1.pa["Cap.lat"] ? "aliased" : "NOT aliased",
                 getfield(fb, Symbol("GEOSFP.T")) === vec(B1.pa["GEOSFP.T"]) ||
                 pointer(getfield(fb, Symbol("GEOSFP.T"))) == pointer(B1.pa["GEOSFP.T"]) ?
                     "aliased (flat view)" : "NOT aliased"))
    uC = copy(B1.u0)
    for (l, (i, j, k)) in enumerate(cells), sp in SPECIES
        uC[B1.vm[@sprintf("%s[%d,1,1]", sp, l)]] = uR[ridx(sp, i, j, k)]
    end
    base = copy(EA.rhs_with_buffers(B1.f)(uC, B1.p, T0, EA.forcing_buffers(B1.f)))
    function bump!(what, f!)
        f!()
        d = EA.rhs_with_buffers(B1.f)(uC, B1.p, T0, EA.forcing_buffers(B1.f))
        m1 = maximum(abs, [d[B1.vm[@sprintf("%s[1,1,1]", sp)]] - base[B1.vm[@sprintf("%s[1,1,1]", sp)]]
                           for sp in SPECIES])
        mo = 0.0
        for l in 2:B1.C, sp in SPECIES
            sl = B1.vm[@sprintf("%s[%d,1,1]", sp, l)]
            mo = max(mo, abs(d[sl] - base[sl]))
        end
        say(@sprintf("    perturb %-16s -> lane 1 moves %.3e, every OTHER lane moves %.3e", what, m1, mo))
    end
    bump!("Cap.lat[1]+10",   () -> (B1.pa["Cap.lat"][1] += 10.0))
    bump!("Cap.lat[1] back", () -> (B1.pa["Cap.lat"][1] -= 10.0))
    bump!("GEOSFP.T[..,2]x1.1", () -> (B1.pa["GEOSFP.T"][:, 1, 2, 2] .*= 1.1))
    bump!("GEOSFP.T back",      () -> (B1.pa["GEOSFP.T"][:, 1, 2, 2] ./= 1.1))
    bump!("Cap.lev[1]=1",   () -> (B1.pa["Cap.lev"][1] = 1.0))
    say(@sprintf("    (lane 1 carries reference cell %s; its Cap.lev = %.1f, Cap.lat = %.3f)",
                 string(cells[1]), B1.pa["Cap.lev"][1], B1.pa["Cap.lat"][1]))
end

say("\n---- capacity build at C = NC + 37 (padding) ----")
CP = NC + 37
B2 = build_capacity(CP)
cells2 = vcat(shuffle(RNG, copy(allcells)), fill((0, 0, 0), 37))
w, at, _, duClean = run_case(B2, cells2, uR, T0)
say(@sprintf("  padded    worst relative |du_cap - du_ref| = %.3e  (at %s lane %d)",
             w, at[1], at[2]))
push!(results, ("padded", CP, w, B2.tbuild))

# A NEGATIVE CONTROL first: the comparison has to be able to FAIL. Corrupt one
# lane's gathered geometry and require the very same check to report it.
let bad = copy(cells2)
    bad[1], bad[2] = bad[2], bad[1]
    wbad, atbad, _, _ = run_case(B2, cells2, uR, T0; gcells = bad)
    say(@sprintf("  NEGATIVE CONTROL (two lanes' INPUTS swapped, answers not): worst relative = %.3e  (at %s lane %d)",
                 wbad, atbad[1], atbad[2]))
    wbad > 1e-6 || error("capC_probe: the comparison did not notice a deliberately " *
                         "wrong gather -- it proves nothing")
end
# Restore the correct gather, then check a padding lane cannot reach a real one:
# push every padding lane's inputs off by a large but still PHYSICAL factor (an
# absurd value just makes the padding lane itself throw on log(PS/Pc)) and
# require every real lane's derivative to be bit-identical.
w, at, _, _ = run_case(B2, cells2, uR, T0)
uC = copy(B2.u0)
for (l, (i, j, k)) in enumerate(cells2)
    i == 0 && continue
    for sp in SPECIES; uC[B2.vm[@sprintf("%s[%d,1,1]", sp, l)]] = uR[ridx(sp, i, j, k)]; end
end
before = EA.rhs_with_buffers(B2.f)(uC, B2.p, T0, EA.forcing_buffers(B2.f))
# Re-gather with the PADDING lanes pointed at a completely different real cell.
# (Scaling their buffers instead would be simpler but drives the padding lane's
#  own arithmetic out of domain -- log(PS/Pc) -- which tests nothing.)
let cells3 = copy(cells2)
    for l in (NC + 1):CP; cells3[l] = allcells[end - (l - NC)]; end
    CapacityChem.gather_forcing!(B2.pa, merged_param, cells3; lon0 = LON0, lat0 = LAT0)
    CapacityChem.gather_geometry!(B2.pa, B2.meta, cells3; Ap = AP, Bp = BP, latp = LATP,
                                  lonp = LONP, E = EREF, lonc = LONC, nlon = NLON)
end
after = EA.rhs_with_buffers(B2.f)(uC, B2.p, T0, EA.forcing_buffers(B2.f))
leak = let m = 0.0
    for (l, (i, j, k)) in enumerate(cells2)
        i == 0 && continue
        for sp in SPECIES
            sl = B2.vm[@sprintf("%s[%d,1,1]", sp, l)]
            m = max(m, abs(after[sl] - before[sl]))
        end
    end
    m
end
say(@sprintf("  padding-perturbation leak into the %d real lanes: max |Δdu| = %.3e", NC, leak))
# and the padding lanes DID move, or the perturbation tested nothing
padmoved = let m = 0.0
    for l in (NC + 1):CP, sp in SPECIES
        sl = B2.vm[@sprintf("%s[%d,1,1]", sp, l)]
        m = max(m, abs(after[sl] - before[sl]))
    end
    m
end
say(@sprintf("  padding lanes finite before the perturbation: %s",
             all(isfinite, [before[B2.vm[@sprintf("%s[%d,1,1]", sp, l)]]
                            for l in (NC + 1):CP, sp in SPECIES]) ? "yes" : "NO -- they would NaN the error norm"))
say(@sprintf("  (the padding lanes themselves moved by max |Δdu| = %.3e)", padmoved))

say("\n" * "="^78)
for (label, C, w, tb) in results
    say(@sprintf("RESULT case=%-9s C=%-5d worst_rel=%.3e  capacity build=%.1f s  (reference build %.1f s)",
                 label, C, w, tb, TBUILD_REF))
end
say(@sprintf("RESULT leak=%.3e", leak))
say("="^78)
