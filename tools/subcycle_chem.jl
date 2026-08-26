# ===========================================================================
# subcycle_chem.jl -- LEVEL-BASED PER-CELL TIME STEPPING, the forward path.
# ===========================================================================
# This is the WIRING between the two halves that already exist:
#
#   tools/capacity_chem.jl   the capacity-C chemistry BUILD (doc surgery, lane
#                            buffers, forcing/geometry gather). Verified to
#                            2.888e-14 against the reference RHS cell for cell,
#                            with 0.000e+00 leak from padding lanes.
#   tools/level_subcycle.jl  the SCHEDULER (assign_levels / pack / cost), pure
#                            index algebra, `selftest()` runs in a second.
#
# Neither of them touches a solver or a device. This file does: it builds a
# LADDER of capacity rungs, compiles `ros23_step` (+ the symbolic-Jacobian `gJ`)
# at each, and runs a macro window as a batch list instead of one global dt.
#
# It is a PLAIN INCLUDE, not a module, on purpose: it needs the driver's `EA`,
# `RX`, `RTI`, `RxSymBlockJac`, `NS`, `NC`, `var_map`, `merged_param`,
# `merged_const`, `ov`, `COPTS`, `JACMODE`, `CLAMP`, `ATOL_C`, `RTOL` and the
# grid metaparameters, and a nested Julia module does not inherit the enclosing
# scope's names. tools/diag/subcycle_verify.jl includes the driver the same way
# tools/diag/level_schedule.jl does, so the whole thing stays reachable.
#
# ---------------------------------------------------------------------------
# WHY A LADDER (do not collapse it to one capacity). Level j costs
# `2^j x (lanes run)` and a level's lanes round up to a whole chunk of the
# build's capacity C. At CONUS levels 6-8 hold 0.5% of the cell-windows and at a
# single C = 1024 cost MORE THAN HALF the total: the dyadic scheme's 5.14x is
# spent entirely on padding and the result is 0.75x -- SLOWER than the global
# controller it replaces. Single C = 4096 is 0.27x. Measured ladders (slurm
# 10123660): 16/64/256/1024 -> 4.91x at 4.6% waste, 8/32/128/512/2048 -> 5.07x
# at 1.5% waste. The ladder is affordable only because a capacity build is
# seconds (11.4 s at C = 4096 against 650 s for the CONUS reference).
#
# WHAT THIS MEASURED, AND IT IS NOT WHAT WAS EXPECTED (slurm 10127723 / 10128099,
# 6x6x8, RESEACT_SUB_NW=2, ladder 8/32/128/512).
#
#   RESULT cellstep_speedup 0.023x     RESULT WALL_speedup   0.021x
#   RESULT calls 213 global vs 27659   RESULT ms_per_call    8.6 global vs 3.1 sub
#   padding waste 0.0%                 gather+permute < 1% of window wall time
#
# THE PER-CALL DISPATCH OVERHEAD IS NOT THE PROBLEM. It was the main worry going
# in -- a ladder issues thousands of small calls where the global controller
# issues ~200 large ones -- but a subcycled call costs 3.1 ms against the global
# program's 8.6 ms, and wall speedup tracks cell-step speedup to two digits
# (0.021 vs 0.023). Dispatch is fully amortised; the ladder mechanics are sound.
#
# THE PROBLEM IS THAT A LEVEL IS 2^j EQUAL STEPS, so it must resolve the cell's
# WORST INSTANT over the whole window, while the 7.53x/5.14x projections price
# `SUM dd/dtc(t)` -- the count for a per-cell VARIABLE dt. Measured per cell:
#
#   CONUS  ratio worst/integral median  5.1x (max 655x); cell-steps vs global
#          4.914e5: INTEGRAL 1.177e5 = 4.18x, but a LEVEL 2.363e6 = 0.21x
#   6x6x8  ratio median 37.5x (max 208x); INTEGRAL 2.71x, a LEVEL 0.11x
#
# Even with a PERFECT predictor the level scheme lands ~20x below the figure it
# was projected from. At CONUS it does not even terminate at a sane cap: a
# 512-cell batch still rejected at LEVEL 14 (16,384 equal substeps of 0.018 s).
# tools/level_subcycle.jl's header carries the full table; check 3b of
# tools/diag/subcycle_verify.jl is what measures it, and its integral histogram
# reproduces level_schedule.jl's published CONUS histogram term for term -- which
# is the control that makes the two numbers comparable.
#
# Part of the gap against the GLOBAL controller is that the subcycle solves a
# STRICTER problem: the global test is an RMS over the whole domain, the per-cell
# test is a max, and at CONUS max_c(ce)/EEst runs a median 20.8x. That does NOT
# explain the gap against the 4.18x projection, which uses the same per-cell
# target -- but it does mean 0.21x is not a like-for-like accuracy comparison.
#
# So this file is a WORKING and VERIFIED forward path whose scheduling policy
# does not pay at MACRO_DT = 300 s. The wiring below -- capacity ladder, gather,
# permute, predictor, rejection -- is what a fixed policy would reuse unchanged;
# what has to change is `predict_need` and the window length, not the machinery.
# `stats.t_*` splits each window's wall time into gather / permute / device so
# the next policy is judged on a measurement rather than a projection.
#
# ---------------------------------------------------------------------------
# ACCEPT CRITERION. The global controller accepts on the RMS error over the
# WHOLE domain; a batch accepts on the MAX over its REAL lanes (`cellwise=true`
# gives the per-cell norm for one extra reduce). Two reasons, and the first is
# not optional:
#   1. PADDING LANES DUPLICATE A REAL CELL (capacity_chem.jl `_filled`), so an
#      RMS over the batch is diluted by however many copies the padding made --
#      a batch of 3 cells at capacity 8 would average one cell's error five
#      extra times and could accept a step that another of its three cells
#      failed. A max over real lanes cannot be diluted.
#   2. It is the criterion the PREDICTOR targets: `need` comes from the per-cell
#      error extrapolated to ce = 1, exactly as tools/diag/cell_stiffness.jl and
#      level_schedule.jl priced it. Accepting on a domain RMS would let cells
#      through that the schedule had already decided were under-resolved.
# Max-over-cells >= RMS-over-cells, so this is STRICTER than the global
# controller, never looser: the subcycled trajectory is not being bought with a
# weaker error test.
#
# REJECTION. A rejected batch is re-run FROM ITS OWN WINDOW START one level up
# (`2^(j+1)` steps at half the dt), which is what "promote those cells one
# level" means when the level is the only knob. The work already done in that
# batch is discarded; that is the honest cost and it is counted in `stats`.
# ===========================================================================

include(joinpath(REPO, "tools", "capacity_chem.jl"));  using .CapacityChem
include(joinpath(REPO, "tools", "level_subcycle.jl")); using .LevelSubcycle

const SUB_LADDER = sort(parse.(Int, split(get(ENV, "RESEACT_SUBCYCLE_LADDER",
                                             "8,32,128,512,2048"), ',')))
const SUB_JMAX   = parse(Int,     get(ENV, "RESEACT_SUBCYCLE_JMAX", "10"))
const SUB_SAFETY = parse(Float64, get(ENV, "RESEACT_SUBCYCLE_SAFETY", "1.0"))
const SUB_MAXPROMO = parse(Int,   get(ENV, "RESEACT_SUBCYCLE_MAXPROMO", "3"))
# The capacity document is loaded at a LEGAL grid (NLEV >= 6 or the transport
# half's vertical stencil regions invert at load) and at LON0 = LAT0 = 1 so the
# native forcing reads land at `[r, gk, 1+gj, 1+gi]`; the index-set sizes are
# rewritten to the C x 1 x 1 lane grid afterwards. See capacity_chem.jl.
const SUB_CAPMP = Dict("NLON" => 70, "NLAT" => 45, "NLEV" => 6, "LON0" => 1, "LAT0" => 1)

# --------------------------------------------------------------------------- #
# The runner's own layout, read back off `var_map`.
# --------------------------------------------------------------------------- #
# `species_masks` has already asserted that the state vector is species-major
# with one contiguous run of NC per species and the SAME cell sequence in every
# run; this just reads that layout out so a lane can be pointed at a cell.
function runner_layout(vm, NSl::Int, NCl::Int)
    sp = Vector{String}(undef, NSl)
    cells = Vector{NTuple{3,Int}}(undef, NCl)
    for (nm, idx) in vm
        m = match(r"^(.*)\[(\d+),(\d+),(\d+)\]$", nm)
        m === nothing && error("runner_layout: unparseable state name `$nm`")
        s = (idx - 1) ÷ NCl + 1
        c = (idx - 1) % NCl + 1
        sp[s] = String(m.captures[1])
        cells[c] = (parse(Int, m.captures[2]), parse(Int, m.captures[3]),
                    parse(Int, m.captures[4]))
    end
    return sp, cells
end

# --------------------------------------------------------------------------- #
# Reference geometry: the eight index-derived lane inputs, plus the NEI maps.
# --------------------------------------------------------------------------- #
# `NEIRegrid.E_*` is the conservative regrid of the 137,241-cell inventory,
# CONST-FOLDED at build time -- it exists nowhere at runtime, so the only way to
# read the values a capacity build has to be GIVEN is the build inspection. That
# is why the driver hands `inspect=` to its part-2 build when the subcycle is on
# (and only then: `build_evaluator`'s return is documented identical either way).
function insp_array(insp, name)
    for reg in (insp.setup_arrays, insp.const_arrays)
        haskey(reg, name) && return vec(Float64.(reg[name]))
    end
    return nothing
end

function reference_geometry(insp, pR, mconst, mp)
    nlon, nlat = mp["NLON"], mp["NLAT"]
    lon0, lat0 = mp["LON0"], mp["LAT0"]
    pget(nm, dflt) = haskey(pR, Symbol(nm)) ? Float64(getfield(pR, Symbol(nm))) : dflt
    lat0d = pget("Transport3D.lat0_deg", -90.0 + 4.0 * lat0)
    lon0d = pget("Transport3D.lon0_deg", -182.5 + 5.0 * lon0)
    dlat  = pget("Transport3D.dlat_deg", 4.0)
    dlon  = pget("Transport3D.dlon_deg", 5.0)
    nl0d  = pget("NEIRegrid.lon0_deg", lon0d)
    ndlon = pget("NEIRegrid.dlon_deg", dlon)
    # The FORMULAS are the verified path: tools/diag/capC_probe.jl reproduced the
    # reference RHS to 2.888e-14 cell for cell under a random cell permutation
    # using exactly these, and nothing has verified an inspection array in their
    # place. So the formula is what is used -- and where the build DID
    # materialise the array, it is cross-checked rather than preferred, because a
    # silent disagreement here puts every lane at the wrong latitude with no
    # symptom other than a slightly wrong photolysis rate.
    latp = [lat0d + (j - 1) * dlat for j in 1:nlat]
    lonp = [lon0d + (i - 0.5) * dlon for i in 1:nlon]
    lonc = [nl0d + (i - 0.5) * ndlon for i in 1:nlon]
    for (nm, want) in ("Transport3D.latp" => latp, "Transport3D.lonp" => lonp,
                       "NEIRegrid.lonc" => lonc)
        got = insp_array(insp, nm)
        got === nothing && continue
        # Compare only when the build materialised the array at the SAME extent.
        # A different length is a different array (a native-indexed table, say),
        # not a disagreement, and turning that into a hard error would kill a run
        # over a coincidence of naming.
        if length(got) != length(want)
            say(@sprintf("  (build array `%s` has %d entries against the coordinate formula's %d; not comparable, formula used)",
                         nm, length(got), length(want)))
            continue
        end
        d = maximum(abs, got .- want)
        # `@sprintf` needs a LITERAL format string: `"a" * "b"` fails at MACRO
        # EXPANSION, not at runtime, so a concatenated one dies the moment the
        # file is loaded. Every format string below is therefore one string.
        d <= 1e-9 * max(1.0, maximum(abs, want)) || error(@sprintf("reference_geometry: `%s` from the build differs from the coordinate formula by %.3e; the template changed and the lane gather would put cells at the wrong coordinates", nm, d))
    end
    E = Dict{String,Vector{Float64}}()
    for nm in CapacityChem.EMIS_E
        v = insp_array(insp, nm)
        v === nothing && error("subcycle: `$nm` is not in the build inspection -- the " *
                               "NEI regrid is const-folded, so a capacity build cannot " *
                               "be given its values without one")
        length(v) == nlon * nlat ||
            error("subcycle: $nm has length $(length(v)), expected $(nlon*nlat)")
        E[nm] = v
    end
    ap = Float64.(mconst["Transport3D.Ap"]); bp = Float64.(mconst["Transport3D.Bp"])
    return (; latp, lonp, lonc, E, ap, bp, nlon, nlat, lon0, lat0)
end

# --------------------------------------------------------------------------- #
# A ladder rung.
# --------------------------------------------------------------------------- #
struct SubRung
    C::Int
    f
    u0h::Vector{Float64}
    p
    vm::Dict{String,Int}
    pa::Dict{String,Any}
    meta
    th
    jacE
    plan
    gjb
    dev_bufs
    dev_bufsJ
    cstep
    capsel::Vector{Int}     # (s-1)*C + l  ->  capacity state index, s in RUNNER species order
    tbuild::Float64
    tjac::Float64
    tcomp::Float64
end

"""
    gather_lanes!(pa, meta, geom, lane_cells)

Fill one capacity build's lane buffers from the reference grid: the GEOS-FP
forcing (both time records, so the interpolation stays a function of `t`) and
the eight index-derived geometry inputs.

`lane_cells` is one reference `(i, j, k)` PER LANE, including the padding lanes.
A padding lane is not "off" -- the compiled program evaluates it like any other
and its result lands in the same vector the error norm reduces over -- so every
lane must carry a real cell's inputs or the batch NaNs. See capacity_chem.jl.
"""
function gather_lanes!(pa, meta, geom, lane_cells::Vector{NTuple{3,Int}})
    CapacityChem.gather_forcing!(pa, merged_param, lane_cells;
                                 lon0 = geom.lon0, lat0 = geom.lat0)
    CapacityChem.gather_geometry!(pa, meta, lane_cells; Ap = geom.ap, Bp = geom.bp,
                                  latp = geom.latp, lonp = geom.lonp, E = geom.E,
                                  lonc = geom.lonc, nlon = geom.nlon)
    return pa
end

# The traced step at one rung. Written as a closure factory so `@compile` gets a
# call expression whose callee carries the rung's own RHS, plan and sizes; the
# host structure rides in the closure exactly the way `gC`/`gJ` ride in the
# driver's globals.
function _mk_rung_step(frhs, plan, gjb, NSl::Int, Cl::Int, masks)
    sj = plan === nothing ? nothing :
        ((uu, tt, th) -> RxSymBlockJac.block_jac(plan,
            gjb(RxSymBlockJac.gather_uj(plan, uu), th.p, tt, th.bufsJ)))
    return (u, th, t, dt) -> RTI.ros23_step((uu, tt) -> frhs(uu, th.p, tt, th.bufs),
        u, t, dt, NSl, Cl, masks, ATOL_C, RTOL;
        unrolled = true, jac = JACMODE,
        symjac = sj === nothing ? nothing : ((uu, tt) -> sj(uu, tt, th)),
        cellwise = true)
end

function build_rung(docCAP0, C::Int, spnames::Vector{String}, geom,
                    cells::Vector{NTuple{3,Int}}; say = say)
    cd, meta = CapacityChem.capacity_doc(docCAP0, C; say = say)
    pa  = CapacityChem.lane_buffers(meta, merged_param, C)
    ca  = Dict{String,Any}(k => v for (k, v) in merged_const if k in meta.variables)
    ovc = Dict{String,Float64}(k => v for (k, v) in ov if k in meta.variables)
    tb = time()
    f, u0c, pc, _, vmc = Logging.with_logger(Logging.NullLogger()) do
        EA.build_evaluator(cd; form = :oop, parameter_overrides = ovc,
                           const_arrays = ca, param_arrays = pa)
    end
    tbuild = time() - tb
    length(u0c) == NS * C ||
        error("build_rung: capacity build at C=$C has $(length(u0c)) states, " *
              "expected NS*C = $(NS*C) -- the species set differs from the runner's")
    masks = RTI.species_masks(vmc, NS, C)     # also re-asserts the species-major layout
    # (s, lane) -> capacity state index, with `s` in the RUNNER's species order.
    capsel = Vector{Int}(undef, NS * C)
    for s in 1:NS, l in 1:C
        nm = @sprintf("%s[%d,1,1]", spnames[s], l)
        haskey(vmc, nm) || error("build_rung: the C=$C build has no state `$nm`; the " *
                                 "capacity build's species set differs from the runner's")
        capsel[(s - 1) * C + l] = vmc[nm]
    end

    # PRIME THE LANE BUFFERS BEFORE ANYTHING EVALUATES THIS BUILD, and this is
    # not tidiness. `lane_buffers` hands back ZEROS, and a zero lane is not an
    # off lane: the RHS runs it and returns NaN through `log(PS/Pc)`. The very
    # first thing to evaluate the build is `validate_plan` below, which then
    # reports a worst-case of NaN -- and the guard, correctly, refuses the rung.
    # (Measured: slurm 10127512 died exactly here at C=8.) Point every lane at
    # reference cell 1; the first real batch overwrites all of it.
    gather_lanes!(pa, meta, geom, fill(cells[1], C))
    let du = EA.rhs_with_buffers(f)(u0c, pc, T0, EA.forcing_buffers(f))
        nb = count(!isfinite, du)
        nb == 0 || error("build_rung: the C=$C RHS returns $nb of $(length(du)) NON-FINITE derivatives at the primed base point; the lane gather is not reaching this build")
    end

    jacE = plan = gjb = dev_bufsJ = nothing; tjac = 0.0
    if SYMJAC
        tj = time()
        jacE = Logging.with_logger(Logging.NullLogger()) do
            EarthSciASTDiff.prepare_jacobian(EA.coerce_esm_file(cd); model_name = "Flattened",
                wrt = :states, build_kwargs = (; form = :oop, parameter_overrides = ovc,
                                                 const_arrays = ca, param_arrays = pa))
        end
        tjac = time() - tj
        jacE.oop || error("build_rung: the C=$C band model came back IN-PLACE; it " *
                          "captures host scratch per node and cannot be traced")
        String(jacE.structure) == "block_diagonal" ||
            error("build_rung: the C=$C Jacobian is $(jacE.structure), not block_diagonal")
        plan = RxSymBlockJac.block_jac_plan(jacE;
                    runner_names = first.(sort(collect(vmc), by = last)))
        gjb  = EA.rhs_with_buffers(jacE.fJ!)
        dev_bufsJ = map(RX.ConcreteRArray, EA.forcing_buffers(jacE.fJ!))
        # One host evaluation, and it is the only thing between a transposed
        # gather and a plausible Jacobian that is wrong in every lane.
        let w = validate_plan(plan, jacE, u0c, pc, T0;
                              gjb = gjb, bufs = EA.forcing_buffers(jacE.fJ!))
            w <= 1e-12 || error("build_rung: the C=$C gather plan does not reproduce " *
                                "the host Jacobian (worst relative $w)")
        end
        say(@sprintf("    C=%-5d prepare_jacobian %6.1f s  %s", C, tjac, string(plan)))
    end

    dev_bufs = map(RX.ConcreteRArray, EA.forcing_buffers(f))
    th = thC(_devp(pc), dev_bufs, dev_bufsJ)
    stepfn = _mk_rung_step(EA.rhs_with_buffers(f), plan, gjb, NS, C, masks)
    UD = RX.ConcreteRArray(copy(u0c))
    TD = RX.ConcreteRNumber(T0); DD = RX.ConcreteRNumber(DT0C)
    tc = time()
    cstep = RX.@compile compile_options=COPTS stepfn(UD, th, TD, DD)
    tcomp = time() - tc
    say(@sprintf("    C=%-5d build %5.1f s  compile %6.1f s  (nstates=%d, buffers=%d)",
                 C, tbuild, tcomp, length(u0c), length(dev_bufs)))
    return SubRung(C, f, copy(u0c), pc, vmc, pa, meta, th, jacE, plan, gjb,
                   dev_bufs, dev_bufsJ, cstep, capsel, tbuild, tjac, tcomp)
end

# --------------------------------------------------------------------------- #
# The ladder.
# --------------------------------------------------------------------------- #
struct SubLadder
    rungs::Dict{Int,SubRung}
    caps::Vector{Int}
    geom
    spnames::Vector{String}
    cells::Vector{NTuple{3,Int}}    # runner cell position -> reference (i, j, k)
    tload::Float64
end

function build_subcycle_ladder(insp; caps::Vector{Int} = SUB_LADDER)
    spnames, cells = runner_layout(var_map, NS, NC)
    geom = reference_geometry(insp, p, merged_const, GRID_MP)
    say("\n---- building the capacity LADDER " * string(caps) * " ----")
    tl = time()
    docCAP0 = Logging.with_logger(Logging.NullLogger()) do
        file = EA.load_path(MODEL; metaparameters = SUB_CAPMP)
        flat = EA.flatten(file)
        pre  = EA.algebraic_states_to_observeds(flat)
        flat = EA.promote_downstream_shapes(pre)
        promoted = EA.promoted_array_names(pre, flat)
        parts = split_system(flat, stencil_following_rule(flat); nparts = 2)
        index_promoted_refs_by_loop!(EA.flattened_to_esm(parts[2]), promoted)
    end
    tload = time() - tl
    say(@sprintf("  load+split (capacity document) %.1f s", tload))
    rungs = Dict{Int,SubRung}()
    for C in caps
        rungs[C] = build_rung(docCAP0, C, spnames, geom, cells)
    end
    tb = sum(r.tbuild for r in values(rungs))
    tj = sum(r.tjac for r in values(rungs))
    tc = sum(r.tcomp for r in values(rungs))
    say(@sprintf("  LADDER READY: %d rungs, builds %.1f s + jacobians %.1f s + compiles %.1f s (+ %.1f s load) = %.1f s one-off",
                 length(caps), tb, tj, tc, tload, tb + tj + tc + tload))
    return SubLadder(rungs, sort(collect(caps)), geom, spnames, cells, tload)
end

# --------------------------------------------------------------------------- #
# One batch.
# --------------------------------------------------------------------------- #
mutable struct SubStats
    calls::Int          # compiled-program invocations (the thing XLA dispatch is paid per)
    lanesteps::Float64  # capacity x substeps, padding included
    cellsteps::Float64  # real cells x substeps (what a per-cell scheme "should" cost)
    rejects::Int
    batches::Int
    promoted::Int
    t_gather::Float64
    t_perm::Float64
    t_dev::Float64
    t_total::Float64
    maxlevel::Int
end
SubStats() = SubStats(0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)

function Base.:+(a::SubStats, b::SubStats)
    SubStats(a.calls + b.calls, a.lanesteps + b.lanesteps, a.cellsteps + b.cellsteps,
             a.rejects + b.rejects, a.batches + b.batches, a.promoted + b.promoted,
             a.t_gather + b.t_gather, a.t_perm + b.t_perm, a.t_dev + b.t_dev,
             a.t_total + b.t_total, max(a.maxlevel, b.maxlevel))
end

"Gather this batch's forcing and geometry into the rung's lane buffers, and push."
function batch_gather!(L::SubLadder, r::SubRung, lane_cells::Vector{NTuple{3,Int}})
    gather_lanes!(r.pa, r.meta, L.geom, lane_cells)
    EA.sync_forcing!(r.dev_bufs, EA.forcing_buffers(r.f))
    SYMJAC && EA.sync_forcing!(r.dev_bufsJ, EA.forcing_buffers(r.jacE.fJ!))
    return nothing
end

# `lanepos[l]` is the RUNNER cell position lane l carries (padding lanes carry a
# duplicate of a real one, so every lane is arithmetically valid -- a lane left
# at zero returns NaN through log(PS/Pc) and NaNs the whole batch's error norm).
function lanes_in!(uc::Vector{Float64}, u::Vector{Float64}, r::SubRung,
                   lanepos::Vector{Int})
    C = r.C; sel = r.capsel
    @inbounds for s in 1:NS
        br = (s - 1) * NC; bc = (s - 1) * C
        for l in 1:C
            uc[sel[bc + l]] = u[br + lanepos[l]]
        end
    end
    return uc
end

function lanes_out!(u::Vector{Float64}, uc::Vector{Float64}, r::SubRung,
                    lanepos::Vector{Int}, nreal::Int)
    C = r.C; sel = r.capsel
    @inbounds for s in 1:NS
        br = (s - 1) * NC; bc = (s - 1) * C
        for l in 1:nreal
            u[br + lanepos[l]] = uc[sel[bc + l]]
        end
    end
    return u
end

"""
    run_batch!(u, L, b, t0, W, st) -> j_final

Advance `b.cells` across `[t0, t0+W]` with `2^j` steps of `W/2^j` on the rung of
capacity `b.cap`, promoting a level at a time on rejection. Writes the result
back into the runner's state vector `u` in place.
"""
function run_batch!(u::Vector{Float64}, L::SubLadder, b::LevelSubcycle.Batch,
                    t0::Float64, W::Float64, st::SubStats)
    r = L.rungs[b.cap]
    nreal = length(b.cells)
    nreal <= b.cap || error("run_batch!: $(nreal) cells do not fit capacity $(b.cap)")
    # lane -> runner cell position; padding lanes repeat real lanes ROUND-ROBIN
    # rather than all repeating lane 1, so a padded batch's device work is a
    # faithful copy of the real work rather than one cell run many times.
    lanepos = Vector{Int}(undef, b.cap)
    @inbounds for l in 1:b.cap
        lanepos[l] = b.cells[l <= nreal ? l : (mod1(l - nreal, nreal))]
    end
    lane_cells = [L.cells[c] for c in lanepos]

    tg = time(); batch_gather!(L, r, lane_cells); st.t_gather += time() - tg
    tp = time()
    uc0 = lanes_in!(copy(r.u0h), u, r, lanepos)
    st.t_perm += time() - tp

    j = b.j
    while true
        nsub = 1 << j
        dt = W / nsub
        uc = copy(uc0)
        ok = true
        ndone = 0
        for k in 0:(nsub - 1)
            t = t0 + k * dt
            td = time()
            res = r.cstep(RX.ConcreteRArray(uc), r.th, RX.ConcreteRNumber(t),
                          RX.ConcreteRNumber(dt))
            raw = Array(res[1]); ce = Array(res[3])
            st.t_dev += time() - td
            st.calls += 1
            # MAX over the REAL lanes, and NaN-safe: `worst > 1` is false for a
            # NaN, so an unguarded max silently accepts an all-NaN batch. This
            # is the same trap that let three earlier runs "pass" with every
            # lane NaN, so the test is written as `!(x <= 1)`.
            worst = 0.0
            @inbounds for l in 1:nreal
                e = ce[l]
                worst = isfinite(e) ? max(worst, e) : Inf
            end
            ndone += 1
            if !(worst <= 1.0)
                ok = false
                break
            end
            uc = CLAMP[] ? max.(raw, 0.0) : raw
        end
        # `ndone`, not `nsub`: a rejected batch stops at the substep that failed,
        # and billing it for the whole level would flatter every rejection into
        # looking like work that was actually done.
        st.lanesteps += ndone * b.cap
        st.cellsteps += ndone * nreal
        if ok
            st.batches += 1
            st.maxlevel = max(st.maxlevel, j)
            tp2 = time(); lanes_out!(u, uc, r, lanepos, nreal); st.t_perm += time() - tp2
            return j
        end
        st.rejects += 1; st.promoted += 1
        j += 1
        j <= SUB_JMAX + SUB_MAXPROMO ||
            error(@sprintf("run_batch!: level %d (%d substeps) still rejects for a batch of %d cells over [%.3f, %.3f]; the window is not resolvable by halving alone",
                           j, 1 << j, nreal, t0, t0 + W))
    end
end

# --------------------------------------------------------------------------- #
# The predictor.
# --------------------------------------------------------------------------- #
# `need[c]` is how many equal substeps cell c wants over the window. Two sources,
# combined by MAX because under-resolving is what costs a rejected batch (2x the
# work) while over-resolving costs one dyadic bucket:
#   * ONE PROBE STEP of the FULL-GRID cellwise program at the window's start.
#     `dtc = dt * (1/ce)^(1/3)` is the embedded pair's own exponent (ROS23 is
#     order 2 with an order-3 embedded error, so the norm scales as dt^3), which
#     is exactly what tools/diag/cell_stiffness.jl priced the 7.53x with.
#   * THE PREVIOUS WINDOW's realised level, which carries the information a
#     single probe at t0 cannot: a cell that only becomes stiff mid-window.
# The probe's OUTPUT IS DISCARDED -- the subcycle starts from `u`, not from a
# state that took one global step. It costs one program call out of ~1,400.
function predict_need(u::Vector{Float64}, t0::Float64, W::Float64,
                      need_prev::Union{Nothing,Vector{Float64}}, dt_probe::Float64,
                      st::SubStats)
    dtp = min(dt_probe, W)
    td = time()
    res = CROSCW(RX.ConcreteRArray(u), THC, RX.ConcreteRNumber(t0),
                 RX.ConcreteRNumber(dtp))
    ce = Array(res[3])
    # the probe is a FULL-GRID device call and is billed as one: it is the single
    # most expensive call the window makes, and hiding it in "other" would make
    # the per-call figure this file exists to measure look better than it is.
    st.t_dev += time() - td
    st.calls += 1
    need = Vector{Float64}(undef, NC)
    @inbounds for c in 1:NC
        e = ce[c]
        # a non-finite per-cell norm means the probe step blew up in that cell;
        # ask for the ceiling rather than silently proposing one step.
        dtc = isfinite(e) && e > 0 ? dtp * (1.0 / e)^(1 / 3) : dtp / (1 << SUB_JMAX)
        need[c] = SUB_SAFETY * W / max(dtc, 1e-12)
    end
    # THE MEMORY MUST DECAY. `max(probe, need_prev)` with `need_prev` set to the
    # level a window actually realised is a RATCHET: a cell promoted once can
    # never come back down, because its own history keeps re-proposing the level
    # it was promoted to. Measured on slurm 10127723, that carried cells to level
    # 13 and never released them. Halving the memory lets a cell drop ONE LEVEL
    # PER WINDOW -- fast enough to track a relaxing plume, slow enough that a
    # cell which is stiff every window stays where it belongs, and a rejection
    # puts it straight back up.
    need_prev === nothing || (need .= max.(need, need_prev ./ 2))
    return need, ce
end

# --------------------------------------------------------------------------- #
# One macro window of chemistry, level-scheduled.
# --------------------------------------------------------------------------- #
"""
    subcycle_chem(L, u, t0, t1; need_prev, dt_probe) -> (u_new, need_next, stats)

The drop-in replacement for `host_adaptive!(CROS, ...)` over one macro window.
`u` is not modified; the returned vector is a copy with the chemistry half
advanced. `need_next` is the realised per-cell step demand, to be fed back as
`need_prev` on the next window.
"""
function subcycle_chem(L::SubLadder, u::Vector{Float64}, t0::Float64, t1::Float64;
                       need_prev::Union{Nothing,Vector{Float64}} = nothing,
                       dt_probe::Float64 = DT0C)
    W = t1 - t0
    st = SubStats()
    ttot = time()
    need, _ = predict_need(u, t0, W, need_prev, dt_probe, st)
    levels = LevelSubcycle.assign_levels(need, SUB_JMAX)
    batches = LevelSubcycle.pack(levels, SUB_LADDER; jmax = SUB_JMAX)
    for b in batches
        b.cap in keys(L.rungs) ||
            error("subcycle_chem: no rung for capacity $(b.cap); the ladder is " *
                  "$(L.caps) and the packer asked for $(b.cap)")
    end
    need_next = Vector{Float64}(undef, NC)
    uo = copy(u)
    for b in batches
        jf = run_batch!(uo, L, b, t0, W, st)
        @inbounds for c in b.cells
            need_next[c] = 2.0^jf
        end
    end
    st.t_total = time() - ttot
    return uo, need_next, st
end

# --------------------------------------------------------------------------- #
# Cross-window state. The predictor's memory, and the running totals.
# --------------------------------------------------------------------------- #
const SUB_NEED    = Ref{Union{Nothing,Vector{Float64}}}(nothing)
const SUB_DTPROBE = Ref(DT0C)
const SUB_STATS   = Ref(SubStats())
const SUB_LAST    = Ref(SubStats())

"""
    subcycle_report(st; nglobal_steps = 0)

`nglobal_steps` is the number of ACCEPTED global-dt chemistry steps the same
window took, when it is known (it is not known on a subcycled run -- that is
what the A/B against RESEACT_SUBCYCLE=0 is for). Pass 0 to omit the ratio rather
than print one against a step count that means something else: under the
subcycle `counts[3]` is the CALL count, not a global step count, and reporting
one as the other is how a speedup gets invented.
"""
function subcycle_report(st::SubStats; nglobal_steps::Int = 0)
    say(@sprintf("    subcycle: %d batches, %d program calls, %d rejections, max level %d",
                 st.batches, st.calls, st.rejects, st.maxlevel))
    if nglobal_steps > 0
        say(@sprintf("    cell-steps %.4g vs global %.4g  =>  %.2fx  (calls %d vs %d)",
                     st.cellsteps, Float64(nglobal_steps) * NC,
                     Float64(nglobal_steps) * NC / max(st.cellsteps, 1),
                     st.calls, nglobal_steps))
    else
        say(@sprintf("    cell-steps %.4g", st.cellsteps))
    end
    say(@sprintf("    lane-steps %.4g (padding waste %.1f%%)",
                 st.lanesteps, 100 * (st.lanesteps - st.cellsteps) / max(st.lanesteps, 1)))
    say(@sprintf("    wall %.2f s = device %.2f s + gather %.2f s + permute %.2f s + %.2f s other   (%.2f ms/call on the device)",
                 st.t_total, st.t_dev, st.t_gather, st.t_perm,
                 st.t_total - st.t_dev - st.t_gather - st.t_perm,
                 1000 * st.t_dev / max(st.calls, 1)))
end
