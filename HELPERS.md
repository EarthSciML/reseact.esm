# Helper code behind `run_reseact.jl` and `run_reseact_reactant.jl`

The two root scripts are the runners. Both default to the same simulation — **one
week of CONUS at 13×7×72** (6,552 cells, 85,176 states, 300 s macro steps) — and
differ only in who does the arithmetic: `run_reseact.jl` on the CPU through the
SciML operator-splitting solver, `run_reseact_reactant.jl` through Reactant/XLA.
All the reusable machinery lives in helper files they `include`. This document
(1) summarizes exactly what each configuration depends on and (2) proposes where
each helper should eventually live — preferably in an existing EarthSciML-org
package or extension — so the drivers can shrink to `using`-lines.

Measured on this machine, CONUS 13×7×72:

| | build | compile | solve | notes |
|---|---:|---:|---:|---|
| `run_reseact.jl`, 1 week, `julia -t 8` | 683 s | — | 15,227 s | ~40× realtime; 2,016 macro steps, 4 bisections, continuity rms ~5e-13 all week |
| `run_reseact.jl`, 24 h | — | — | 2,143 s | the reference the traced arm is compared against |
| `run_reseact_reactant.jl`, 24 h | 718 s | 557 s | 5,923 s | 288 macro steps ⇒ **~2.8× slower than native**; the compile is paid once for the whole run |

The traced arm's step cost is strongly **diurnal** — ~9.6 s/step pre-dawn against
~35 s/step at midday, 20.6 s/step averaged over the day — because photochemistry is
what makes the chemistry half stiff. Do not extrapolate it from a short window: a
3-macro-step sanity run sits in the dark *and* carries the first-step warm-up, which
is how an earlier estimate of this ratio came out badly wrong in both directions.

Agreement of the traced arm with the native arm over 24 h (289 aligned records): max
relative difference 6.3e-5 on O3_mean, 1.6e-3 on O3_min, 1.1e-3 on OH_max, 5.2e-12 on
air mass, `cos_sza` exactly 0. Two independent implementations of the same scheme with
different Jacobians, well inside the 5e-2 validation tolerance.

---

## 1. What each runner requires

### Common to both (operator-split machinery)

| Helper | Provides | Notes |
|---|---|---|
| `prototypes/reseact_3d_chem/split_common.jl` | `prepare_split_docs`, `index_promoted_refs_by_loop!`, `build_split_run`, `reseact_forcing`, `native_slice`, `forcing_days_for`, `validate_reseact`, `hydrostatic_dp` | Splits the model with `EarthSciASTSplitter.split_system(·, stencil_following_rule)` and wires GEOS-FP forcing exactly as `EarthSciAST.simulate` does. `reseact_forcing` is model-specific; `index_promoted_refs_by_loop!` is the post-promotion fix-up described below; `native_slice` is the domain seam described below. |
| `prototypes/reseact_3d_chem/block_jac.jl` | `cellmajor_perm`, `cellmajor_rhs`, `block_fd_jac` | Species-major ↔ cell-major permutation (derived purely from state names) and the NS-color block-diagonal finite-difference Jacobian for the pointwise/chemistry part (NS+1 RHS evals per Jacobian instead of N). |
| `prototypes/reseact_3d_chem/blockdiag_local.jl` | `BlockDiag` module → `BlockDiagonal`, `MapBroadcast` | `include`s EarthSciMLBase's `blockdiagonal.jl` + `map_algorithm.jl` **directly from a sibling checkout** to get `BlockDiagonal` without pulling the full EarthSciMLBase (ModelingToolkit/Catalyst) dependency tree. |
| `tools/reactant_handoff/op_split.jl` | `lie_trotter_solve` — drives `OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov((SSPRK43, Rosenbrock23/BD))` as a macro-step loop (positivity clamp + forcing refresh between steps). `include`s `blockdiag_similar.jl`. | The real SciML operator-splitting solver, usable only because of the fix below. |
| `tools/reactant_handoff/blockdiag_similar.jl` | `Base.similar(::BlockDiagonal, ::Type)` + `Base.zero(::BlockDiagonal)` | **The densification fix.** DiffEqBase's `promote_f` runs `similar(jac_prototype, uElType)` at every `init`; `BlockDiagonal` defined only the *nullary* `similar`, so the element-type form fell through to the dense `AbstractArray` fallback and a plain `ODEProblem` (as `LieTrotterGodunov` builds per operator) turned the 3528×(13×13) block Jacobian into a dense 45864² matrix. These two methods preserve the block structure. Type piracy; durable home is EarthSciMLBase's `blockdiagonal.jl`. |
| `prototypes/reseact_3d_chem/hybrid_coefs.json` | 72-level hybrid-sigma vertical coordinate coefficients | Model data, not code. |

**The slice origin has two currencies and nothing in the model ties them.**
`LON0`/`LAT0` are **metaparameters** folded into the index expressions that read
the GEOS-FP arrays (local cell `i` is native `LON0+i`); `Transport3D.lon0_deg` /
`lat0_deg` are **parameters in degrees** that the model's solar chain uses to
place the sun. An `.esm` parameter default cannot be an expression over a
metaparameter, so the file cannot derive one from the other, and a run with the
two disagreeing is *completely silent*: it produces a full, plausible trajectory
with the meteorology of one place and the sunlight of another.

`native_slice(; lon0, lat0, nlon, nlat, nlev)` is the single source of truth —
it returns the metaparameters, the degree parameters, the raw index origin, and
the domain extent, and bounds-checks the halos against the native 72×46 grid.
`build_split_run` takes it as `slice=` and **applies the degree parameters
itself** (they win over `parameters=`), then echoes it back as `run.slice` for
`hydrostatic_dp(...; slice = run.slice)` and any diagnostic that reads the native
arrays. Do not write `29 + j` / `14 + i` at a call site again.

**Multi-day forcing.** `reseact_forcing(dir; ndays)` hands each provider a
`t -> url` resolver over GEOS-FP's one-file-per-day layout instead of a fixed
URL, so a run is no longer capped at the 19.5 h a single day of files could
bracket. Size `ndays` with `forcing_days_for(t0, tf)`, which adds the extra day
the final bracket's *successor* lands in — come up short and the bracket
degenerates to `[last, last]` and the meteorology quietly freezes for the rest of
the run. This needs the EarthSciIO fix that locates a record inside *its own*
file (`_file_record`): the cadence tick is global, the file is local, and tick 9
of a 3-hourly cadence used to ask an 8-record file for record 9.

### `run_reseact.jl` (native in-place) — the above **plus** `op_split.jl`.

Builds two in-place `f!(du,u,p,t)` closures and drives the Lie-Trotter split with
the real operator-splitting solver — `LieTrotterGodunov((SSPRK43 transport,
Rosenbrock23/BlockDiagonal chemistry))` via `lie_trotter_solve_bisect`. This is the
reference the Reactant runner is validated against. (`RESEACT_MACRO_DT`, default
300 s, sets the splitting interval.)

It is `lie_trotter_solve_bisect`, not `lie_trotter_solve`, because the week run
needs it: at the pre-dawn NO minimum the inner Rosenbrock23's TRIAL sub-steps go
negative and convergence collapses, and `clamp_nonneg` is **inert** against that —
it inspects only ACCEPTED states, which never go negative. Retrying the macro step
over two halves clears it in 2 bisections and costs *less* chemistry work than the
failed 300 s attempt. The durable fix is §3 below.

**Forcing-refresh fencepost — read this before touching `lie_trotter_solve`'s stop
grid.** Its `fstops` interval is deliberately **half-open on the right**
(`t0 < x <= tf`). Strict interiority is the obvious-looking choice and is a silent
forcing-freeze: every driver calls `lie_trotter_solve` **one macro step at a time**,
so a cadence boundary landing on a macro-step edge is excluded from the step that
*ends* there and again from the step that *starts* there, and never refreshes.
Every GEOS-FP tstop is a multiple of 1800 (I3 `10800k`, A3 `5400+10800k`, A1
`1800+3600k`), so at the default `macro_dt=300` with a `T0` on the cadence, **every**
boundary was skipped — measured **0** refreshes over a 9 h run, i.e. U/V/PS/T/PBLH
frozen at `T0` for the whole simulation. `macro_dt=250` (which does not divide 1800)
fired normally, which is what kept it invisible.

Refreshing *at* `tf` is correct rather than a fudge: at a boundary both brackets are
valid, because the old bracket's right record and the new bracket's left record are
the same record. It cannot double-fire, since the next call has that instant as its
`t0` and the strict left bound excludes it. The invariant to test is that the firing
set is **independent of `macro_dt`** — `{300, 250, 900, 10800}` must all give the
same 12 times over a 9 h window.

The symptom to recognise, because it does not look like stale forcing: the
continuity drift `|m-dp|/dp` sits at roundoff *within* each 3 h window, then steps
by a constant at exactly the I3 record boundaries and stays flat between them,
perfectly linear in boundaries crossed. That is `m` integrating a constant `dPSdt`
while the diagnostic's weight `w_I3(t) = (t mod 10800)/10800` sawtooths back to 0,
so `dp` drops discontinuously by one window's PS change and `m` does not. **A drift
that steps at forcing boundaries and is flat in between is stale forcing, not a
discretisation error.**

### `run_reseact_reactant.jl` (Reactant/XLA) — the above **plus** two Reactant helpers

It does **not** use `op_split.jl`: Reactant cannot trace through the SciML solver
stack, so the Lie-Trotter step is written out by hand in `rx_traced_integrator.jl`.
The two arms are therefore independent implementations of the same scheme, which is
what makes the agreement between them (24 h at CONUS: max 6.3e-5 relative on
O3_mean, `cos_sza` exactly 0) worth anything.

| Helper | Provides | Status |
|---|---|---|
| `tools/reactant_handoff/rx_native_patch.jl` | (a) native `stablehlo` broadcast lowering — same-shape primitive broadcasts lower straight to `Reactant.Ops.*` with no `<op>_broadcast_scalar` helper per site (kills the 10 000-name cap); (b) `make_tracer` opaque-leaf registration for `EarthSciAST._Node/_AccKernel/_OopAccPlan/_AccScratch` (skips the O(IR-size) capture walk — hours → minutes). | **Runtime monkey-patch.** Needs a durable home. |
| `tools/reactant_handoff/rx_traced_integrator.jl` | The purpose-built traced integrator: ROS23 + SSPRK43 stage algebra, batched pivot-free block Gaussian elimination over cells, masked FD block Jacobian, exact PI controller, and the `@trace while` adaptive loop. This is why one macro step compiles as a single `stablehlo.while` program. | **The substantial deliverable.** Generic (not reseact-specific). Needs a package home. |

`tools/reactant_handoff/` now holds only these plus `op_split.jl` /
`blockdiag_similar.jl`. The probe and experiment scaffolding that established the
Reactant path (`rx_probe.jl`, `rx_count_walk.jl`, `rx_split_probe.jl`,
`rx_merge_kernels.jl`, the lowering/cap experiments, and the `run_traced_*` /
`run_split_reactant.jl` predecessors of the full-loop runner) has been **deleted** —
their conclusions are in that directory's README and in the runners' headers.
`rx_merge_lib.jl` went with them: the in-package merge (below) superseded it, and its
driver-side reconstruction predated materialized array-observed levels, so its own
bit-identity gate had been failing and falling back to the stock RHS on every run.

---

## 2. Where the helpers should live

Ordered roughly by payoff / readiness.

### Done ✅ — kernel-class merge, and the three emitter fixes that made CONUS compile
The lane-batching merge that makes the transport half tractable under Reactant is
**in EarthSciAST `main`** (`src/tree_walk/oop_merge.jl`, run from `build.jl` before
the xcse gate, for both `:oop` and `:inplace`). `rx_merge_lib.jl` and the
driver-side merge block are deleted.

Three further `:oop` emitter changes landed upstream, and together they are what
turned a CONUS `@compile` that never finished in 3h45m into one that takes ~550 s:

* **lane-batch the per-cell scalar surface** — groups congruent per-cell/per-column
  scalar entries by structural signature and evaluates each group once over its lane
  axis. All 1,274 ReSEACT per-column fills collapse to 3 lane-groups.
* **level-major prefix scan** — scans a whole level at once, which takes the
  `dynamic_slice` count off the grid entirely (2,016 at both 7×7 and 13×7, Δ0).
* **read interning** — emits one read per `(SSA value, window)`, so XLA stops
  rediscovering duplicates. Slices 8,127 → 3,199 (the exact distinct-pair floor),
  duplicates → 0.

That last one is why the host-unrolled block Jacobian is now the unconditional
choice in `run_reseact_reactant.jl`: XLA's `cse_slice` pattern is **pairwise** over
slice ops, so before interning the unrolled form's 16 traced RHS copies cost ~12×
the compile of the traced loop's 4. With the duplicates gone the quadratic has
nothing to chew on, and the unrolled form compiles 2% *faster* while solving 1.67×
faster (rejected chem steps 23 → 5).

### `EarthSciASTReactantExt` (already exists) — the two `rx_native_patch.jl` pieces
`ext/EarthSciASTReactantExt.jl` already owns the Reactant seam and knows the
`EarthSciAST` node types.
- **`make_tracer` opaque leaves** are unambiguously ext material — they reference
  `EA._Node`/`_AccKernel`/`_OopAccPlan`/`_AccScratch` directly. Move verbatim.
- **Native broadcast lowering** is a *generic* Reactant improvement (nothing
  EarthSciAST-specific), so the true home is upstream Reactant (the `elem_apply`
  fast paths, cf. issue #1616). Until that lands, carry it in the ext as a
  documented stopgap, guarded so it no-ops on a Reactant version that already
  lowers natively.

### New lightweight package — the traced integrator
`rx_traced_integrator.jl` is a self-contained traced adaptive ODE stepper with no
reseact specifics. Best home is a small EarthSciML-org package, e.g.
**`EarthSciMLTracedIntegrators.jl`** (or a `solve`-side companion inside
`EarthSciASTReactantExt`). It pairs naturally with the traced `:oop` RHS the ext
already produces: the ext gives you a traceable RHS, this gives you a traceable
adaptive *solve* of it. Its block-diagonal Jacobian + cell layout overlap with
`block_jac.jl` (below) — share one implementation.

The file now also carries the **one-step discrete adjoint** (`ros23_step_vjp` /
`ssprk43_step_vjp`) and an **exact coloured-AD block Jacobian** (`ad_block_jac`,
`jac=:ad`). Those travel with the stepper, not separately: the adjoint's only
real constraint is that the step body it differentiates contains no
`stablehlo.while`, which is a property of the stage code sitting next to it.
Validated by `tools/rx_adjoint_check.jl`.

### Long-run driving (now IN the root runners) and `tools/scaling_study.jl`

The long-run drivers used to be separate files (`tools/diurnal_run.jl` and its
traced twin) sitting alongside two shorter-window root runners. They have been
**merged into the root runners**, which is where the long-run behaviour belongs:
`run_reseact.jl` and `run_reseact_reactant.jl` now default to the one-week CONUS
simulation and carry the macro-step loop, the streaming sink, and (native arm) the
bisect-on-failure retry. The separate drivers are deleted.

Things that were learned the hard way and are now load-bearing in those files:

* Full gridded output goes through EarthSciAST's streaming-sink protocol
  (`derive_output_meta` → `build_zarr_sink` → `sink_open!/write!/flush!/close!`),
  the same surface `simulate` drives via `sinks=`. Two traps: the native solve is
  CELL-major while the sink wants `var_map` (species-major) order, and EarthSciIO
  keeps its compressors as **weakdeps** — the default `:diagnostic` profile is
  Blosc-zstd, so `using Blosc` is required or `sink_flush!` throws. The sink is
  state-only in its current wave, so j-rates/cos_sza cannot be written through it.
* Flush periodically, not at the end. Over a multi-day run that is the difference
  between a crash at hour 40 costing an hour and costing the whole run.
* A macro step that did not succeed must not be papered over by advancing `t`: the
  state comes back unchanged and the run marches on producing a frozen trajectory
  that still *looks* like a completed simulation.
* Threading is worth ~10× on the native arm (a CONUS week went from ~1:1 realtime
  to ~40×), and it needs both `julia -t N` **and** Polyester loadable so
  `EarthSciASTPolyesterExt` activates. `run_reseact.jl` reports which of the two it
  actually got, because a 10×-slow run otherwise looks completely normal.

* **`scaling_study.jl`** runs grid/window ladders, timing **build**, **first macro
  step** and **subsequent steps** separately. That separation is the whole point:
  the first step carries the one-time codegen compile and can be ~800x the cost of
  the identical second step, so an undifferentiated "wall vs cells" curve measures
  the Julia compiler rather than the model. Builds are cached by grid (a `warm`
  column flags reuse). Minimum extent per axis is **6**, not 1 — the PPM rules
  carve three boundary regions per end plus an interior `[4, N-3]`, so `NLEV=4`
  fails to load with `makearray_region_inverted`.

**Do not edit `reseact.esm` or `reseact_forcing` while either is running**: a
long-lived process holds the old `reseact_forcing` in memory but re-reads the
model from disk per grid, so a newly-required forcing key fails every subsequent
build with `E_TREEWALK_UNBOUND_VARIABLE`.

### Continuity / CWC gates (`tools/continuity_residual.jl`, `tools/cwc_gate.jl`)
The two acceptance tests for the pressure fixer. Both are cheap to *read* and
expensive to *run* (~4 min build, then a first-macro-step codegen compile), so
they are drivers you launch and come back to, not inner-loop checks.

* **`continuity_residual.jl`** evaluates the transport RHS **once** and compares
  `dm/dt` against `dB[k]*dPSdt` cell by cell. No solver, no stepping, so whatever
  it reports is pure discretisation. Its most useful output is not the magnitude
  but the *shape*: the residual reported **relative to `dp`, per level**. A
  k-independent relative residual (constant down each column to ~1e-11) means the
  dp-weighted column correction is surviving uncancelled — a one-line bug in which
  divergence the `m` equation reads. Anything with vertical structure, or an
  interior/wall split, means the stencils. That discriminator is what turned a
  vague "3e-6 1/s, worst cell around k≈50" into an exact diagnosis.
* **`cwc_gate.jl`** synthesises a gate copy of the model — it appends the gate
  states to `reseact.esm` and writes `_cwc_gate.esm` **as a sibling**, because the
  relative `$ref`s only resolve from that directory — then deletes it on exit
  (`RESEACT_KEEP_GATE=1` to keep). It runs four instruments, and the split between
  them is the point:
  - `qtest` — the mixing-ratio form the SuperFast species actually use. The gate
    that matters operationally.
  - `qg = mqg/mg` — the mass form, with the gate's own air/tracer mass pair
    integrated from the RAW divergences, so it tests the discretisation rather
    than the fixer.
  - `dev_x`/`dev_y`/`dev_z` — the tracer-vs-continuity divergence difference **one
    axis at a time**. These are what name the failing rule pair instead of just
    reporting that something is off, and they are what the verdict gates on.
  - `dev` — the same difference for all three axes summed. **Reported but not
    gating.** If the per-axis states are all exactly zero then the two sums have
    bitwise-identical addends, so any residue is the association order of the two
    sums in the emitted code (IEEE `+` is commutative but not associative), not
    the transport operator. It measured ~9e-15 Pa against `m ~ 1500 Pa` while
    `mg === mqg` stayed bitwise equal throughout — a ~1 ulp RHS difference is
    ~1e5x below `ulp(1500)`, which is exactly why a state starting at `0.0` can
    see it and the air mass cannot.

  `clamp_nonneg` is deliberately **off** here: it is a production safety net for
  the stiff chemistry, but it is a nonlinear edit of the state vector and would
  silently repair the identity under test.

### `EarthSciAST` — fold closed functions on literal arguments
`datetime.year … datetime.day_of_year` are evaluated as **opaque host calls**
(`Dates.unix2datetime`), so they cannot appear anywhere in a traced RHS. That is
not just a "don't make it depend on `t`" rule: the tree-walk evaluator lowers
every leaf as `convert(T, node.literal)`, and under Reactant `T` is a
`TracedRNumber`, so **even a constant timestamp fails to trace**
(`MethodError: no method matching Float64(::Reactant.TracedRNumber{Float64})`).
The oop lane arm says as much in a comment — *"a trace fails loudly inside the
opaque callee"*.

The fix is to fold a `fn` node whose arguments are all literals to a literal at
build time. Closed functions are pure by spec, so this is unconditionally safe,
and it also removes the per-cell opaque call from the native RHS. It has to fire
before CSE assigns the argument an observed slot, so it likely wants to sit with
a constant-scalar-observed inline pass rather than in `_compile_fn_node` alone.

Until that exists, a photolysis component that decomposes a UTC timestamp
internally (`components/gaschem/fastjx/fastjx.esm`) **cannot be mounted in a
Reactant-traced model at all**. This repo sidesteps it by mounting the sibling
`fastjx_interp_troposphere.esm`, which takes `cos_sza` as an input, and computing
the solar geometry on the Transport3D side as pure arithmetic — which has the
happy side effect that `cos_sza` can then depend on `t`, so the sun tracks the
solve instead of being frozen. `interp.linear` / `interp.bilinear` are NOT
affected: they already have branch-free `ifelse`-select lane forms written
specifically to trace.

### `EarthSciAST` — finish `promote_downstream_shapes` (the FastJX blocker)
`promote_downstream_shapes` lifts a scalar 0-D physics chain fed by array sources
into the grid shape — exactly what mounts the Fast-JX photolysis component on the
slice (its `lat`/`lon`/`T`/`P` arrive `[lon,lat,lev]`, so every j-rate becomes
`[lon,lat,lev]`). It rewrites each promoted variable's OWN equation into an
`arrayop` with indexed leaves, but leaves its **consumers** alone; it only hands
back the promoted-name set for the caller to finish with `index_promoted_refs!`.

For reseact the consumers are the pointwise-lifted species ODEs, which were
array-ified earlier inside `flatten`'s pointwise lift — before the promotion ran,
so `FastJX.j_NO2` was still scalar there and survives as a **bare** name inside a
per-cell `aggregate` body. The build then dies with
`E_TREEWALK_UNBOUND_VARIABLE: FastJX.j_NO2`.

`index_promoted_refs_by_loop!` (in `split_common.jl`) closes it, driving
`index_promoted_refs!` per equation with that equation's own `output_idx` —
necessary because the two consumer families use different loop names (the lift's
`["i","j","k"]` vs a promoted definition's `["_p0","_p1","_p2"]`) and
`index_promoted_refs!` takes only one `spatial_loops` list.

**This belongs upstream**, in `promote_downstream_shapes` itself: it already knows
the promoted set and can read each consumer's loop names off its own `output_idx`.
Note `EarthSciAST.simulate` has the identical hole today (`simulate.jl` calls
`promote_downstream_shapes` and never calls `index_promoted_refs!`), so **any**
model that mounts a 0-D chain onto a pointwise-lifted mechanism hits this through
the canonical runner too — this is not a split-specific quirk.

### `EarthSciASTSplitter.jl` — the split build + block-diagonal driver machinery
The splitter already owns `split_system`; the glue that turns split parts into a
runnable, forcing-wired, block-diagonally-solvable problem belongs beside it:
- **`build_split_run` / `prepare_split_docs`** (from `split_common.jl`) — a public
  "build a forcing-wired operator-split run" entry point mirroring `simulate`.
- **`cellmajor_perm` / `cellmajor_rhs` / `block_fd_jac`** (from `block_jac.jl`) —
  generic for *any* pointwise (cell-local) split part, not just reseact chemistry.
Leaving only `reseact_forcing` (model data) in this repo.

### `EarthSciMLBase` — unbundle `BlockDiagonal` and fold in the `similar` fix
`blockdiag_local.jl` exists solely to borrow `BlockDiagonal` + `map_algorithm`
without dragging in ModelingToolkit/Catalyst. The clean fix is on the library side:
factor those two files into a **lightweight subpackage** (e.g. `BlockDiagonalMaps.jl`
or an `EarthSciMLBaseCore`) that consumers can depend on directly. Then
`blockdiag_local.jl` disappears and both runners just `using` it.

While doing so, move the two methods in **`tools/reactant_handoff/blockdiag_similar.jl`**
(`similar(::BlockDiagonal, ::Type)`, `zero(::BlockDiagonal)`) into
`blockdiagonal.jl` next to the existing nullary `similar`/`copy`/`inv` loop. They
are the reason a plain `ODEProblem` — and therefore
`OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov` — no longer densifies a
block-diagonal `jac_prototype` at `init` (DiffEqBase's `promote_f` calls
`similar(jac_prototype, uElType)`, which without them returned a dense `Matrix`).
Once upstream, delete `blockdiag_similar.jl` and the `include` of it in `op_split.jl`.

### Stays in this repo (model-specific, not for migration)
`reseact_forcing` (GEOS-FP provider chain + hybrid coefs) and `hybrid_coefs.json`.

---

### Net effect if all of the above lands
Both root scripts collapse to `using EarthSciASTSplitter, EarthSciMLBase[Core],
EarthSciMLTracedIntegrators` + the ~10-line `reseact_forcing`, with **no** `include`
of prototype files and **no** runtime monkey-patch.

---

## 3. Upstream: OrdinaryDiffEqOperatorSplitting drops every callback

**Status: NOT DONE. Two defects, both in v0.3.2, both one-liners at the call site.**
Until they land, positivity is enforced by `lie_trotter_solve_bisect` (op_split.jl),
which retries a failed macro step over two halves. That is the right MECHANISM
(shorten the step; do not touch values) at the wrong GRANULARITY (macro, not
sub-step), and it exists only because the callback route is unavailable.

### The defects

1. **The leaf child never receives the callback.** `src/integrator.jl` builds each
   inner integrator with `callback` already bound in its parameter list, then calls
   `SciMLBase.__init(prob2, alg; dt, tstops, saveat = (), d_discontinuities,
   save_everystep = false, advance_to_tstop = false, adaptive, controller, verbose)`
   — no `callback`. The argument is accepted and dropped. Fix: add `callback,` to
   that kwarg list.
2. **The outer integrator never APPLIES its callback.** It stores a `CallbackSet`,
   `DiffEqBase.initialize!`s it and `finalize!`s it, but `apply_discrete_callback!` /
   `handle_callbacks!` appear NOWHERE in the package, so nothing fires during the
   step loop. Fix is larger: apply callbacks in the outer loop.

Consequence today: `callback = DiffEqCallbacks.PositiveDomain(copy(u))` is **silently
ignored** at both levels. Worse than an error — it looks like it worked. This is why
`op_split.jl` says the package "does NOT apply discrete callbacks", and why the split
runners degraded to `clamp_nonneg` and then to macro-step bisection.

### Why it matters here (measured, not assumed)

The CONUS week run dies at the pre-dawn NO minimum (11.33 h). `clamp_nonneg` is
**inert** against it: disabling it is bit-identical, and both arms end `nneg=0`. The
ACCEPTED states never go negative — the inner Rosenbrock23's TRIAL sub-steps do, the
RHS is evaluated at negative concentrations, and convergence collapses. A
macro-boundary clamp inspects only accepted states, so it cannot see this by
construction. `PositiveDomain` can, because it rejects the step and retries smaller.

### How to validate the fix before proposing it

Do NOT edit `~/.julia/packages/.../src/integrator.jl` in place — it is shared depot
state, invisible to git, and clobbered by the next `Pkg` operation. Instead copy the
package out, `Pkg.develop` the copy into `run-model-jl`, add `callback,` to the leaf
`__init`, and run the 11.33 h reproduction (see
[[week-run-failure-11h-jacobian-refuted]] in session memory, or the recipe below).
`run-model-jl/Manifest.toml` is tracked, so `git checkout` reverts the environment.

Reproduction, ~12 min instead of the 11 h the failure takes to reach live: restore a
state from a run's zarr with a **FRESH cache** (the default cache serves stale blobs
for a store rewritten in place), map `(i,j,k,rec)` to cell-major `u[(c-1)*NS+s]` via
`cellmajor_perm`, refresh forcing at the last cadence boundary `<= t` (NOT at `t`),
use `reltols=(1e-4,1e-4)`, `abstols=(1e-6,1e-9)`, and replay macro steps.

If inner `PositiveDomain` proves materially better than macro-level bisection, drop
`lie_trotter_solve_bisect` and pass the callback instead.

---

## 4. Differentiable simulation: what is already there, and what is missing

The point of the Reactant path is not speed — it is **∂(simulation result)/∂(parameter
vector)**, the thing DifferentialEquations.jl gives you by letting `p` be a solve-time
argument. This section records what was measured, not what was assumed. The probe is
`diffprobe.jl` (see the session bench dir); every claim below is a run, on a small
reaction–diffusion model built through the same emitter the real model uses.

### Already works, measured

| | result |
|---|---|
| host ForwardDiff `d/dp` through the `:oop` RHS | works; nonzero, finite |
| host ForwardDiff `d/dp` through the in-place `f!` | agrees with `:oop` to rtol 1e-12 |
| `p` under Reactant | a **real XLA input** — same compiled program, new parameter values, answers follow and match the host |
| Reactant + Enzyme reverse `d/du` of the traced RHS | matches host ForwardDiff to rtol 1e-8 |
| Reactant + Enzyme reverse `d/dp` of the traced RHS | matches host ForwardDiff to rtol 1e-8 |

That last row is the one worth pausing on: **reverse-mode parameter gradients through
the traced RHS already work today.** And because `p` is a genuine XLA input rather
than a trace-time constant, a gradient does not force a recompile per parameter value
— which is what makes an optimizer loop affordable at all given a ~550 s compile.

Two pieces of existing design are load-bearing here and should not be "simplified"
away. `_rhs_value_type` derives the RHS value type from `u`, `p` AND `t` together, so
`p` going `Dual` while `u` stays `Float64` works (differentiating w.r.t. parameters is
exactly that case). And `const_tier.jl` deliberately does **not** constant-fold
parameter-only subexpressions at build time, precisely because freezing them would
return a zero derivative for every parameter sensitivity — a wrong Jacobian that still
looks plausible.

### Missing, in dependency order

1. **There is no parameter-VECTOR ABI.** `p` must be a `NamedTuple`;
   `_rhs_value_type(u, ::Vector, t)` is a `MethodError`. A vector API works only by
   rebuilding the NamedTuple per call, which needs scalar indexing of a traced array
   — allowed only under `@allowscalar` (verified: it does then produce the right
   gradient). ReSEACT has ~56 parameters, so the NamedTuple is not a scaling problem
   here; the problem is that an optimizer wants `∇_p L` as a dense vector in a stable
   order. **Fix:** a `_rhs_value_type(u, p::AbstractVector, t)` method, an `_NK_PARAM`
   variant carrying an index instead of a symbol, and a `param_map::Dict{String,Int}`
   returned from `build_evaluator` beside `var_map`. Mechanical, low risk. (The
   ordering already exists implicitly — `param_names` is sorted, so `keys(p)` is a
   stable order — it is just not exposed as a map.)

2. **`simulate()` REFUSES runtime parameters, by design.** It throws:
   "parameter overrides are baked into the evaluator at prepare() time (they feed
   build-time constant folding: setup geometry, value-invention extents, binning
   coordinates, ic() folds)." This is the deepest item on the list, and it is a design
   decision rather than plumbing: parameters have to be **partitioned** into
   STRUCTURAL (fold at build; not differentiable; changing one is a rebuild) and
   NUMERIC (runtime; differentiable; a solve-time argument). The good news is the
   partition already half-exists — the tree walk treats `_NK_PARAM` as a runtime read
   and refuses to fold it — so the work is to make the split explicit at the
   `prepare`/`simulate` seam and let the numeric half arrive as a vector at solve time.

3. **REVERSE MODE NEEDS A *STATICALLY KNOWN* TRIP COUNT — the `while` region itself is
   fine.** *(Corrected 2026-08-12. This item previously read "REVERSE MODE CANNOT CROSS A
   `stablehlo.while`". That was wrong; the retraction is at the end of this item.)*

   The rows above all differentiate the RHS, which is straight-line traced code. For the
   time loop, what reverse mode actually requires is that the **trip count be a
   compile-time constant**, because Enzyme-MLIR's `AutoDiffWhileRev` tapes the trajectory
   into a dense `[N, state…]` tensor: if `N` is not static the tape is `tensor<?x…>`,
   which XLA cannot translate. Measured on Reactant 0.2.274 — the same version as before:

   | loop shape | reverse mode |
   |---|---|
   | `@trace for _ in 1:20`, literal bound | ✅ exact (rel 0.0), and `stablehlo.while` is **still present in the differentiated module** — not unrolled away |
   | trip count passed as a `ConcreteRNumber` argument | ❌ `'stablehlo.dynamic_pad' op can't be translated to XLA HLO` (Int counter) / `WhileOp does not have induction variable for cache removal` (Float64 counter) |
   | data-dependent condition (`while s < thresh`) | ❌ `WhileOp does not have known iteration count for cache removal` — [Enzyme-JAX #2565](https://github.com/EnzymeAD/Enzyme-JAX/issues/2565), open |

   Checkpointing (`checkpointing=Periodic(n)`/`Binomial(n)`, `mincut=true`) does **not**
   rescue the data-dependent case. Two hazards found while establishing this: `Binomial(n)`
   on a traced-bound loop **compiles and returns a silently wrong gradient** (exact at
   n=2,4; 1.0e-3 rel at n=5; 2.8e-3 at n=10 — cf. Reactant.jl #1895), and `Binomial`
   segfaults in `BinomialProgressConstProp` on a static-bound loop. `stablehlo.case` has no
   reverse rule at all — keep it out of differentiated code.

   **Retraction, and the lesson.** The old claim rested on probe `whilegrad.jl` K1, labelled
   "FIXED trip count". It was not fixed: line 55 passes
   `nr = RX.ConcreteRNumber(Float64(NSTEP))` as a runtime **argument** into
   `RX.@trace while i < nlim`. Fixed in the Julia source, statically *unknown* in MLIR — so
   K1 was a second dynamic-trip-count measurement, not the control it was read as. That one
   mislabelled row carried the inference "it fails for fixed counts too, therefore it is the
   region and not adaptivity", and that inference set the entire host-loop architecture.
   **A probe's label is not its semantics: check what the compiler sees, not what the
   source says.**

   Reverse is still the only affordable mode at this parameter count — forward costs
   ∝ n_params, ~88 h for a 24 h run over 56 parameters against ~4–5 h for an adjoint — so
   Phase 3b's per-step VJP remains correct and is unaffected. What changes is that the host
   loop may not be *necessary*: an adaptive loop needs a *bound*, not a `while` region.
   Rewritten as a `@trace for` over a compile-time attempt cap with termination as an
   `ifelse` on the carry, a full accept/reject PI-controller stepper differentiates exactly —
   and `adaptive_solve`'s body is **already written that way**, every update already an
   `ifelse(accept, …)`; only the loop condition has to move. Priced: at cap == actual trips
   the masked form is 0.83× the while's wall time, overshoot linear (2× cap → 1.49×, 8× →
   5.39×); a two-phase scheme — cheap forward `while` to learn the count, round up to a
   power of two, differentiate the masked loop at that bucket — holds overshoot to 1.28×.
   Untried at ReSEACT state sizes; the tape is dense `[cap, state]`, so large state × large
   cap is the thing to check before committing. Probes: `EarthSciAST/bench/whilegrad{3,4}.jl`,
   `maskedloop.jl`, `maskedcost.jl`. **See DIFFERENTIABILITY_PLAN.md** for the phased plan
   and the capability matrix.

4. **The traced arm's block Jacobian is FINITE DIFFERENCE.** `ros23_step` builds it by
   FD. Differentiating through an adaptive implicit step whose linearization is
   FD-perturbed is valid discretize-then-optimize but contaminates the gradient, and
   `clamp_nonneg` plus the step-size controller make the step piecewise on top of that.
   The native arm already defaults to an exact `block_ad_jac` (ForwardDiff); the traced
   arm has no equivalent. Worth closing before trusting a gradient quantitatively.

5. **Enzyme on the CPU needs `Enzyme.API.strictAliasing!(false)`**, because the walk
   **loads** a `payload::Any` field, which is heterogeneous by design. With the flag,
   reverse mode matches ForwardDiff to ~1e-16. *(Corrected 2026-08-12 — the conclusion
   holds, three details did not.)*

   * **The struct is `_Node`, not `_VecNode`.** `_VecNode` no longer exists anywhere in
     `EarthSciAST/src` except in comments — zero non-comment uses; the only definition is
     `struct _Node` at `tree_walk/compile.jl:69`. Reproduced on Enzyme 0.13.199 in under a
     minute: `IllegalTypeAnalysisException` in `_oop_eval` from a `getproperty` load of
     `_Node.payload::Any`.
   * **It is not an `:oop` problem.** The in-place `f!` fails identically at `_eval_node`.
   * **The tree need not contain the offending node kind.** A 0-D model with no loop-var
     node still fails on the loop-var arm, because Enzyme analyses the whole statically
     reachable method, not the values actually walked.

   **The flag is a wart, not a hazard** — worth knowing before anyone "hardens" it away.
   `EnzymeStrictAliasing` is read in exactly one upstream file, `TypeAnalysis.cpp`, and at
   5 of 7 sites turning it off makes analysis *decline to learn* facts (it skips upward
   propagation through phis/selects). The flag that fabricates unproven types and can
   yield wrong derivatives is `looseTypeAnalysis!` — a **different** global, read by
   different code (`AdjointGenerator.h` et al.). No issue-tracker report of wrong
   gradients from strict-aliasing-off; Rust's Enzyme frontend ships with it off by
   default. The real cost is blast radius: it is process-global with no scoped
   alternative.

   **"Payload-free lowering fixes it" is over-claimed.** A micro-harness reproducing the
   exact production failure shows it is the **load**, not the field: a struct still
   declaring `payload::Any` but never loading it differentiates fine, and `@noinline` /
   `EnzymeRules.inactive` barriers work too. A small closed `Union` does **not** help.
   Walking the real code with barriers cleared two blockers before hitting one no barrier
   can fix — `_oop_fn`'s boxed `Vector{Any}` of **active** values handed to a
   `String`-dispatched registry. And the convergence claim (`codegen_kernel.jl` already
   lowers array kernels to Julia source at build time) is real but half-built: with that
   tier on, Enzyme fails with an internal `UndefVarError: codegen_ft`, so the lowering is
   not by itself a route to Enzyme support.

6. **Discrete closed functions contribute no derivative, by contract** —
   `interp.searchsorted` and the calendar `datetime.*` accept a `Dual` and return zero
   partials. Correct, but it means a parameter reaching the answer *only* through one
   of them is silently zero-gradient rather than an error.

### The shortest path to a first real result — TAKEN: `tools/sensitivity_forward.jl`

FORWARD mode already crosses the while regions exactly, so a sensitivity w.r.t. a
HANDFUL of parameters needed nothing from this list. That driver now computes
d(domain-mean surface O₃)/d(`NEIRegrid.scale`, `Transport3D.tau_pblmix`,
`NEIRegrid.g0`) through the compiled macro step, chained across the window, and checks
itself against a central-finite-difference step-size sweep: agreement 5.5e-9 / 1.6e-7 /
7.5e-9 relative at CONUS 13×7×72 over three 300 s macro steps (build 770 s, JVP
`@compile` 770 s, 124 s per window pass). It is the reference an adjoint gets checked
against — and it should be checked against the AD number, not against fd, because the
fd floor is domain-size dependent (~5e-7 at 6×6×8, where a handful of kinked states
dominates a 36-cell mean). **There is no split fraction to differentiate** — `NEIRegrid.F_NO`
… `F_FORM` are the NEI2016 source-grid emission FIELDS, const arrays folded at build
time, not scalars (item 6's silent-zero hazard, caught by name rather than by a zero).
The full 56-parameter gradient still needs (1) then (3) then a time-loop driver. See
**DIFFERENTIABILITY_PLAN.md**.
