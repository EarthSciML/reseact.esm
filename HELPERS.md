# Helper code behind `run_reseact.jl` and `run_reseact_reactant.jl`

The two root scripts are thin drivers. All the reusable machinery lives in helper
files they `include`. This document (1) summarizes exactly what each configuration
depends on and (2) proposes where each helper should eventually live — preferably
in an existing EarthSciML-org package or extension — so the drivers can shrink to
`using`-lines.

---

## 1. What each runner requires

### Common to both (operator-split machinery)

| Helper | Provides | Notes |
|---|---|---|
| `prototypes/reseact_3d_chem/split_common.jl` | `prepare_split_docs`, `index_promoted_refs_by_loop!`, `build_split_run`, `reseact_forcing`, `native_slice`, `forcing_days_for` | Splits the model with `EarthSciASTSplitter.split_system(·, stencil_following_rule)` and wires GEOS-FP forcing exactly as `EarthSciAST.simulate` does. `reseact_forcing` is model-specific; `index_promoted_refs_by_loop!` is the post-promotion fix-up described below; `native_slice` is the domain seam described below. |
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
Rosenbrock23/BlockDiagonal chemistry))` via `lie_trotter_solve`. This is the
reference the Reactant runner is validated against. (`RESEACT_MACRO_DT`, default
300 s, sets the splitting interval.)

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

### `run_reseact_reactant.jl` (Reactant/XLA) — the above **plus** three Reactant helpers

| Helper | Provides | Status |
|---|---|---|
| `tools/reactant_handoff/rx_native_patch.jl` | (a) native `stablehlo` broadcast lowering — same-shape primitive broadcasts lower straight to `Reactant.Ops.*` with no `<op>_broadcast_scalar` helper per site (kills the 10 000-name cap); (b) `make_tracer` opaque-leaf registration for `EarthSciAST._Node/_AccKernel/_OopAccPlan/_AccScratch` (skips the O(IR-size) capture walk — hours → minutes). | **Runtime monkey-patch.** Needs a durable home. |
| `tools/reactant_handoff/rx_merge_lib.jl` | Driver-side kernel-class merge (`RxOopMerge.merge_oop_rhs`). | **Superseded** — the merge now runs inside EarthSciAST `build.jl`, so `build_evaluator(form=:oop)` already returns merged kernels. Kept only as a bit-identity **gate** (346→346 no-op) and as a fallback for a pre-merge EarthSciAST. Removable once the pinned EarthSciAST is guaranteed ≥ the merge commit. |
| `tools/reactant_handoff/rx_traced_integrator.jl` | The purpose-built traced integrator: ROS23 + SSPRK43 stage algebra, batched pivot-free block Gaussian elimination over cells, masked FD block Jacobian, exact PI controller, and the `@trace while` adaptive loop. This is why the whole window compiles as one `stablehlo.while` program. | **The substantial deliverable.** Generic (not reseact-specific). Needs a package home. |

Everything else in `tools/reactant_handoff/` is experiment/probe scaffolding
(`rx_count_walk.jl`, `rx_split_probe.jl`, `rx_iip_merge_probe.jl`, `rx_native_bench.jl`,
`rx_native_lowering_experiment.jl`, `rx_raise_cap_experiment.jl`, `rx_merge_trace_inc.jl`,
`rx_merge_kernels.jl` (the annotated prototype `rx_merge_lib.jl` was distilled from),
`run_split_reactant.jl` (forward-pass-per-eval predecessor of the full-loop runner),
and `run_traced_chem.jl` (chemistry-only validation)). None are needed to run the model
— they can be archived or deleted.

---

## 2. Where the helpers should live

Ordered roughly by payoff / readiness.

### Already migrated ✅ — kernel-class merge
The lane-batching merge that makes the transport half tractable under Reactant is
**in EarthSciAST `main`** now (`src/tree_walk/oop_merge.jl`, run from `build.jl`
before the xcse gate, for both `:oop` and `:inplace`). `rx_merge_lib.jl` is therefore
already redundant. **Action:** once the repo pins EarthSciAST ≥ that commit, delete
`rx_merge_lib.jl` and drop the driver-side merge block from `run_reseact_reactant.jl`.

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
adaptive *solve* of it. Its block-diagonal FD Jacobian + cell layout overlap with
`block_jac.jl` (below) — share one implementation.

### Long-run drivers (`tools/diurnal_run.jl`, `tools/scaling_study.jl`)
Two drivers added alongside the two root runners. Both reuse `prepare_split_docs`
/ `build_split_run` / `lie_trotter_solve` — they are drivers, not second solvers.

* **`diurnal_run.jl`** steps the Lie-Trotter split by macro dt and records a time
  series. Full gridded output goes through EarthSciAST's streaming-sink protocol
  (`derive_output_meta` → `build_zarr_sink` → `sink_open!/write!/flush!/close!`),
  the same surface `simulate` drives via `sinks=`. Two traps: the solve is
  CELL-major while the sink wants `var_map` (species-major) order, and EarthSciIO
  keeps its compressors as **weakdeps** — the default `:diagnostic` profile is
  Blosc-zstd, so `using Blosc` is required or `sink_flush!` throws. The sink is
  state-only in its current wave, so j-rates/cos_sza cannot be written through it.
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
drivers degraded to `clamp_nonneg`. Note `tools/run_scale.jl` and `tools/run_sweep.jl`,
which do NOT use the splitter, still pass the real `PositiveDomain` and are unaffected.

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
