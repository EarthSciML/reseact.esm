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
| `prototypes/reseact_3d_chem/split_common.jl` | `prepare_split_docs`, `build_split_run`, `reseact_forcing` | Splits the model with `EarthSciASTSplitter.split_system(·, stencil_vs_pointwise)` and wires GEOS-FP forcing exactly as `EarthSciAST.simulate` does. `reseact_forcing` is model-specific. |
| `prototypes/reseact_3d_chem/block_jac.jl` | `cellmajor_perm`, `cellmajor_rhs`, `block_fd_jac` | Species-major ↔ cell-major permutation (derived purely from state names) and the NS-color block-diagonal finite-difference Jacobian for the pointwise/chemistry part (NS+1 RHS evals per Jacobian instead of N). |
| `prototypes/reseact_3d_chem/blockdiag_local.jl` | `BlockDiag` module → `BlockDiagonal`, `MapBroadcast` | `include`s EarthSciMLBase's `blockdiagonal.jl` + `map_algorithm.jl` **directly from a sibling checkout** to get `BlockDiagonal` without pulling the full EarthSciMLBase (ModelingToolkit/Catalyst) dependency tree. |
| `tools/reactant_handoff/op_split.jl` | `lie_trotter_solve` — drives `OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov((SSPRK43, Rosenbrock23/BD))` as a macro-step loop (positivity clamp + forcing refresh between steps). `include`s `blockdiag_similar.jl`. | The real SciML operator-splitting solver, usable only because of the fix below. |
| `tools/reactant_handoff/blockdiag_similar.jl` | `Base.similar(::BlockDiagonal, ::Type)` + `Base.zero(::BlockDiagonal)` | **The densification fix.** DiffEqBase's `promote_f` runs `similar(jac_prototype, uElType)` at every `init`; `BlockDiagonal` defined only the *nullary* `similar`, so the element-type form fell through to the dense `AbstractArray` fallback and a plain `ODEProblem` (as `LieTrotterGodunov` builds per operator) turned the 3528×(13×13) block Jacobian into a dense 45864² matrix. These two methods preserve the block structure. Type piracy; durable home is EarthSciMLBase's `blockdiagonal.jl`. |
| `prototypes/reseact_3d_chem/hybrid_coefs.json` | 72-level hybrid-sigma vertical coordinate coefficients | Model data, not code. |

### `run_reseact.jl` (native in-place) — the above **plus** `op_split.jl`.

Builds two in-place `f!(du,u,p,t)` closures and drives the Lie-Trotter split with
the real operator-splitting solver — `LieTrotterGodunov((SSPRK43 transport,
Rosenbrock23/BlockDiagonal chemistry))` via `lie_trotter_solve`. This is the
reference the Reactant runner is validated against. (`RESEACT_MACRO_DT`, default
300 s, sets the splitting interval.)

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
