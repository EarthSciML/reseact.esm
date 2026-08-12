# Reactant/XLA support code for `run_reseact_reactant.jl`

This directory started as a **handoff**: an investigation into whether EarthSciAST's
Reactant/XLA path could compile and run the full ReSEACT model at all. That question
is closed — it can, at CONUS 13×7×72 (85,176 states), and `../../run_reseact_reactant.jl`
runs a week-long simulation through it. What is left here is the four files that
runner and `../../run_reseact.jl` actually depend on. The probe and experiment
scaffolding that answered the original question has been deleted; the answers it
produced are recorded in `../../HELPERS.md` and in the runners' own headers.

## What the investigation concluded

1. **Tracing needs the out-of-place RHS.** An in-place `f!` captures a host scratch
   buffer per `_VecNode`, so it is *fundamentally* untraceable — the extension's
   docstring says so. Both runners build `form = :oop`, per split part.

2. **Live forcing survives compilation.** The 3-arg wrapper `fo(u,p,t)` **refuses** a
   trace over host buffers (audit J5 / `E_TREEWALK_XLA_LIVE_FORCING`); the supported
   route is `rhs_with_buffers(fo)(u,p,t,dev)` with
   `dev = map(ConcreteRArray, forcing_buffers(fo))` and `sync_forcing!(dev, host)` at
   each cadence boundary. A `copyto!` into those device arrays between calls is
   observed on the next call with **no retrace** — which is what makes the
   macro-stepped runner possible at all.

3. **The traced program is grid-independent, but only after three emitter fixes.**
   The original fear — that the pointwise chemistry lift degraded to O(#cells)
   scalar-spine reads — was real, and it was fixed upstream in EarthSciAST rather
   than worked around here: lane-batching the per-cell scalar surface, a level-major
   prefix scan, and read interning that emits one read per (SSA value, window).
   Together those took the CONUS compile from *never finished in 3h45m* to ~550 s.

4. **The split does not need a splitter `:oop` path.** Both halves are built `:oop`
   independently over shared forcing buffers, and the Lie-Trotter step is written out
   by hand in `rx_traced_integrator.jl` rather than driven through the SciML solver
   stack (which Reactant cannot trace).

5. **Reactant is not faster than the native runner on CPU for this model** — ~2.8×
   slower per macro step, measured at CONUS. It is the route to a differentiable,
   device-portable simulation, not a speedup. See `../../run_reseact_reactant.jl`.

## Files

| File | Used by | What it is |
|---|---|---|
| `op_split.jl` | `run_reseact.jl` | `lie_trotter_solve` / `lie_trotter_solve_bisect` — drives `OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov((SSPRK43, Rosenbrock23/BlockDiagonal))` as a macro-step loop, with the positivity clamp, the forcing-refresh fencepost, and the bisect-on-failure retry. `include`s `blockdiag_similar.jl`. |
| `blockdiag_similar.jl` | via `op_split.jl` | `Base.similar(::BlockDiagonal, ::Type)` + `Base.zero(::BlockDiagonal)`. **The densification fix**: without them DiffEqBase's `promote_f` turned the block Jacobian into a dense N×N matrix at every `init`. Type piracy; durable home is EarthSciMLBase. |
| `rx_native_patch.jl` | `run_reseact_reactant.jl` | Two trace enablers: native `stablehlo` broadcast lowering (kills the 10,000-name cap) and `make_tracer` opaque-leaf registration for EarthSciAST's node types (skips an O(IR-size) capture walk — hours → minutes). **Runtime monkey-patch**; needs a durable home in `EarthSciASTReactantExt`. |
| `rx_traced_integrator.jl` | `run_reseact_reactant.jl` | The purpose-built traced integrator: ROS23 + SSPRK43 stage algebra, batched pivot-free block Gaussian elimination over cells, masked block FD Jacobian, exact PI controller, and the `@trace while` adaptive loop. This is why one macro step compiles as a single `stablehlo.while` program. Generic — not reseact-specific. |

`HELPERS.md` §2 tracks where each of these should eventually live so the runners can
shrink to `using`-lines.

## Pointers

- Extension: `EarthSciAST/pkg/EarthSciAST.jl/ext/EarthSciASTReactantExt.jl` — the
  docstring explains every seam and why `f!` cannot trace.
- Working usage + the two silent-staleness traps: `.../test/reactant_oop_test.jl`.
- Buffers/refresh API: `.../src/tree_walk/oop.jl` (`rhs_with_buffers`,
  `forcing_buffers`) and `.../src/data_refresh.jl` (`sync_forcing!`).
