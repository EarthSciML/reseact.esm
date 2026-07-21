# Handoff: running the full ReSEACT model under Reactant.jl

**Goal.** Determine whether EarthSciAST's Reactant/XLA weakdep can actually compile
and run the full ReSEACT 7×7×72 model (`../../reseact.esm`), and if not, what blocks it.

**Status: inconclusive on the machine it was attempted on (Apple M1, 8 GB RAM) — the
XLA `@compile` is memory-bound and did not finish there.** Everything *up to* the
compile works; the compile itself needs to be run on a bigger box. This directory has
a portable probe (`rx_probe.jl`) that runs the whole experiment in one shot. Pick it
up on a machine with lots of RAM (≥32 GB, ideally 64 GB).

---

## What was already established (all verified on the 8 GB machine)

1. **The Reactant path traces the out-of-place (`:oop`) RHS, and nothing in `tools/`
   builds one.** `ext/EarthSciASTReactantExt.jl` traces `build_evaluator(model; form=:oop)`.
   The reseact tooling (`prototypes/reseact_3d_chem/split_common.jl`) only builds the
   operator-**split in-place** `f!` pair, and an in-place `f!` captures a host scratch
   buffer per `_VecNode`, so it is *fundamentally* untraceable (the extension docstring
   says so explicitly). `rx_probe.jl` therefore builds the **whole, unsplit** model as a
   single `:oop` RHS itself — this is the missing glue.

2. **That single `:oop` RHS builds and evaluates fine.** `build_evaluator(doc; form=:oop, …)`
   over the flattened full model: **~45 s, 45,864 states**, host eval all-finite and
   bit-agreeing with the `rhs_with_buffers` form. So the model *has* a valid traceable RHS.

3. **`interp.linear` is NOT a blocker.** The one hit in `reseact.esm` is inside a *description
   string* (the `dp` observed's docstring). The model deliberately reads dA/dB directly, so
   the known interp-doesn't-trace gap (`reactant_oop_test.jl`) does not apply here.

4. **Forcing is the expected constraint, and it is handled.** The model binds **6 live
   GEOS-FP buffers** (`GEOSFP_I3.PS`, `_I3.T`, `_A3dyn.U/V/OMEGA`, `_A1.PBLH`) via
   `param_arrays`. The 3-arg wrapper `fo(u,p,t)` must **refuse** a trace over host buffers
   (audit J5 / `E_TREEWALK_XLA_LIVE_FORCING`); the supported route is
   `rhs_with_buffers(fo)(u,p,t,dev)` with `dev = map(ConcreteRArray, forcing_buffers(fo))`
   and `sync_forcing!(dev, host)` at each cadence boundary. `forcing_buffers` /
   `forcing_buffer_index` / `rhs_with_buffers` all return correctly on this model (verified
   on host).

5. **The blocker that stopped us: memory.** On 8 GB RAM the first `@compile` (even *Test A*,
   which only has to reach the refusal) ran **>11 min without finishing** while macOS swap
   grew past **23 GB** (process pinned in uninterruptible-sleep, thrashing). Killed to protect
   the machine. This is a *machine* limit, not proof of a code blocker — hence the handoff.

---

## The key open question for the next run

Is the traced program **grid-independent in size**, as the extension intends, or did it
degrade to **O(#cells)**? The pointwise chemistry lift emits warnings on this model:

```
pointwise lift: grid dimension 1 of species 'SuperFast.OH' (extent 7) matches multiple
declared index sets ["lat","lon"] by size; … a synthetic dense axis '_liftdim1_7' is used …
```

The **7×7 square grid makes `lat` and `lon` indistinguishable by size**, so the lift falls
back to synthetic dense axes. **If** that fallback turns the lifted chemistry into
scalar-spine (`@allowscalar`) reads/writes per cell instead of whole-array lane ops, the
traced MLIR becomes O(#cells) and *that* is what exploded memory. Two ways to tell:

- **Compile-scaling test (decisive):** run `rx_probe.jl` with `RESEACT_MODEL` pointed at a
  **tiny** grid and then a **larger** one; if compile time/memory grows with cell count, the
  program is not grid-independent → the lift degradation is the real blocker. Build grids with
  `tools/build_full_model.py` (see below). *Note:* build the grids with **NLON ≠ NLAT**
  (e.g. 4×3, 8×5) to dodge the square-grid index-set ambiguity and see whether the warnings —
  and any O(#cells) blowup — go away.
- If it *is* grid-independent, then a ≥32 GB machine should compile 7×7×72 outright and
  Tests B/C will answer "does it run" directly.

---

## Exact steps on the larger machine

```bash
cd <this-repo>          # the reseact.esm repo; siblings EarthSciDiscretizations,
                        # EarthSciModels, EarthSciAST must be checked out alongside it

# 1. Julia env WITH Reactant. run-model-jl is gitignored and has everything EXCEPT
#    Reactant. Add it (Reactant pulls a bundled XLA runtime, ~large):
julia --project=run-model-jl -e 'using Pkg; Pkg.resolve(); Pkg.add("Reactant"); Pkg.precompile()'
#    (Pkg.resolve() is needed regardless: EarthSciAST added a RuntimeGeneratedFunctions dep
#     that the committed manifest predates — a fresh checkout fails to precompile without it.)

# 2. Run the probe on 7x7x72 (default model):
julia tools/reactant_handoff/rx_probe.jl 2>&1 | tee /tmp/rx_probe_7x7.log

# 3. Compile-scaling test — build a couple of grids and re-run the probe on each.
#    IMPORTANT: shell word-splitting differs (this repo's default shell is zsh, which does
#    NOT split unquoted $vars); pass NLON/NLAT explicitly, one build per invocation:
OUT=/tmp/reseact_4x3x72.esm NLON=4 NLAT=3 NLEV=72 python3 tools/build_full_model.py
OUT=/tmp/reseact_8x5x72.esm NLON=8 NLAT=5 NLEV=72 python3 tools/build_full_model.py
RESEACT_MODEL=/tmp/reseact_4x3x72.esm julia tools/reactant_handoff/rx_probe.jl 2>&1 | tee /tmp/rx_4x3.log
RESEACT_MODEL=/tmp/reseact_8x5x72.esm julia tools/reactant_handoff/rx_probe.jl 2>&1 | tee /tmp/rx_8x5.log
```

Watch memory while it runs (macOS: `sysctl vm.swapusage`; Linux: `free -g` / `/usr/bin/time -v`).

### What each test prints / means
- **Test A** should say `refused. mentions param_arrays=true const_arrays=true`. If instead it
  compiles, the J5 live-forcing guard regressed.
- **Test B** is the verdict. `@compile SUCCEEDED` + `approx=true` ⇒ **the RHS runs under
  Reactant.** `@compile FAILED ->` prints the first unsupported op / unhandled seam — *that*
  is the concrete code blocker to file against EarthSciAST.
- **Test C** confirms live GEOS-FP forcing survives compilation (`sync_forcing!` observed).

---

## The remaining blocker even if Test B passes: the SOLVE

A compiled RHS is not yet a simulation. ReSEACT is split for a reason (`split_common.jl`
header): it is **too stiff for a monolithic solve** — chemistry is stiff and block-diagonal,
transport is non-stiff. Reactant compiles **one** RHS, and the splitter has **no `:oop`
path**, so there is currently no way to drive the real operator-split scheme
(SSPRK43 transport ⊕ Rosenbrock23/BlockDiagonal chemistry, see `tools/run_scale.jl`) through
XLA. Options for a genuine end-to-end run, in rough order of effort:
1. Compile only the **transport** operator's `:oop` RHS and keep chemistry on the CPU
   block-diagonal path — needs an `:oop` emit per split part (splitter change).
2. Give `EarthSciASTSplitter` an out-of-place emit so each part yields a traceable RHS.
3. A monolithic implicit solve on the compiled RHS — simplest to wire, but this is exactly
   the stiff dense solve the split was built to avoid; likely intractable at 45,864 states.

So the honest framing of "does it work": **the RHS is expected to compile (pending the
larger-machine run); a full simulation additionally needs an `:oop` split path that does
not exist yet.**

---

## Files here
- `rx_probe.jl` — self-contained, portable probe. Builds the `:oop` RHS and runs Tests A/B/C.
  Paths derive from its own location; `RESEACT_RXENV` and `RESEACT_MODEL` override env/grid.

## Pointers
- Extension: `EarthSciAST/pkg/EarthSciAST.jl/ext/EarthSciASTReactantExt.jl` (read the
  docstring — it explains every seam and why `f!` can't trace).
- Working usage + the two silent-staleness traps: `.../test/reactant_oop_test.jl`.
- Buffers/refresh API: `.../src/tree_walk/oop.jl` (`rhs_with_buffers`, `forcing_buffers`) and
  `.../src/data_refresh.jl` (`sync_forcing!`).
- Split machinery / why it's split: `prototypes/reseact_3d_chem/split_common.jl`.
