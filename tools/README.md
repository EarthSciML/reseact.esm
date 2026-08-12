# Full ReSEACT model — build + horizontal-grid scaling

`../reseact.esm` (repo root) is the **full** ReSEACT model: 3-D monotone-PPM transport
driven by real GEOS-FP 4×5 meteorology, coupled to the 12-species SuperFast gas-phase
chemistry mechanism, on a 72-level hybrid-sigma column. Its metaparameter defaults are
**CONUS, 13×7×72** (`LON0=11`, lon −125..−65) — the domain `../run_reseact.jl` and
`../run_reseact_reactant.jl` simulate by default. It descends from
`prototypes/reseact_3d_chem/reseact_3d_chem.esm` (the 7×7×72 central-US box,
`LON0=14`), but it has since grown a great deal of hand-written science that lives
nowhere else — see below.

Sibling repos `EarthSciDiscretizations` and `EarthSciModels` must be checked out next to
this one (same parent directory); the model refs them by relative path.

## Re-grid or re-position the model

Set its **metaparameters** at load: `NLON`/`NLAT`/`NLEV` for the extent, `LON0`/`LAT0`
for where on the native GEOS-FP 4×5 grid the slice sits. The `.esm` is O(1) in cell
count — grid size lives in rule bindings and slice offsets, not expanded content, so
the file is the same size at every grid and **no regeneration is involved**.

Build the metaparameter dict with `native_slice` rather than by hand: it also produces
the `lon0_deg` / `lat0_deg` parameters that have to move with `LON0`/`LAT0` and that
nothing in the model derives (see HELPERS.md §1). Or set `RESEACT_NLON` / `RESEACT_LON0`
etc. on `../run_reseact.jl`, which does that for you.

Grid limits: **NLON ≤ 57 − LON0, NLAT ≤ 46 − LAT0, NLEV ≤ 72** (`native_slice` checks
the halos against the native 72×46 grid and throws rather than reading off the edge).

**There is no full-model generator any more, deliberately.** `build_full_model.py`
used to chain `prototypes/transport_3d/gen_t3d.py` → `reseact_3d/gen_r3d.py` →
`reseact_3d_chem/gen_c1.py` to emit a whole model at a chosen grid. Both of its jobs
are gone: re-gridding is metaparameters (above), and the generators cannot reproduce
the root model at all. Everything added by hand since — Fast-JX photolysis and its
solar chain, Wesley dry deposition, EMEP wet deposition, the real 3-D water field, and
the column-local pressure fixer (`dPSdt` / `divh` / `divh_col` / `dp_col` / `divh_fix`
and the diagnosed `Mz` built from them) — exists only in `reseact.esm`; `grep divh_fix`
found nothing in the generators. Running it over the root model silently reverted all
of that, and the result still loaded, still ran, and still looked like ReSEACT. A
warning was not worth keeping a tool whose every correct use had become "don't".

`reseact.esm` is now edited directly and is the source of truth. The per-prototype
generators under `prototypes/` are untouched — each still reproduces its own prototype
`.esm`, which is what those READMEs tell you to edit.

## Run the model

Use the two root runners — `../run_reseact.jl` (native) and
`../run_reseact_reactant.jl` (Reactant/XLA). Both default to a **one-week CONUS
simulation** and take their configuration from the environment; see their headers
and HELPERS.md. First run downloads GEOS-FP fields from S3 (cached thereafter).

```bash
# the default: one week of CONUS, native, threaded
julia -t 8 --project=run-model-jl run_reseact.jl

# a short validation window on the small central-US box instead
RESEACT_LON0=14 RESEACT_NLON=7 RESEACT_SOLVE_SECS=3600 \
  julia -t 8 --project=run-model-jl run_reseact.jl
```

## Measuring how it scales

`scaling_study.jl` runs a ladder of (grid, window) points in ONE warm session and
times **build**, **first macro step** and **subsequent steps** separately. That
separation is the whole point: the first step carries the one-time codegen compile
and can be ~800× the cost of the identical second step, so an undifferentiated
"wall time vs cells" curve measures the Julia compiler rather than the model.
`solve_s / (nT+nC)` is the number that extrapolates.

```bash
RESEACT_SWEEP=horizontal RESEACT_CSV=/tmp/scaling.csv julia -t 8 tools/scaling_study.jl
```

Each point re-binds the grid as `.esm` metaparameters, so no pre-generated model
files are involved. Minimum extent per axis is **6**, not 1 — the PPM rules carve
three boundary regions per end plus an interior `[4, N−3]`, so `NLEV=4` fails to
load with `makearray_region_inverted`.

Two findings that have held at every size measured so far: **build is strongly
sublinear** in cell count (the affine/polyhedral build is ~O(structural groups),
not O(cells)), and **solve is linear**, with stiff-chemistry accepted steps roughly
grid-independent because stiffness is per-cell. Current absolute timings for the
CONUS configuration are in HELPERS.md; the older per-grid table that used to sit
here was measured with `run_scale.jl` (deleted, along with `run_sweep.jl` — both
needed one pre-generated `.esm` per grid) under the pre-merge codegen default, so
it is not comparable to anything you can run today and has been removed rather
than left to be misread.
