# Full ReSEACT model — build + horizontal-grid scaling

`../reseact.esm` (repo root) is the **full** ReSEACT model: 3-D monotone-PPM transport
driven by real GEOS-FP 4×5 meteorology, coupled to the 12-species SuperFast gas-phase
chemistry mechanism, on a 72-level hybrid-sigma column. It is the same model as
`prototypes/reseact_3d_chem/reseact_3d_chem.esm` at a 7×7×72 grid — regenerated at the
repo root (with refs re-relativized) by the tooling here.

Sibling repos `EarthSciDiscretizations` and `EarthSciModels` must be checked out next to
this one (same parent directory); the model refs them by relative path.

## Regenerate / re-grid the model

> **The generators no longer reproduce the root `reseact.esm`. Do not overwrite it
> with `OUT=$PWD/reseact.esm`.** They still reproduce
> `prototypes/reseact_3d_chem/reseact_3d_chem.esm`, which is what the bit-for-bit
> claim below refers to, but the root model has since grown a substantial amount of
> hand-written science that lives nowhere else: Fast-JX photolysis and its solar
> chain, Wesley dry deposition, EMEP wet deposition, the real 3-D water field, and
> the column-local pressure fixer (`dPSdt` / `divh` / `divh_col` / `dp_col` /
> `divh_fix`, and the diagnosed `Mz` built from them). `grep divh_fix` finds
> nothing in the generators. Regenerating over the root model silently reverts all
> of it — the result still loads, still runs, and still looks like ReSEACT, which
> is exactly what makes it dangerous.
>
> To re-grid or re-position the root model, set its **metaparameters** at load
> instead — `NLON`/`NLAT`/`NLEV` for the extent and `LON0`/`LAT0` for where on the
> native GEOS-FP 4x5 grid the slice sits (`prepare_split_docs(MODEL;
> metaparameters = ...)`, or `RESEACT_NLON` / `RESEACT_LON0` etc. on
> `tools/diurnal_run.jl`). The `.esm` is O(1) in cell count, so that is the
> supported path and needs no regeneration at all. Build the metaparameter dict
> with `native_slice` rather than by hand: it also produces the `lon0_deg` /
> `lat0_deg` parameters that have to move with `LON0`/`LAT0` and that nothing in
> the model derives (see HELPERS.md §1).

`build_full_model.py` chains the three committed generators
(`prototypes/transport_3d/gen_t3d.py` → `reseact_3d/gen_r3d.py` →
`reseact_3d_chem/gen_c1.py`), parameterized by horizontal grid. At 7×7×72 it reproduces
the committed `reseact_3d_chem.esm` bit-for-bit (ref-normalized).

```bash
# regenerate the PROTOTYPE (not the root model -- see the warning above)
OUT=/tmp/reseact_3d_chem_regen.esm NLON=7 NLAT=7 NLEV=72 python3 tools/build_full_model.py

# a larger horizontal grid
OUT=/tmp/reseact_28x14x72.esm NLON=28 NLAT=14 NLEV=72 python3 tools/build_full_model.py
```

Grid limits (native GEOS-FP 4×5 window at LON_OFF=14, LAT_OFF=29): **NLON ≤ 57, NLAT ≤ 17**.
The `.esm` is O(1) in cell count — grid size lives in rule bindings and slice offsets,
not expanded content, so every file is ~162 KB regardless of grid.

## Build + run

`run_scale.jl` builds one model and runs a short operator-split (Option A: SSPRK43
transport + Rosenbrock23/BlockDiagonal chemistry) solve. First run downloads GEOS-FP
fields from S3 (cached thereafter).

```bash
# root model at 7×7×72, 60 s solve window
RESEACT_LABEL=7x7x72 RESEACT_SOLVE_SECS=60 julia tools/run_scale.jl

# a specific model
RESEACT_MODEL=/tmp/reseact_28x14x72.esm RESEACT_LABEL=28x14x72 julia tools/run_scale.jl
```

`run_sweep.jl` runs several grids in one warm process (precompile + machinery JIT paid
once via a warmup build), for apples-to-apples scaling numbers:

```bash
julia tools/run_sweep.jl \
  7x7x72=$PWD/reseact.esm \
  10x10x72=/tmp/reseact_10x10x72.esm \
  28x14x72=/tmp/reseact_28x14x72.esm
```

## Measured scaling (warm process, 60 s solve window, this machine)

| grid      | cells  | states  | build | solve | cells× | build× | solve× |
|-----------|-------:|--------:|------:|------:|:------:|:------:|:------:|
| 7×7×72    |  3,528 |  45,864 |  53 s |  27 s |  1.0×  |  1.0×  |  1.0×  |
| 10×10×72  |  7,200 |  93,600 |  69 s |  59 s |  2.0×  |  1.3×  |  2.2×  |
| 14×14×72  | 14,112 | 183,456 |  64 s | 110 s |  4.0×  |  1.2×  |  4.1×  |
| 21×14×72  | 21,168 | 275,184 |  73 s | 172 s |  6.0×  |  1.4×  |  6.4×  |
| 28×14×72  | 28,224 | 366,912 |  91 s | 224 s |  8.0×  |  1.7×  |  8.35× |

- **Build is strongly sublinear** — 8× the grid costs only ~1.7× the build; build-per-cell
  falls from 15 → 3.2 ms. The affine/polyhedral build is ~O(structural groups), not O(cells).
- **Solve is linear** — ~7.9 ms/cell per 60 s window, steady across all grids; stiff-chem
  accepted steps stay ~110–114 (stiffness is per-cell, grid-independent).
- **Correctness holds at every size** — both operators `Success`, air-mass m > 0, and
  O₃ ∈ [39.38, 39.46] ppb identical across all grids (only the domain extent grows).

(Cold single-shot `run_scale.jl` build is ~98 s at 7×7×72; the warm-process figures above
exclude Julia precompilation and one-time build-machinery JIT.)
