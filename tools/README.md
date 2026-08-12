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

## Parameter sensitivities (forward mode)

`sensitivity_forward.jl` computes **d(scalar objective)/d(parameter)** through the
compiled Reactant macro step and chains it across a short macro-step window. It is
Phase 2 of `../DIFFERENTIABILITY_PLAN.md`, and it exists because forward mode crosses a
`Reactant.@trace while` region exactly while reverse mode cannot — so a sensitivity
w.r.t. a *handful* of parameters is available today, at a cost proportional to how many
you ask for, while the full adjoint waits on Phase 3.

```bash
RESEACT_LON0=14 RESEACT_LAT0=29 RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8 \
RESEACT_SENS_PARAM=NEIRegrid.scale,Transport3D.tau_pblmix \
  julia --project=run-model-jl tools/sensitivity_forward.jl
```

It **always** validates itself against a central finite difference of the same
objective, over a step-size sweep rather than one `h`, through the *same* compiled
program with the tangent seed zeroed — so fd and AD never differ merely because they
ran different arithmetic. One `@compile` serves every parameter (a one-hot tangent seed
picks the direction) and every fd evaluation (`p` is a real XLA input, so new parameter
values do not retrace); the compile is the expensive part and it is paid once.

Measured at **CONUS 13×7×72**, `T0=5400`, three 300 s macro steps, objective =
domain-mean **surface** O₃ at the end of the window (39.6531 ppb):

| parameter | ∂J/∂θ | units | best fd/AD rel |
|---|---|---|---|
| `NEIRegrid.scale` | −7.387076375067e−02 | ppb per unit scale | 5.5e−9 |
| `Transport3D.tau_pblmix` | −6.906232719590e−05 | ppb per second | 1.6e−7 |
| `NEIRegrid.g0` | −7.532721546163e−03 | ppb per m s⁻² | 7.5e−9 |

Negative because the window is at night, when NOx titrates O₃. Build 770 s, JVP
`@compile` 770 s, ~124 s per 3-macro-step window pass; ~2 h for the whole self-check.

The chain is pinned separately by an identity finite differences cannot fake: emissions
are literally `∝ scale·g0` and `g0` appears nowhere else, so
`scale·dJ/dscale ≡ g0·dJ/dg0`, and two independently seeded forward sweeps satisfy it to
2.3e−13. Note that the fd/AD agreement is **domain-size dependent** — the same driver at
6×6×8 floors at ~5e−7, flat across four decades of `h`, which is a small fixed set of
kinked states (`clamp_nonneg`'s `max(u,0)`, the `max(|u|,1e-9)` in the FD block-Jacobian
step, the controller's `q` clamps) dominating a 36-cell mean. Validate at a realistic
domain, and check an adjoint against the AD number rather than against fd.

**`NEIRegrid.F_NO` … `F_FORM` are not differentiable through this route, and they are
not split fractions.** They are the whole NEI2016 12US1 source-grid emission *fields*,
wired in as CONST provider arrays and collapsed at build time by the conservative
regrid (deliberately — see `prototypes/reseact_3d_chem/split_common.jl`). They never
reach the runtime `p` (which is 49 scalars and nothing else), so the driver rejects
them by name rather than returning a zero that a finite difference would happily
"confirm".

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
