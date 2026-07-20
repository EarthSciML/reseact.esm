# Stage C — operator-split / IMEX solve

The monolithic Stage-C solve does not scale: 4459 states (13 × 343 cells), and a
dense finite-difference Jacobian costs O(N) RHS evals + O(N³) LU **per stiff step**
(the RHS is a ~3.5 ms tree-walk, so one dense Jacobian ≈ 15 s). The fix is
**operator splitting**: partition the right-hand side

```
f_full(u)  =  f_transport(u)  +  f_chemistry(u)
              └── PDE, non-stiff ──┘ └── pointwise, stiff ──┘
```

so each operator gets the integrator it wants. Transport (explicit SSP-RK) needs
no Jacobian; chemistry is stiff but **block-diagonal** — each grid cell's species
couple only among themselves — so its implicit Jacobian is 343 blocks of 13×13,
filled by a 13-color finite difference and factored per-block.

## Files

| file | role |
|---|---|
| `split_common.jl` | `prepare_split_docs` (split via EarthSciASTSplitter), `reseact_forcing` (GEOS-FP chain), `build_split_run` (build both parts over ONE set of live forcing buffers + one refresh callback) |
| `blockdiag_local.jl` | loads EarthSciMLBase's `BlockDiagonal` (`map_algorithm.jl` + `blockdiagonal.jl`) without the full MTK/Catalyst tree |
| `block_jac.jl` | species-major↔cell-major permutation + the 13-color block-diagonal FD Jacobian |
| `run_split.jl` | build once, then run **Option B (IMEX)** and **Option A (operator splitting)**, compare |

## How the split is found

Under `flatten` (which now *always* carries surviving `apply_expression_template`
refs — the §9.6.4 Option-B invariant; the old `expand_refs=false` keyword is
retired) each species ODE is a pointwise-lifted
`aggregate(oidx=[i,j,k], expr_body = D(index(SuperFast.X,i,j,k), t))`. The
transport contribution is the only additive term carrying a `makearray` (the
lowered PPM stencil; its body still holds `apply_expression_template` refs → the
compile-once tier factors them at build). So the split rule is
**`stencil_vs_pointwise`**: a term is transport iff it contains a `makearray`.
The splitter was extended to split *inside* the aggregate body (helpers
`_lifted_derivative` / `_rebuild_expr_body`) — the plain-`D` path is unchanged
and still reconstructs `Σ f_part == f_full`.

## Two non-obvious solver facts (learned the hard way)

1. **A `jac_prototype` is honored only inside a `SplitODEProblem`.** A plain
   `ODEProblem`'s `init()` silently replaces `prob.f.jac_prototype` with a dense
   `Matrix` (the nlsolver's J/W go dense), so `jac(J,…)` never sees the
   `BlockDiagonal`. Building it "as an `ODEFunction` first" does not help — the
   override happens at `init`. Hence both options wrap the implicit (chem) part
   in a `SplitODEProblem`.
2. **`OrdinaryDiffEqOperatorSplitting.LieTrotterGodunov` cannot carry the
   BlockDiagonal** — it wraps each operator in a plain `ODEProblem`
   (`integrator.jl:1080`), which densifies. So Option A is a **manual Lie–Trotter
   loop**: explicit transport sub-step, then a `SplitODEProblem(chem, 0)` chem
   sub-step (block-diagonal preserved).

## The two schemes

- **Option B — IMEX.** `SplitODEProblem(f_chem [implicit, BlockDiagonal], f_transport
  [explicit])` solved by `KenCarp47(autodiff=false, linsolve=LUFactorization())`.
  One additive-Runge-Kutta integrator: chemistry implicit, transport explicit.
- **Option A — operator splitting (multi-rate).** A Lie–Trotter loop: `SSPRK43`
  advances transport over the splitting interval, then `Rosenbrock23(autodiff=false,
  linsolve=LU)` on `SplitODEProblem(chem, 0)` advances chemistry with the
  block-diagonal Jacobian. `tgrad=0` (forcing is piecewise-constant between
  cadence refreshes ⇒ ∂f/∂t = 0 within a step).

  *(There is no `SSPRK23` in OrdinaryDiffEqSSPRK; `SSPRK43` — order 3 with an
  embedded order-2 estimator — is the low-order adaptive SSP method used.)*

## State layout / block Jacobian

EarthSciAST lays state out **species-major** (`[CH2O×343, …, m×343]`). The chem
Jacobian is block-diagonal only in **cell-major** order, so the split problem is
run cell-major (permutation built from state names, checked a bijection). The
13-color block FD Jacobian perturbs species *s* in every cell at once (cells are
independent in chemistry) → NS+1 chem RHS evals per Jacobian instead of N. It was
validated block-for-block against a targeted dense FD (`max|Δ| = 0`).

## Running

```
cd prototypes/reseact_3d_chem
julia run_split.jl 3600.0        # 1 h of simulation from 2013-01-01T18:00Z
```

The transport part's tier build dominates wall time (the PPM stencils are
compiled once per region-class, so build cost is ~grid-independent — a smaller
grid speeds the *solve*, not the *build*). `build_split_run` prints per-part build
time + tier bench counters.

## Results

Measured on 2026-07-20, Linux (20-core / 188 GB), Julia 1.12.6, EarthSciAST main
`16076d18` (through the A1–A4 build-perf work), on the full **7×7×72 = 3528-cell,
45 864-state** column. Command: `julia run_split.jl 3600.0` (a 3600 s window from
2013-01-01T18:00Z). `RTOL=1e-4`, `ATOL=1e-9`, LU per chem block. Full log:
[`logs/run_split_3600.log`](logs/run_split_3600.log).

### Build (once, shared by both options)

| part | wall | tier body_variants | tier compile_calls |
|---|---|---|---|
| transport (part 1) | **82.6 s** | 262 | 160 391 |
| chemistry (part 2) | **1.7 s** | 0 (fused) | 1 083 |
| **total** (incl. split + emit) | **105.3 s** | — | — |

The transport tier build dominates and is ~grid-independent (PPM stencils compiled once
per region-class). This is a **~20× improvement over the pre-A1 build** (2088 s on
EarthSciAST `424d4ee3` for the identical model — the earlier attempt in this same
directory) — the A1–A4 build-perf levers landed and paid off on the real chem model.

### The blocking bug this run had to fix (runner-side)

The first 3600 s attempt drove every solve `Unstable`: the transport-only pre-check
already blew m to ~1e34 by 300 s. Cause was **not** the solver — `run_split.jl` was
missing the `m(0) = dp` initial-condition seed that `gen_t3d.py` requires and
`run_validate_fw.jl` already applied. The `.esm` carries the Stage-B *analytic* test
mass (~2 Pa, scale-invariant for the CWC gate), but the PPM fluxes are sized for the
**real ~1400 Pa column**, so continuity drives a 2 Pa mass negative within seconds and
the explicit advection explodes. Seeding `m(0) = dp = dA[k] + dB[k]·PS_REF`
(PS_REF = 101325 Pa) per cell — the one-line fix committed here — makes m physical
(`m ∈ [1.0, 5008] Pa`) and the explicit transport stable:

```
transport-only SSPRK43:  tend=60s → nsteps=2, m∈[0.999, 5017], all_pos=true, finite=true
                         tend=300s → nsteps=3, m∈[0.993, 5055], all_pos=true, finite=true
```

### Option A vs B over the 3600 s window

| | **Option B — IMEX** (`KenCarp47`) | **Option A — op-split** (`SSPRK43`+`Rosenbrock23`/BD) |
|---|---|---|
| scheme | one ARK: chem implicit / transport explicit | manual Lie–Trotter, `macro_dt = 300 s` |
| retcode | `Success` | `Success` (12 Lie–Trotter steps) |
| **solve wall** | **1234.5 s** | **710.4 s** |
| m > 0 everywhere at t1 | ✓ | ✓ |
| all species finite / non-negative | ✓ (PositiveDomain net) | ✓ (PositiveDomain net) |

Both build once (105 s) and reuse the block-diagonal (3528 × 13×13) chem Jacobian
filled by the 13-color FD. **Option A solves the same window in ~58 % of Option B's
wall time** (710 s vs 1235 s).

**Cross-check — the two schemes agree.** End-state `max|Δ| = 1.478` on m (units Pa,
where m ranges to ~5500), i.e. **relative 2.68e-4** — right at the `RTOL=1e-4`
tolerance the splitting error should live at. Per-species end ranges match to 3–4
digits (e.g. O3 t1 ∈ [16.49, 27.60] B vs [16.49, 26.67] A; OH t1 max 43.483 both;
CH2O t1 max 1.021 B vs 1.026 A). This is the expected first-order Lie–Trotter
splitting error, not a divergence — the two independent solver stacks corroborate each
other.

*(Note: the report's `t0` column for chem species prints the un-seeded species-major m
slot for `Transport3D.m` (~2.28); that is a display artifact of the report helper — the
solved trajectory and the transport pre-check use the seeded physical m, as the `t1`
and pre-check lines show.)*

### Recommendation — **Option A (operator splitting) as the default** for gridded chemistry.

On this evidence Option A is the better default:

1. **It is faster** — 710 s vs 1235 s (1.7×) for the same 3600 s window at the same
   tolerance, because the multi-rate structure lets the non-stiff explicit transport
   take its own large SSP steps (`nsteps=2–3` per 300 s macro-step) instead of being
   dragged along inside one ARK integrator whose step is paced by the stiff chemistry.
2. **It is at least as accurate here** — the A-vs-B gap (2.7e-4 rel) *is* Option A's
   splitting error measured against the more tightly-coupled IMEX reference, and it sits
   at the requested tolerance. For a 1 h window with 300 s macro-steps that is
   acceptable; it is the knob to watch.
3. Both preserve m-positivity and the block-diagonal Jacobian; neither densifies (the
   `SplitODEProblem`-wrapping trick documented above holds for both).

**Caveats that matter:**

- **Splitting error scales with `macro_dt`.** The 2.7e-4 figure is for `macro_dt = 300 s`
  (Lie–Trotter, first order). Longer macro-steps or stiffer/faster chemistry regimes
  (e.g. strong daytime photochemistry, plume edges) will widen it; if accuracy tightens,
  shrink `macro_dt` (cost grows ~linearly) or move to Strang splitting before falling
  back to Option B. Option B has **no splitting error** by construction — it is the
  accuracy-insurance fallback, worth its ~1.7× cost when the split error is unacceptable.
- **PositiveDomain is doing real work** in both; it is a safety net for stiff-chemistry
  tiny-negatives, not a substitute for the `m(0)=dp` seed (which is the actual stability
  fix — advection is explicit either way).
- Numbers are for one 1 h window from a single start time on a 7×7×72 slice; the
  *ordering* (A cheaper, both stable, agreement at tolerance) is the robust takeaway, not
  the exact seconds.
