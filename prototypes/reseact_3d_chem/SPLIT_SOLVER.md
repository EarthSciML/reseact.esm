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

_(filled in from the 3600 s run — pending)_
