# ReSEACT de-risk prototypes (G1, G2)

Two minimal, runnable v0.8.0 prototypes that resolve the two "genuine gap" risks from the
feasibility audit, plus the EarthSciAST/schema findings uncovered while building them.

Run environment: `reseact.esm/run-model-jl/` (copied from wildlandfire; `Pkg.develop`s
`../EarthSciAST/pkg/EarthSciAST.jl` and `../EarthSciIO/julia`, adds a stiff solver).

---

## G2 — deposition as an additive sink on chemistry species  ✅ PROVEN

**Question:** can first-order deposition loss be applied as `dc/dt += -k*c` onto SuperFast
species in v0.8.0?  **Answer: yes — via `operator_compose` merging a model whose equations are
`D(SuperFast.<X>) ~ -(k)*SuperFast.<X>`.**

Files:
- `../../EarthSciModels/components/atmospheric_deposition/superfast_deposition_sink.esm` —
  shared, reusable applicator model (imported **by ref**). `k_*` params are driven from the
  real DryDepositionGas/WetDeposition in a full assembly via `param_to_var`.
- `g2_depsink/g2_dep.esm` / `g2_nodep.esm` — 0-D SuperFast (inlined) with / without the sink,
  `g2_dep` mounts the shared sink model by ref + one `operator_compose` coupling.
- `g2_depsink/g2_run.jl` — simulates both for 1 s, prints per-species Δ.

Result (exaggerated rates for a visible 1 s effect): O3 39.99→38.04 (−1.95 ppb ≈ −k·c·Δt);
HNO3/H2O2/CH2O get **both** dry+wet sinks summed; non-deposited species show only small
indirect chemical feedback. `julia prototypes/g2_depsink/g2_run.jl`.

**Finding — the `couple`/`additive` connector does NOT work.** The spec-sanctioned form
(`{from,to,transform:"additive",expression}`, esm-spec §10.3) expands correctly but
`flatten`'s `_apply_couple!` only applies `{lhs,rhs}` connector equations; a
`transform:"additive"` edge is pushed to `opaque_refs` and **silently dropped** (no ODE effect).
Use `operator_compose` (which sums equations sharing a dependent variable) instead.

---

## G1 — vertical layer geometry (Δz) from GEOS-FP hybrid Ap/Bp  ✅ PROVEN

**Question:** is layer thickness `dz` (and `dP`, edge pressures) expressible/evaluable in
v0.8.0, given geosfp.esm marks `δPδlev`/`Z_agl` "not representable"?  **Answer: yes** — the
note only applied to the `DataInterpolations.derivative` spelling; the edge-pressure /
hypsometric form needs only an **offset lookup** (Ap/Bp at `lev` and `lev+1`) + arithmetic + `log`.

Files: `g1_layergeom/g1.esm` (built by `scratchpad/build_g1.jl`), `g1_layergeom/g1_run.jl`.
`P_edge(q) = Ap(q) + Bp(q)*PS`, `dP = P_lo - P_hi`, `dz = (Rd/g)*T*ln(P_lo/P_hi)`.
Verified against a direct Ap/Bp computation at lev = 1,10,30,50,60,72 — exact match
(dz: 115 m surface → 5275 m near top). `julia prototypes/g1_layergeom/g1_run.jl`.

**Finding — use `interp.linear`+`const`, NOT `table_lookup`, for runtime observeds.** The
tree-walk simulate path does not lower/evaluate the `table_lookup` op
(`E_TREEWALK_UNSUPPORTED_OP: table_lookup`); it supports `interp.linear`/`interp.bilinear` with
`const`-array args (as `fastjx.esm` is authored). The function-table conformance README says
`table_lookup` "lowers to interp.linear", but that lowering is not applied in `flatten→simulate`.
Consequence: `geosfp.esm`'s `P` observed (authored with `table_lookup`) would not evaluate in
the tree-walk runtime as-is — it needs the `interp.linear` form (or the lowering pass wired in).

---

## Cross-cutting findings (affect the full ReSEACT build)

1. **Migration to v0.8.0 = content edits, not a version bump.** The `"esm"` string is not the
   gate; content validity is. Per component: SuperFast — delete `coupletype` (**done**, in place);
   wesley_dry_gas — loads as-is; fastjx — delete `coupletype` + unwrap `initial_state`
   (`{type:"per_variable",values:X}`→`X`); wet_deposition — unwrap `initial_state`; geosfp/nei2016 —
   delete model-level `regrid` + the loader `grid` blocks (this drops source-grid geometry —
   real regridding becomes an assembly-time `aggregate`/injection concern).
2. **Reaction systems cannot be imported by ref.** The top-level model-ref inliner and the
   subsystem-ref resolver both require a model/data-loader; no mechanism mounts an external
   `reaction_systems` file. SuperFast must be **inlined** (the prototypes generate the inline
   block from the canonical `superfast.esm` via jq). Models, couplings, and data-loaders *can* be
   ref'd. (A ref mechanism for reaction systems would be an EarthSciAST enhancement worth requesting.)
3. **`coupling_import` refs resolve against the process CWD**, not the assembly file's dir
   (`simulate` calls `flatten(input)` with default `base_path="."`). Workaround used here:
   pre-`flatten(f; base_path=dirname(path))` and pass the flattened system to `simulate`.
4. **The tree-walk RHS is not ForwardDiff-compatible** — stiff solvers must use a
   finite-difference Jacobian (`Rosenbrock23(autodiff=false)`), not autodiff.
