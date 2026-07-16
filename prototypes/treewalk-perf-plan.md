# Tree-walk RHS performance plan (EarthSciAST) — investigation + proposed fixes

**Status: investigation + plan only. No EarthSciAST code changed (per instruction).**

## Symptom
The ReSEACT 0-D box (SuperFast + FastJX photolysis) run 24 h with a stiff solver
(`Rosenbrock23(autodiff=false)`) was killed mid-integration after **~21e9 allocations**
making little progress.

## Measurements (profiled on merged `main`, box.esm)
Per RHS (`f!`) call, warm:

| system | states | allocs/call | time/call |
|---|---|---|---|
| SuperFast only (reaction system) | 15 | **0 B** | 3.0 µs |
| box (SuperFast + FastJX) | 15 | **~81 KB** | 169 µs |

So the entire per-call cost is **FastJX**, and it is **100% in the `interp.linear`
evaluation path** — SuperFast (pure arithmetic reaction rates) is allocation-free.

Allocation attribution (`Profile.Allocs`, box RHS):
- `registered_functions.jl:272-320` `_validate_interp_axis` / `_interp_linear_core` —
  the axis-monotonicity **validation runs on every evaluation**, and the core allocates a
  `Memory{Float64}`.
- `compile.jl:141` `_fn_const_arg_spec` — scanning the heterogeneous `_FN_CONST_ARG_SPECS`
  tuple **boxes the returned NamedTuple every call** (~470 KB / 100 calls).
- `compile.jl:716-731` `_eval_node_op` `:fn` arm — `args = Vector{Any}(undef, arity)` +
  `enumerate(const_positions)` allocate a `Vector{Any}` / `Memory{Any}` / `Enumerate` /
  `UnitRange` **per fn-node evaluation**, and the args are boxed `Float64`s.

FastJX has ~100 `interp.linear` observeds ⇒ ~100 × (these per-fn allocations) ≈ 81 KB/call.

## Why it explodes under a stiff solve
1. **Per-call allocation** (81 KB) — structural, independent of the values.
2. **Finite-difference Jacobian** does `N+1 = 16` RHS calls per Jacobian column-block; each
   recomputes FastJX in full.
3. **FastJX is state-independent** — it depends only on `t` and parameters (lat, lon, t_utc,
   T, P, H2O), never on the chemistry state. So 15 of every 16 FastJX evaluations inside a
   Jacobian are **recomputing an identical result**.

81 KB × 16 × (thousands of stiff steps) ≈ the observed ~21 GB.

---

## Proposed fixes (ranked by ROI)

### Fix 1 — Make `interp.*` evaluation allocation-free  ★ highest ROI, localized
Turn the box RHS from 81 KB/call → ~0 (like SuperFast) by removing all per-call allocation
in the `:fn`/interp path. Files: `src/tree_walk/compile.jl`, `src/tree_walk/registered_functions.jl`.
1. **Validate interp axes once, at compile/build time**, not per evaluation. `_validate_interp_axis`
   (monotonic/sorted checks) belongs in `_compile_op` when the const arrays are pinned into the
   node payload — not in `evaluate_closed_function` on the hot path.
2. **Resolve the const-arg spec at compile time.** `_fn_const_arg_spec` is a compile-time
   constant per node; store the resolved arity/const_positions on the node (or specialize the
   node) so the hot path does no tuple scan / NamedTuple boxing.
3. **Eliminate the per-call `Vector{Any}` arg splice.** interp.* have fixed small arity (3/5);
   build the argument list as a `Tuple` (or evaluate scalar args directly into a monomorphic
   `_interp_linear_core(table, axis, x)` call) instead of `Vector{Any}(undef, arity)` +
   `enumerate`. Avoids `Vector{Any}` / `Memory{Any}` / `Enumerate` / boxed `Float64`.
4. **Non-allocating `_interp_linear_core`** — index the captured const array in place; no
   `Memory{Float64}` temporary.
The cleanest form: at compile time lower each interp node into a specialized closure that
captures the (validated) const table+axis and evaluates `x -> Float64` with zero allocation.
Expected result: box RHS ≈ 0 alloc, per-call time dominated by the ~100 scalar interpolations
(µs range). This alone should make the 24 h stiff solve tractable.

### Fix 2 — Hoist / cache state-independent (time-only) observeds  ★ structural, higher effort
FastJX (and any met/data-interp forcing) depends only on `t` + params. Partition observeds
into **state-independent** (recompute only when `t` changes) vs **state-dependent**, and cache
the former. Then a finite-difference Jacobian (fixed `t`, perturbed state) reuses the single
FastJX evaluation across all `N+1` columns instead of recomputing it 16×. Benefits every model
with expensive time-only forcing. Interacts with the solver: needs the RHS to memoize on `t`
(and invalidate when the integrator advances/​rejects a step). More invasive than Fix 1; do it
after Fix 1 if the per-call cost (now ~0 alloc but still ~100 interpolations) is still the
bottleneck.

### Fix 3 — Reduce RHS calls per Jacobian  (solver-side, no EarthSciAST change)
- Provide a **sparse / colored Jacobian** (`jac_prototype` + matrix coloring) so the
  finite-difference Jacobian needs far fewer than `N+1` RHS evals — SuperFast's Jacobian is
  sparse and the FastJX→species coupling is low-rank (6 j-rates).
- Or an **analytic Jacobian** for the reaction system (Catalyst can emit one).
- Or operator-split: integrate the (stiff) chemistry with the j-rates held over a short window,
  refreshing FastJX on a cadence callback (met/photolysis are already refreshed this way for
  gridded runs).
These are runner-side levers (the box driver), independent of the tree-walk fixes.

---

## Validation plan (once a fix is implemented)
1. Re-run `scratchpad/profile_rhs.jl` — box RHS allocs/call should drop from ~81 KB toward 0.
2. Re-run `scratchpad/alloc_profile.jl` — the interp.* sites should no longer appear.
3. Re-run the box 24 h stiff solve (`box_run.jl`, realistic H2O + deposition) — should
   complete, with total allocations orders of magnitude below 21 GB.
4. Run the EarthSciAST function-table conformance fixtures (`tests/conformance/function_tables/`)
   to confirm interp.* results are unchanged (bit-for-bit) after the refactor.

## Reproducers (in this session's scratchpad)
- `profile_rhs.jl` — per-call allocs/time, box vs SuperFast-only.
- `alloc_profile.jl` — `Profile.Allocs` attribution by type + EarthSciAST source line.
