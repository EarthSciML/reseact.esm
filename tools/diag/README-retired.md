# Probes retired with the traced right-hand side

Every compiled program in this repository is emitted by EarthSciAST's direct
StableHLO emitter (`tools/rx_rhs.jl`). The probes listed here named a surface
that no longer exists — `EarthSciAST.rhs_with_buffers`, the
`oop_intern_stats` / `oop_gvn_stats` / `oop_ssa_stats` counters, the
`ESS_OOP_SSA` / `ESS_OOP_GVN` / `ESS_OOP_INTERN` / `ESS_OOP_SHIFT_SLICE` build
flags, `RESEACT_RHS`, or `tools/reactant_handoff/rx_native_patch.jl` — so they
cannot run and are deleted in `6f592ca`; a second group, listed in its own table
below, was deleted when `origin/main` was merged on 2026-09-15.

**The probes themselves are in git history.** `git show 6f592ca^:<path>` prints
any of them, headers, measured results and all. What is NOT recoverable from a
document alone is which measurement a number came from, so that is what this
file records: for each probe, the document it is the provenance for.

| Deleted | Backed |
| --- | --- |
| `direct_rhs_agree.jl`, `direct_rhs_agree.sbatch` | **AGREEMENT.md** in full — the file-mediated comparison of the two emitters at three times across a day: worst relative 6.5e-14 (transport) and 7.4e-16 (chemistry), no non-finite values, both inside the algebraic tolerance class. It ran the two lanes in two processes against one `inputs.bin`, which is why it had a `traced` arm. |
| `direct_rhs_census.jl`, `direct_rhs_census_conus.sbatch` | **DIRECT_RHS_CENSUS.md** in full (emission, agreement, cost and gradient per program, per `ESS_OOP_SSA` setting — measured on the degenerate split AGREEMENT.md supersedes), and **COMPILE_COST.md**'s runtime table: transport `runs` 343.5 s / 3.3 ms / 18,060 ops against `gather` 239.0 s / 1.9 ms / 6,510 ops. |
| `direct_rhs_slot_probe.jl` | **AGREEMENT.md** sections 1 and 2 — the read-before-write classification that showed `Transport3D.Mz[i,j,9]` (flat 4705..4740) to be a structural zero rather than an ordering bug, and the byte counts that exposed the degenerate operator split (part 2's `SuperFast.NO` right-hand side 1,849 B against part 1's 44,542 B). It reflected on the `:oop` closure's `mat_levels` / `acc_plans` / `scan_folds` and replayed the emitter's walk order without compiling anything. |
| `hlo_dump.jl`, `hlo_dump.sbatch`, `hlo_compare.sh` | The trace-time sharing measurements: the engagement counters of the `(operand, window)` read memo (`oop_intern_stats`) and of emission value numbering (`oop_gvn_stats`), and the before/after op tables built from dumped module text. The conclusions are in **HELPERS.md** (read interning — one read per `(SSA value, window)` — as one of the three fixes that took the CONUS compile from *never finished in 3h45m* to ~550 s) and in **DIFFERENTIABILITY_PLAN.md** §6. |
| `p6_shift_slice_time.jl`, `p6_shift_slice_pair.sbatch` | **DIFFERENTIABILITY_PLAN.md** §6, measured negatives: stencil shifts emitted as strided slices converted 67% of transport reads and removed two thirds of the transport VJP's scatter-adds, and the transport VJP was 0.97x; chemistry lost under it. It needed an EarthSciAST branch that was never merged. |
| `p6_ascalled_time.jl` | **DIFFERENTIABILITY_PLAN.md** §6: "the driver's call pattern costs 1–7% over standalone for all four programs" — the measurement that refuted the "in-driver is 10x slower" inference twice. |
| `capC_probe.jl`, `capC_probe.sbatch` | The fixed-lane-capacity chemistry build, which `tools/capacity_chem.jl`, `tools/subcycle_chem.jl` and the shard workers all rest on: a capacity build reproduces the reference right-hand side cell for cell under a RANDOM cell permutation to 2.888e-14 relative, padding lanes cannot leak into a real lane, and a capacity build's cost depends on C rather than on the real grid. Cited from `tools/subcycle_chem.jl` and `tools/diag/subcycle_verify.jl`. |
| `astdiff_probe.jl` | Gate A for `jac=:sym`: EarthSciASTDiff's analytic Jacobian on ReSEACT's chemistry half is `:block_diagonal`, 3744x3744, nnz 19,872, exact to 2.3e-16 against a ForwardDiff JVP — on the host. |
| `astdiff_traced_probe.jl`, `astdiff_traced_probe.sbatch`, `astdiff_conus_probe.sbatch` | Gate B for `jac=:sym`, the numbers `tools/adjoint_gradient.jl` section 1b and `tools/rx_adjoint_check.jl` quote: 5.0e-16 max entry-wise against `jac=:ad`, the dot-product identity `<lam,Jv> == <J'lam,v>` restored to 5.3e-16 (stages 6–9, ~1e-16), and 12.0 s / 109k lines of MLIR against `:fd`'s 49.9 s / 323k. |
| `rhs_kink_probe.jl`, `rx_ssprk_basepoint.jl` | **`run_reseact_adjoint.jl`, "READ THIS BEFORE CHANGING THE BASE POINT"**: the ~9e-6 AD-vs-FD gap at the model's default initial condition is the PPM monotonicity limiter's local-extremum test sitting at exact zero on a spatially uniform field, so both signs of a perturbation flip the same branch and every difference quotient lands on the other one. Jittering the base point collapses the gap to FD accuracy — which is why `RESEACT_ADJ_UJITTER` exists and why the FD acceptance stages use it. |
| `rof_repro.jl`, `rof_batchfwd.jl`, `rof_sweep.sh` | **`tools/diag/UPSTREAM_reverse_over_forward.md`** and `rof_results.tsv`: the model-free bisection of the Enzyme-MLIR reverse-over-forward segfault (`AutoDiffCallRev` → `func::CallOp::build` → `getAttr` on a null `FuncOp`), and the batched-forward-mode comparison of three ways to get the same NS Jacobian. `rof_concat_repro.jl` and `rof_bisect_pattern.sh`, which need neither the model nor the patch file, are kept. |
| `tools/reactant_handoff/rx_native_patch.jl` | Not a probe. It lowered same-shape primitive broadcasts straight to `Reactant.Ops.*`, which kept the traced emitter under MLIR's 10,000-name cap; the direct emitter never mints those helpers. Its second half, the `make_tracer` opaque-leaf registration for EarthSciAST's IR types, is still needed and lives in `tools/rx_rhs.jl`. |

Two probes were kept and re-pointed rather than deleted, because their question
survives the emitter they were written against: `direct_rhs_compile_cost.jl`
(the harness behind **COMPILE_COST.md**, which takes one program's `@compile`
apart into trace / pass pipeline / XLA:CPU codegen) and
`p6_rxops_semantics.jl`.

## The build-scaling probes, retired in the merge of 2026-09-15

`origin/main` landed a build-and-resolution scaling investigation
(**BUILD_SCALING.md**, **SCALE_025.md**) on the branch point that still had the
traced right-hand side. Its probes all share one line — a `sha256` /
finiteness acceptance check of the freshly built model, taken by CALLING the
`:oop` build product on host vectors:

```julia
du = fi(u0i, pi, TSAMP)
sha = bytes2hex(sha256(reinterpret(UInt8, vec(du))))
```

An `:oop` build product is now the compiled IR a backend lowers, and calling it
raises `E_TREEWALK_OOP_NOT_EVALUABLE` (EarthSciAST `src/tree_walk/oop.jl`), so
each of these aborts at its FIRST rung. They cannot be re-pointed at
`rx_host_eval` either, and that is the reason they are deleted rather than
fixed: an `rx_host_eval` check costs one `Reactant.@compile` of the program
being checked, in the same process, and every one of these probes exists to
measure `build_evaluator`'s wall time, resident size, GC and page faults —
which a Reactant load and an XLA compile in that process are exactly what
destroy. There is no version of the check that leaves the measurement intact.

`git show <merge>^2:<path>` prints any of them. What they are the provenance
for:

| Deleted | Backed |
| --- | --- |
| `build_res_window.jl`, `build_res_window.sbatch`, `build_res_window_split.sbatch` | **BUILD_SCALING.md** — the native-resolution window ladder: the decode/build/compile decomposition at `BRW_GRID=25x13x72` (slurm 10404098), the ~0.0058 s/cell residual in `build_evaluator`, and the `18480c62f94b7dc5` right-hand-side hash the synthetic ladders are checked against. |
| `forcing_size_ladder.jl`, `forcing_size_ladder.sbatch` | **BUILD_SCALING.md** §"isolate array size from work" and **SCALE_025.md**'s `e` exponents (slurm 10404593): the controlled ladder that embeds ONE 2x2.5 decode into synthetic native arrays up to the 0.25x0.3125 shape, holding cells, geometry, values and gather count fixed so that only the size of the array being gathered from moves. Its `loop` stage was also the only remaining named call of `EarthSciAST.rhs_with_buffers` in the tree. |
| `aktbl_ladder.jl`, `aktbl_ladder.sbatch` | **BUILD_SCALING.md**'s NLEV-only ladder (slurm 10423120 / 10423122), one process per grid, part 1. |
| `axisb_ab.jl`, `axisb_ab.sbatch` | **BUILD_SCALING.md**'s axis-B A/B: two EarthSciAST checkouts built side by side, compared on the cascade tally, the spine templates and `sha256(du)` at the build's own base point. |
| `axisb_evaltime.jl` | **BUILD_SCALING.md**'s "a build-time win that costs right-hand-side time is a bad trade" control. Its question does not survive at all: it timed 200 reps of the HOST tree-walk evaluation, which is the surface EarthSciAST deleted. |
| `build_prof_delta.jl`, `build_prof_delta.sbatch` | **BUILD_SCALING.md** B3 — the `@profile` attribution of the unexplained per-cell build time. |

`build_driver_decomp.jl` / `.sbatch` from the same investigation is KEPT: it
times `build_evaluator` and never evaluates the product, so it runs unchanged.

## Known residue: two older probes with the same dead line

`tools/diag/native_window_probe.jl` (two call sites) and
`tools/diag/res_switch_smoke.jl` call the `:oop` product on the host in exactly
the way described above, and were missed by the cleanup in `6f592ca`. They abort
at that line on the direct lane. They are recorded here rather than deleted
because they predate this merge and the same trade applies to them: their
measurement is a build measurement, and the only available check costs a
compile.
