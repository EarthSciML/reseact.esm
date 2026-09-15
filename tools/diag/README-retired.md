# Probes retired with the traced right-hand side

Every compiled program in this repository is emitted by EarthSciAST's direct
StableHLO emitter (`tools/rx_rhs.jl`). The probes listed here named a surface
that no longer exists — `EarthSciAST.rhs_with_buffers`, the
`oop_intern_stats` / `oop_gvn_stats` / `oop_ssa_stats` counters, the
`ESS_OOP_SSA` / `ESS_OOP_GVN` / `ESS_OOP_INTERN` / `ESS_OOP_SHIFT_SLICE` build
flags, `RESEACT_RHS`, or `tools/reactant_handoff/rx_native_patch.jl` — so they
cannot run and are deleted in `6f592ca`.

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
