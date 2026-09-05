# Upstream issues filed from this repo — status 2026-09-05

Six issues were filed against the EnzymeAD stack on 2026-08-24, all found while
building the ReSEACT adjoint. Two are already fixed upstream; four are open with
no PR against them. This file is the status record; the write-ups that were
pasted into them live in `tools/diag/UPSTREAM_reverse_over_forward.md`, and every
one of them has a reproducer in `tools/diag/`.

| | issue | filed to | state (2026-09-05) | in-repo reproducer |
|---|---|---|---|---|
| Bug B | [Enzyme-JAX #2938](https://github.com/EnzymeAD/Enzyme-JAX/issues/2938) — `concat_broadcast_slice` merges `concatenate` operands along the wrong axis | Enzyme-JAX | **CLOSED**, fixed | `tools/diag/rof_concat_repro.jl` |
| Bug A | [Enzyme #3169](https://github.com/EnzymeAD/Enzyme/issues/3169) — segfault instead of a diagnostic when `CreateReverseDiff` fails in `AutoDiffCallRev` | Enzyme | **CLOSED**, fixed | `tools/diag/rof_repro.jl` |
| Bug C | [Reactant.jl #3217](https://github.com/EnzymeAD/Reactant.jl/issues/3217) — batched forward mode fails to lower by either route | Reactant.jl | open, no PR | `tools/diag/rof_batchfwd.jl` |
| — | [Enzyme-JAX #2939](https://github.com/EnzymeAD/Enzyme-JAX/issues/2939) — no autodiff rules for `stablehlo.case` | Enzyme-JAX | open, labelled *good first issue*, cross-linked to the missing-HLO-derivatives tracker (#88) | `tools/diag/mwe_case_reverse.jl` |
| — | [Reactant.jl #3215](https://github.com/EnzymeAD/Reactant.jl/issues/3215) — `Ops.constant(::Number)` is not memoized while `Ops.constant(::DenseArray)` is | Reactant.jl | open, no PR | `tools/diag/reactant_emission_repro.jl` (A) |
| — | [Reactant.jl #3216](https://github.com/EnzymeAD/Reactant.jl/issues/3216) — `broadcast_to_size` emits `broadcast_in_dim` for already-matching shapes; `materialize` leaves a dead zero fill | Reactant.jl | open, no PR | `tools/diag/reactant_emission_repro.jl` (B) |

## The two fixes, and the version that carries them

Both landed on 2026-08-25, one day after filing:

* **#2938** — [Enzyme-JAX PR #2953](https://github.com/EnzymeAD/Enzyme-JAX/pull/2953)
  ("make sure that slice is along concat dim in BroadcastReshapeSlice", Pangoraw,
  merge `4e557062`). This is the fix the issue asked for: `mergeConcatSlicedElems`
  now checks that the sliced dimension maps onto `concatDim`.
* **#3169** — [Enzyme PR #3172](https://github.com/EnzymeAD/Enzyme/pull/3172)
  (merge `fc6bb335`), null-checks in both `callReverseHandler` and
  `callForwardHandler`, with tests. **It is the one-line diagnostic that was
  asked for, not a cure**: reverse-over-forward on the real RHS will now *name
  the callee it could not differentiate* instead of dying at
  `mlir::Operation::getAttr`. The WALL of `DIFFERENTIABILITY_PLAN.md` §Phase 3
  stands; it is now diagnosable.

The chain from those merges to a Julia version, verified commit by commit:

| link | commit | date |
|---|---|---|
| Enzyme-JAX pins `ENZYME_COMMIT = fc6bb335…` (= PR #3172's merge) | `aea4857f` | 2026-08-25 |
| PR #2953's merge `4e557062` is an ancestor of `aea4857f` | — | verified with `git merge-base --is-ancestor` |
| Reactant pins `ENZYMEXLA_COMMIT = aea4857f…` | `918d443c` | 2026-08-25 |
| first Reactant tag containing `918d443c` | **v0.2.283** | Project.toml pins `Reactant_jll = 0.0.407` |

So **Reactant ≥ v0.2.283 (Reactant_jll 0.0.407) is the first release carrying
both fixes**. This repo's Manifest pins **v0.2.280 / jll 0.0.405** — neither fix
is in the environment the 2026-09-05 results ran on. The jll↔WORKSPACE mapping is
inferred from the commit that bumped both in the same series, not from a build
hash; confirm against the jll before treating a re-run as a regression test.

## What this changes here

1. **The one experiment worth running.** Bump a probe environment to Reactant
   ≥ 0.2.283 and re-run the reduced reverse-over-forward case — `jacrev` at
   `NCOL=1` (`tools/diag/rof_repro.jl`, or `tools/rx_adjoint_check.jl` with
   `jac=:ad`). It previously took ~595 s to build at 6×6×8 and ended in
   `EXIT=139`. With #3172 in, the same run should print the name of the callee
   `CreateReverseDiff` failed on. That name is the whole content of the real
   upstream bug, which Bug A was only ever a placeholder for.
2. **A workaround can be retired.** `excluded_passes=["concat_broadcast_slice"]`
   — reachable as `RESEACT_ADJ_EXCL` in `tools/rx_adjoint_check.jl` — is
   unnecessary on ≥ 0.2.283. This is *not* the default
   `RESEACT_EXCLUDED_PASSES=dynamic_update_to_concat,sub_const_prop`, which is a
   throughput exclusion (§6) and is unaffected.
3. **Do not move the pin on the strength of these two fixes alone.** Nothing on
   the critical path is blocked by either: `jac=:sym` is the default, so nothing
   here computes a reverse-over-forward Jacobian, and the concat miscompile never
   fired in a production configuration. Against that, 0.2.274–0.2.284 all show
   the checkpointing bug, the XLA:CPU race workaround is required on every one of
   them, and there is a fresh upstream report that reverse mode through a
   `@trace mincut=true` loop got ~30× slower on **GPU** with Reactant ≥ 0.2.281
   ([Enzyme-JAX #2999](https://github.com/EnzymeAD/Enzyme-JAX/issues/2999)) —
   almost certainly irrelevant to a CPU run with host-lifted loops, but exactly
   the shape of regression that argues for clearing the 6×6×8 gate before the
   CONUS baseline moves. Keep 0.2.280 as the record for the published numbers.
4. **The four open issues cost nothing today.** #3217 keeps `ad_block_jac`
   serial, which only matters if `jac=:ad` returns; #2939 stays a rule about our
   own code (only an explicit `Reactant.Ops.case` emits `stablehlo.case`, so
   keeping it out of differentiated code is entirely in our hands); #3215/#3216
   are trace-time and module-size taxes, on a path where EarthSciAST's own
   emitter fixes already took the order-of-magnitude win. None of them is
   worth waiting on. If #3215/#3216 are wanted, they are small enough that the
   cheapest route is to send the patches upstream ourselves rather than to
   track them.

**Not read here:** the comment threads. #3169 (1 comment), #3215 (1), #3216 (3)
carry discussion this session could not fetch — the GitHub API is scoped to this
repository and the issue pages render comments client-side. Everything above is
from issue/PR state, linked PRs, and the upstream git history, all of which are
readable. Read those three threads before acting on #3215/#3216.
