# Does EarthSciAST's direct StableHLO emitter compute what ReSEACT's traced right-hand side computes?

Measurement, 2026-09-14, branch `direct-rhs`. Supersedes sections 1, 2, 3 and 4
of [DIRECT_RHS_CENSUS.md](DIRECT_RHS_CENSUS.md), which was measured on a
degenerate operator split (see "The split was not a split" below) and against an
emitter that still had three coverage gaps.

## The answer

**Yes, to double-precision rounding, on both halves.** Both operator-split
halves of ReSEACT emit and compile with ZERO refusals, and the direct program
agrees with the traced program ReSEACT runs on today to a worst relative
difference of 6.5e-14 (transport) and 7.4e-16 (chemistry) at three times across a
day, with no non-finite values anywhere. Both halves land inside the
**algebraic** tolerance class (1e-13), which is tighter than the transcendental
class the earlier report predicted for them.

Three things had to be true first, and each was a separate defect:

1. The emitter had one refusal left, on a prefix scan. It was a false positive.
2. The operator split was not splitting; both "halves" were the whole model.
3. The traced lane's failure on Reactant 0.2.285 — recorded as a version
   regression — was a consequence of (2), not of the version.

## 1. The last refusal was a false positive: `Transport3D.Mz[i,j,9]`

The refusal, on the transport half at 6x6x8:

```
E_DIRECT_EMIT_UNSUPPORTED: … cannot lower a read-before-write (from a prefix scan
at materialization level 5): flat slot 4705 is read before anything in this
emission wrote it.
```

`tools/diag/direct_rhs_slot_probe.jl` names it and classifies it without
compiling anything: it builds both halves, reflects on the `:oop` closure's
`mat_levels` / `acc_plans` / `scan_folds`, and replays `_de_emit!`'s walk order
over a "written" vector.

| | |
| --- | --- |
| slot | **`Transport3D.Mz[1,1,9]`** (flat 4705), and its 35 siblings `Mz[i,j,9]`, flat 4705..4740 |
| what it is | the TOP INTERFACE NODE of the diagnosed vertical air-mass flux, one per grid column |
| where | materialization level 5 of 5, the level that materializes `Mz`; the fold is `len=9` (NLEV+1 nodes) x `lanes=36` (NLON x NLAT columns), `⊕ = +`, EXCLUSIVE |
| position | position **9 of 9** in every one of the 36 lanes — and no other position |
| written by | nothing in the fill. The only writer is the fold itself, at the same step it reads |
| verdict | **(A), a structural zero** |

**Why it is (A) and not (B).** Nothing is mis-ordered. `Mz` is a STAGGERED prefix
reduction — `Mz[ke] = -Σ_{k < ke} …` with the output on the level NODES and the
terms on the level CENTRES — and `_scan_term_iters` (EarthSciAST
`src/tree_walk/build.jl`) admits that shape and documents that it leaves the last
output node UNCOVERED by the term build, because that node has no centre of its
own. The fold then writes every node and reads each slot only into an
accumulator that, at the last step, it discards. So the value read there cannot
reach any output — on host, under either interpreter, or in the emitted program.
The level scheduler is correct and needed no change.

The chemistry half has **zero** read-before-write slots.

```
transport  read-before-write=36  never=0  same-section=36  LATER-section=0
chemistry  read-before-write=0   never=0  same-section=0   LATER-section=0
```

**The fix** (EarthSciAST, `ext/reactant_direct/`) gives the emitter the
distinction it was missing instead of a blanket refusal. `_de_plan_writes!`
computes, before the walk and from the same plan data the walk consumes, which
SECTION of the emission first writes each slot — one section per materialization
level, then one for the state equations. A read of an unwritten slot is then:

* **a zero the section is entitled to** — nothing writes the slot, or only this
  same section does and it has not reached the write yet. Both evaluators run
  these units in this order over an extended vector that starts at zero, so 0.0
  is what the interpreter reads too, and the emitter now emits it (one zero
  constant per run of such slots, the same piece `_de_assemble` already emitted
  for an unwritten output run);
* **an ordering violation** — a LATER section writes it, i.e. a fill level reads
  a level above it. That is still a hard refusal, and the message now names the
  section that writes it.

## 2. The split was not a split

EarthSciAST's `aggregate` -> `faq` node rename (28f21bd56, 2026-09-12) was never
propagated to EarthSciASTSplitter, whose lifted-derivative recogniser matched
`"aggregate"`. EarthSciAST normalizes that alias away on load, so the recogniser
returned `nothing` for every state equation and `split_equations` copied the
WHOLE tendency into BOTH parts. Nothing downstream noticed: two identical halves
build, compile, and run.

Two fixes, both needed:

* **EarthSciASTSplitter** — `86d4d2661` on branch `fix/faq-node-tag` (worktree
  `/scratch/$USER/jacbisect/wt/splitter-fix`). Not this repository's to commit;
  both environments below dev-link it.
* **This repository** — `contracts_space` in
  `prototypes/reseact_3d_chem/split_common.jl` matched `"aggregate"` too, so the
  detector that keeps a spatially contracting aggregate (the GEOS-Chem
  boundary-layer column integral) out of the chemistry half never fired. Fixed
  in `1a2be79`.

`tools/diag/direct_rhs_slot_probe.jl` now checks the split BEFORE reporting
anything, because a degenerate split is invisible in every downstream number:

| | broken split | fixed split |
| --- | ---: | ---: |
| part 1 right-hand side | 1,413,830 B | 1,413,830 B |
| part 2 right-hand side | (identical to part 1) | 915,203 B |
| `SuperFast.NO` rhs, part 1 | — | 44,542 B (`index`=495) |
| `SuperFast.NO` rhs, part 2 | — | **1,849 B** (`index`=20, `ifelse`=0, `abs`=0, `min`=0) |
| transport `n_total` | 65,966 | 12,326 |
| chemistry `n_total` | 65,966 | 57,708 |
| transport materialization levels | 11 | 5 |
| chemistry materialization levels | 11 | 11 |

Part 2's `SuperFast.NO` tendency is the mechanism term alone — 1,849 bytes with
no PPM-limiter flood — and part 1 carries the advection stencil for every
species. Under the broken split the two parts' right-hand sides were byte-for-byte
identical, which is why the earlier census saw the same refusal counts, the same
gap profile and nearly the same module size in both halves.

## 3. The traced lane's 0.2.285 failure was the split, not the version

DIRECT_RHS_CENSUS.md section 4a recorded `Scalar indexing is disallowed` out of
`_oop_ssa_resolve` (EarthSciAST `src/tree_walk/oop.jl:2540`, a `reduce` over a
`Vector{Any}`) as a Reactant 0.2.280 -> 0.2.285 regression in the `mapreduce`
overlay. It is not: the same failure reproduces on **0.2.280** with EarthSciAST
`main` when the split is degenerate, and it does NOT occur on either version once
the split is fixed. Both traced halves compiled and ran cleanly below. Nothing
about `mapreduce` needs to change; what the unsplit half produced was an SSA
class-to-class resolve over a heterogeneous list large enough to reach Reactant's
unwrapping path.

## 4. Cost, direct lane, zero refusals

Grid 6x6x8 (the drivers' demonstration preset), GEOS-FP 4x5 row: 3,744 states,
162 runtime parameters, 15 discrete forcing providers, 16 live forcing buffers
per half. Reactant 0.2.285, `ESS_OOP_SSA=1` (production's setting, read at BUILD
time), the adjoint driver's production compile options
(`sync = true`, `xla_cpu_prefer_vector_width = 128`,
`excluded_passes = ["dynamic_update_to_concat", "sub_const_prop"]`). Per-call
median of 20 calls after warm-up; module with `optimize = true`.

| half | `@compile` wall | per-call median | optimized module |
| --- | ---: | ---: | ---: |
| transport, direct | 343.5 s | 3.3 ms (min 2.4) | 18,064 lines / 18,060 ops |
| chemistry, direct | 191.5 s | 0.8 ms (min 0.7) | 2,694 lines / 2,690 ops |

Repeated inside the agreement job on the same options: transport 344.7 s /
3.0 ms / 18,065 lines, chemistry 352.7 s / 0.9 ms / 2,695 lines. The line counts
differ by one because the agreement probe prints the module for a different `t`.

For reference, the traced halves in the production environment (Reactant 0.2.280,
EarthSciAST `main`) compiled in 220.7 s (transport) and 261.2 s (chemistry). Those
are not a like-for-like cost comparison — different Reactant, different
EarthSciAST — and no ratio is claimed from them.

Build cost, for scale: 193-226 s for the two `:oop` halves at this grid.

## 5. Agreement

The two lanes cannot meet in one process: ReSEACT's production environment pins
Reactant 0.2.280 and the emitter was built against 0.2.285. The comparison
therefore goes through FILES, which is stronger than building the same point
twice — every input both programs see comes out of one file, so a difference can
only be the program.

`tools/diag/direct_rhs_agree.jl` in `RESEACT_AGREE_MODE=traced` writes the state
vector, the parameter values, the three times, and the CONTENTS of all 16 live
forcing buffers at each time, plus its own `du`; in `RESEACT_AGREE_MODE=direct` it reads those
exact inputs back, asserts that the state layout, the parameter keys and the
buffer order match, and writes its own `du`; `=compare` loads neither Reactant nor
the model.

Base point: the harnesses' jittered state (`RESEACT_ADJ_UJITTER=1e-1`, seed
31337) with `Transport3D.m` seeded from the real GEOS-FP surface pressure — never
the uniform default, which sits on the PPM limiters' switching surfaces. Times
5400, 34200 and 63000 s, with the forcing re-sampled at each.

Relative differences are taken against the traced value with an absolute floor of
1e-14; a component whose traced value is ~0 has no meaningful relative error.

| half | t (s) | max abs | max rel | max abs traced | non-finite | worst variable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| transport | 5400 | 6.505e-19 | 2.290e-14 | 2.687e-02 | 0 | `SuperFast.HO2[5,5,6]` |
| transport | 34200 | 8.674e-19 | 6.507e-14 | 2.048e-02 | 0 | `SuperFast.OH[1,4,8]` |
| transport | 63000 | 8.674e-19 | 5.625e-14 | 3.107e-02 | 0 | `SuperFast.HO2[1,4,8]` |
| chemistry | 5400 | 1.084e-19 | 7.377e-16 | 1.335e-03 | 0 | `SuperFast.OH[1,2,1]` |
| chemistry | 34200 | 1.084e-19 | 3.919e-16 | 7.517e-04 | 0 | `SuperFast.HNO3[4,1,1]` |
| chemistry | 63000 | 1.084e-19 | 7.092e-16 | 1.338e-03 | 0 | `SuperFast.HNO3[4,3,1]` |

Worst relative difference per variable block, over all three times:

| block | transport | chemistry |
| --- | ---: | ---: |
| `SuperFast.OH` | 6.507e-14 | 7.377e-16 |
| `SuperFast.HO2` | 5.625e-14 | 0 |
| `SuperFast.CH3O2` | 4.995e-14 | 0 |
| `SuperFast.O3` | 3.720e-14 | 4.357e-16 |
| `SuperFast.NO2` | 1.651e-14 | 3.613e-16 |
| `SuperFast.HNO3` | 1.591e-14 | 7.092e-16 |
| `SuperFast.CH2O` | 1.070e-14 | 2.171e-16 |
| `SuperFast.CH3OOH` | 7.966e-15 | 2.208e-16 |
| `SuperFast.CO` | 5.267e-15 | 0 |
| `SuperFast.H2O2` | 4.986e-15 | 0 |
| `SuperFast.NO` | 3.836e-15 | 0 |
| `SuperFast.ISOP` | 1.972e-17 | 4.930e-18 |
| `Transport3D.m` | 0 | 0 |

**Where each half lands.** Both inside the ALGEBRAIC class (1e-13 relative):
transport at 6.5e-14, chemistry at 7.4e-16. Nothing exceeds any class, so there
is no variable to single out. Two observations:

* The tolerance classes the earlier report predicted (transcendental, 1e-12, for
  both halves) were too loose. The direct emitter reproduces the interpreter's
  own fold order below the cut where a long `⊕`-fold becomes one
  `stablehlo.reduce`, and the transcendentals that do appear agree far better
  than their class allows.
* The absolute differences are 1e-19 against tendencies of 1e-2 to 1e-3, i.e.
  the floor. The transport half's larger RELATIVE numbers are the radical
  species (`OH`, `HO2`, `CH3O2`) whose transported tendency is a difference of
  nearly equal fluxes: 6.5e-14 relative on a 5.7e-26 absolute difference is
  cancellation in the reference, not error in the program.
* `Transport3D.m` and, in the chemistry half, six species agree BIT-FOR-BIT.

## 6. Reproducing

```bash
# Environments. Both dev-link the EarthSciASTSplitter faq-node-tag fix; the
# traced one is otherwise production's (Reactant 0.2.280, EarthSciAST main), the
# direct one is the emitter's (Reactant 0.2.285, EarthSciAST oop-retire).
TRACED=/scratch/$USER/oopretire-env-280traced
DIRECT=/scratch/$USER/oopretire-env-faq2

# A PRIVATE depot first, always: the shared depot's precompile lock otherwise
# hangs Pkg, and two of these at once block each other in `using Reactant`.
export JULIA_DEPOT_PATH=/scratch/$USER/oopretire-depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia

# The slot probe: no Reactant, no MLIR, the build is the only cost (~190 s).
RESEACT_RXENV=$DIRECT ESS_OOP_SSA=1 \
  timeout -k 15 3000 julia --project=$DIRECT --heap-size-hint=12G \
  tools/diag/direct_rhs_slot_probe.jl > /scratch/$USER/slotprobe.log 2>&1

# The census, direct lane only (the traced oracle is skipped: its own failure
# used to cost the whole run).
RESEACT_RXENV=$DIRECT ESS_OOP_SSA=1 RESEACT_CENSUS_JAC=0 \
RESEACT_CENSUS_TRACED=0 RESEACT_CENSUS_STAGES=emit,cost \
  timeout -k 15 7200 julia --project=$DIRECT --heap-size-hint=18G \
  tools/diag/direct_rhs_census.jl > /scratch/$USER/census.log 2>&1

# The agreement: four compiling runs, ONE PROCESS EACH, in one allocation.
sbatch tools/diag/direct_rhs_agree.sbatch
```

The slot probe names materialized-observed slots only if it is given a directory
of `extvarmap-<n_total>.json` dumps (`RESEACT_SLOT_NAMES`); those come from a
TEMPORARY nine-line patch to EarthSciAST's `_build_compile_evaluator` that writes
`layout.var_map_ext`, saved at
`/scratch/$USER/oopretire-logs/extvarmap-patch.diff` and reverted before
committing. Without it the probe prints flat slot numbers and everything else is
unchanged — which is what lets the committed probe run against a clean worktree.
The maps are PER HALF (each half materializes different observeds), and the probe
selects by `n_total`: naming a transport slot out of the chemistry half's map is
a mistake this indirection exists to prevent, and one this session made once.

## 7. Where the runs are

| file | what |
| --- | --- |
| `/scratch/$USER/oopretire-logs/slotprobe4-fixedsplit.log` | the slot identification, fixed split, per-half names |
| `/scratch/$USER/oopretire-logs/census4-285-ssa1-fixedsplit.log` | the census: both halves emit, compile, and cost |
| `/scratch/$USER/oopretire-logs/agree-10549691.log` | the agreement job (Slurm 10549691) |
| `/scratch/$USER/oopretire-logs/agree/` | `inputs.bin`, `traced_*.bin`, `direct_*.bin` |
| `/scratch/$USER/oopretire-logs/slotprobe2-ssa0.log` | the same probe on the BROKEN split, for the contrast |

## 8. What is still open

* **The `jac = :sym` band model** (DIRECT_RHS_CENSUS.md section 5) is untouched
  here. EarthSciASTDiff still spells `"aggregate"` at 13 sites, so
  `prepare_jacobian` returns an empty band model against any faq-normalizing
  EarthSciAST — and the production adjoint driver's `validate_plan` passes
  vacuously on zero entries. That is the same rename defect as section 2, in a
  third repository, and it is the one that silently affects a production run.
* **CONUS.** `tools/diag/direct_rhs_census_conus.sbatch` is the reproducer and
  needs no editing, but it must be pointed at the two fixed environments before
  it is worth an allocation.
* **Differentiation.** Reverse mode through the direct lane is now answerable
  (`RESEACT_CENSUS_STAGES=emit,grad`) and was not measured here.
