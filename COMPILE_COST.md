# Why the direct StableHLO lane compiled slower than the traced one, and what fixed it

Measurement, 2026-09-14/15, branch `direct-rhs`, grid 6x6x8, Reactant 0.2.285,
EarthSciAST `oop-retire`, `ESS_OOP_SSA=1`. Companion to
[AGREEMENT.md](AGREEMENT.md), which establishes that the two lanes compute the
same thing; this one is about what they COST, which is what the port of the
production entry points onto `RESEACT_RHS=direct` turned up.

## The answer

The direct emitter turned every read of the extended state into **one slice per
contiguous run plus a concatenate**. On ReSEACT's transport half those runs are
two to four elements long, so the four-stage transport step reached the pass
pipeline as **4.1 MILLION ops, 4,078,400 of them `stablehlo.slice`** — against
the traced emitter's 113,489 ops and 2,712 slices. The pipeline then spent
399 s grinding that down to 66,830 ops, and the reverse-mode program built from
it did not finish compiling in **five hours**.

The decision that produced it is three clauses in `_de_emit_runs`
(`ext/reactant_direct/values.jl`), and each of the three independently blocked
the gather that should have been emitted instead. Replacing them with the cost
the code is actually deciding — **`ESM_DIRECT_EMIT_READ=gather`, now the
default** — takes the transport step's pre-pass module from 4,104,019 ops to
88,983, its `@compile` from 894 s to 207 s against the traced lane's 170 s, its
optimized module from 18,060 ops to 6,510, and its per-call median from 3.3 ms
to **1.9 ms**. It is faster to compile AND faster to run, and it changes no
number.

## 1. The measurement

`tools/diag/direct_rhs_compile_cost.jl`, one program per process. `@compile`
rolls three costs into one number and the probe separates them by running the
same program three times in one process, in this order:

| stage | what it runs | what the difference means |
| --- | --- | --- |
| `trace` | `@code_hlo`, `optimization_passes = false`, COLD | Julia's own JIT of the tracing stack, plus emission |
| `opt` | `@code_hlo`, the production pipeline, warm | emission plus Enzyme-JAX / StableHLO passes |
| `raw2` | `@code_hlo`, `optimization_passes = false`, warm | **emission alone** |

so `raw2` is emission, `opt - raw2` is the pass pipeline, and `trace - raw2` is
Julia JIT. Compile options are the adjoint driver's production set in every run
(`sync = true`, `xla_cpu_prefer_vector_width = 128`,
`excluded_passes = dynamic_update_to_concat,sub_const_prop`).

### `ssp_step` — the transport half, four SSPRK43 stages

| | traced | direct, `runs` | direct, `gather` |
| --- | ---: | ---: | ---: |
| **emission** (`raw2`) | 10.0 s | 116.0 s | **3.6 s** |
| **pass pipeline** (`opt - raw2`) | 12.0 s | 399.1 s | **34.1 s** |
| Julia JIT (`trace - raw2`) | 93.7 s | 60.6 s | 73.2 s |
| pre-pass ops | 113,489 | **4,104,019** | 88,983 |
| pre-pass `slice` | 2,712 | **4,078,400** | 58,620 |
| pre-pass `gather` | 576 | 0 | 4,728 |
| pre-pass `func.func` | 121 | 1 | 1 |
| post-pass ops | 11,151 | 66,830 | 22,745 |
| post-pass `slice` | 878 | 55,142 | 13,200 |
| post-pass `gather` | 392 | 0 | 310 |

The ARITHMETIC is the same computation by three routes — `multiply` is 3,071 and
`select` 828 in every post-pass module. The difference is entirely data
movement, and it is one op kind.

### `ssp_vjp` — the same half under Enzyme reverse mode

| | traced | direct, `runs` | direct, `gather` |
| --- | ---: | ---: | ---: |
| emission (`raw2`) | 11.2 s | 46.6 s | **3.4 s** |
| pass pipeline (`opt - raw2`) | 39.3 s | — | — |
| pre-pass ops | 113,744 | **1,849,190** | 89,238 |
| pre-pass `slice` | 2,712 | **1,821,224** | 58,620 |
| pre-pass `gather` | 576 | 0 | 4,728 |
| post-pass ops | 31,888 | — | — |

EMISSION IS NOT THE PROBLEM in any of the three — 46.6 s at its worst against
11.2 s. What the emitter hands over is: under `runs`, a module 16x larger than
the traced lane's and **98.5% slices**. The adjoint driver's `@compile ssp_vjp`
on it ran for more than SIX HOURS without finishing
(`/scratch/$USER/oopretire-logs/port/adj6-direct-local.log`), against 115.4 s in
the traced lane.

**The fix shrinks what Enzyme is given by 20.7x, to smaller than the traced
lane's own module, and the reverse-mode compile is still the wall.** `gather`
made `ssp_step` comparable with the traced lane (207 s against 170 s) and it did
not make `ssp_vjp` comparable. Emission is 3.4 s; the probe's `opt` stage — the
MLIR pass pipeline alone, before XLA:CPU codegen is even reached — was killed by
its own 3-hour cap.

### Which pass, by name

`perf record -F 199 -g` on the live compile, 60 s, 11,937 samples. It is not a
mystery and it is not XLA:

| share | symbol |
| ---: | --- |
| 13.1% | `enzyme::CheckedOpRewritePattern<stablehlo::SliceOp, SliceElementwise>::matchAndRewrite` |
| 12.7% | `mlir::OperationEquivalence::isEquivalentTo` (CSE, two frames) |
| 1.6% | `CSE<stablehlo::SliceOp>::matchAndRewriteImpl` |
| 2.8% | `StaticSlice::get` / `StaticSlice::StaticSlice` |
| 2.7% | `SliceOp::getStrides` / `SliceOp::getStartIndices` |

with the remainder dominated by the generic accessors those patterns call
(`DenseArrayAttrImpl<long>`, `RankedTensorType::getShape`, `hasStaticShape`,
`SmallVectorImpl<long>::operator=`). Every frame in the profile is the greedy
rewrite driver working on slices. The two registered pattern names are
`slice_elementwise` (push a slice through an elementwise op, which CREATES two
more slices per rewrite) and `cse_slice`, both reachable from the driver's
existing `RESEACT_EXCLUDED_PASSES`.

**Excluding them is not the fix.** Tried, with
`RESEACT_EXCLUDED_PASSES=dynamic_update_to_concat,sub_const_prop,slice_elementwise,cse_slice`:
the pipeline was OOM-killed thirteen minutes in. That is consistent rather than
surprising — `cse_slice` is also what KEEPS the slice set from growing, so
removing it leaves the other patterns multiplying an un-deduplicated 58,620.
The cost is the slice POPULATION, not one pattern's implementation.

The cost is the slice POPULATION. What that paragraph then GUESSED — that
getting further needs a materialization layout in which a stencil neighbour read
is one contiguous span, i.e. `src/tree_walk/oop_merge.jl` rather than the read
lowering — is wrong, and section 5 is the measurement that says so. Three
quarters of the 58,620 came from ONE clause in the read lowering after all.

### End to end, the adjoint driver's four compiles at 6x6x8

| program | traced | direct, `runs` | direct, `gather` |
| --- | ---: | ---: | ---: |
| `ssp_step` | 170.1 s | 893.7 s | **206.7 s** |
| `ros_step` | 91.0 s | 68.4 s | 69.4 s |
| `ssp_vjp` | 115.4 s | > 6 h, unfinished | (see the driver header) |
| `ros_vjp` | 72.5 s | — | |

**The chemistry half was never the problem** — it is cheaper in the direct lane
in both read forms, and it emits ONE gather, because its reads are not
shattered. The transport half is, and reverse mode over the transport half is
where a constant factor became a wall.

### And the runtime, which decides the trade

`tools/diag/direct_rhs_census.jl`, `RESEACT_CENSUS_STAGES=emit,cost`, per-call
median of 20 calls after warm-up, optimized module:

| half | read form | `@compile` | per-call median | optimized module |
| --- | --- | ---: | ---: | ---: |
| transport | `runs` | 343.5 s | 3.3 ms (min 2.4) | 18,060 ops |
| transport | `gather` | **239.0 s** | **1.9 ms (min 1.7)** | **6,510 ops** |
| chemistry | `runs` | 191.5 s | 0.8 ms (min 0.7) | 2,690 ops |
| chemistry | `gather` | 190.6 s | 0.9 ms (min 0.8) | 2,690 ops |

(the `runs` row is AGREEMENT.md section 4's measurement, same options, same
grid.) So this is not a compile-time win paid for at runtime: the transport
half's optimized module is 2.8x smaller and its call is 1.7x faster.

## 2. Where in the emitter it came from

`_de_emit_runs`, `ext/reactant_direct/values.jl`. Every read of the extended
state, of a live forcing buffer, or of a CSR reduce's body buffer goes through
it. It decomposes the read's positions into (same producer, arithmetic-position)
runs and emits one slice per run plus one concatenate, with a single escape to a
real `stablehlo.gather`:

```julia
if nzero == 0 && length(runs) > 1 && n > 8 && length(runs) > n ÷ 2 &&
   all((r[1]::_DEVal).v == (runs[1][1]::_DEVal).v for r in runs)
```

Three clauses, each of which alone is enough to block it on this model:

1. **`length(runs) > n ÷ 2`** defines "shattered" as an average run shorter than
   TWO. ReSEACT's PPM stencil reads average two to four, so a read of 1,104
   positions decomposing into 400 runs is not shattered by this test and costs
   400 slices. Measured on the emitted module: 84% of the direct lane's 55,117
   post-pass slices are five elements or narrower and 46% are exactly two, while
   the traced lane emits none narrower than six.
2. **`all(... == runs[1]...)`** requires every run to lie in ONE producer value.
   Of the 506 concatenates in the emitted step, **241 are single-producer and
   255 span exactly two**; ten span more. So half the reads were disqualified by
   a second producer alone.
3. **`nzero == 0`** disqualifies any read containing a structural zero.

## 3. The fix

`ESM_DIRECT_EMIT_READ`, default `gather`; `runs` restores the previous shape
exactly, as the negative control.

* **Decide on the average run length.** One gather costs one op plus an O(n) i64
  index constant; the slice path costs one op per run plus a concatenate. The
  gather is emitted when `npieces >= 8 && npieces * 4 > n` — the average run is
  shorter than four. Long affine runs, which are what the slice path exists for
  and what the traced lane's own 36- and 216-wide slices are, stay where they
  were.
* **Gather across producers.** The DISTINCT producer values are concatenated
  once and the gather reads that, at remapped positions. The concatenation is
  cached on the emission context keyed by the producer set, so a stencil's
  recurring read set pays for it once; and it is declined when the base would be
  more than 8x the read, which is the case where the concatenate would be the
  new cost.
* **Structural zeros are allowed in.** A one-element zero is appended to the
  base and every zero lane points at it — a gather may read one position many
  times — which removes the third clause without a special case.

Numerically inert, and the emitter's own test file pins that: `test/
reactant_direct_emit_test.jl`, "the read cost model", emits the 343-cell
3-axis-stencil fixture under both settings, checks both against the interpreter
at `rtol = 1e-12`, and pins the decision function's thresholds directly.

## 4. Two traps this measurement walked into

**`@code_hlo compile_options = X optimize = false` does not turn the passes
off.** `__compile_options_from_kwargs` (Reactant `src/CompileOptions.jl`) returns
`X` unchanged the moment `compile_options` is present, so `optimize` is dropped
silently — and `optimize` is only ever a spelling of the `optimization_passes`
FIELD anyway. The first run of this probe reported "optimize=false" and
"optimize=true" censuses that were identical to the op, which is what gave it
away; a first pass at this document quoted 66,731 ops as the direct lane's
PRE-pass module when it was the post-pass one, understating the problem by 61x.
The probe now builds two `CompileOptions` differing in that one field.

**The calendar broadcast helpers are not the mechanism.** The
`floor_broadcast_scalar` x116 / `_cal_month_broadcast_scalar` x20 helpers in the
direct run's heartbeat line come from the CHEMISTRY half's calendar chain, which
the direct emitter lowers through the registry kernel and Reactant's broadcast
overlay on purpose (`ext/reactant_direct/interp.jl`; it is O(1) in the grid).
The transport step's whole trace mints three helpers, both `ssp_step` modules
are ONE `func.func` after the pipeline, and the half where all the helpers live
is the half that compiles FASTER in the direct lane.

## 5. Which STAGE, and the second emitter clause

`perf` named the pattern; it did not name the stage, because `opt` is one number
over a dozen passes. `RESEACT_CC_STAGES=split` (the same probe) reproduces
Reactant's `optimization_passes = :all` list for a CPU backend
(`src/compiler/Compiler.jl`, the `:all` branch, `raise = false`) one stage at a
time on ONE module, timing and censusing each. Validated on `ssp_step`, where
the whole pipeline is seconds: the twelve stages sum to 17.1 s and end at 22,817
ops against a plain `opt` run's 22,745 on the same program.

### The stages of `ssp_vjp`, direct lane, 6x6x8

`ESM_DIRECT_GATHER_BASE_MAX` is the budget knob introduced with the fix below;
4096 is a STRICTER control than the shipped-at-72cbadc30 rule (which allowed
`max(8n, 4096)` and therefore let the wider reads through at 58,620 slices), so
read the left column as an upper bound on the old shape rather than a
reproduction of it.

| stage | what it is | budget 4096 | budget 65536 (the fix) |
| --- | --- | ---: | ---: |
| emitted `slice` | | 379,140 | **15,252** |
| emitted `gather` | | 4,248 | 4,968 |
| emitted `concatenate` | | 1,112 | 400 |
| `mark` | `mark-func-memory-effects` | 1.1 s | 0.3 s |
| `opt1` | `enzyme-hlo-opt`, pre-Enzyme | 69.9 s | **10.5 s** |
| `ebatch` | `enzyme-batch` | 0.1 s | 0.0 s |
| `opt2a` | `enzyme-hlo-opt` again | 7.4 s | 1.8 s |
| `enzyme` | **the differentiation itself** | **1297.6 s** | **142.8 s** |
| `opt2b` | `enzyme-hlo-opt` on the ADJOINT | not reached | > 37 min, capped |

Two things fall out. **The differentiation itself is superlinear in the slice
count** — 1297.6 s against 142.8 s for a 25x smaller slice set — which no
amount of pass exclusion touches, because `enzyme` is not one of the excludable
patterns. And **what is left is `opt2b`**: the FIRST `enzyme-hlo-opt` run over
the differentiated module, which at the fix's slice count carries 27,610
`slice`, 9,583 `pad` (the reverse of a slice is a pad-and-add), 10,354
`reshape` and 18,351 `add`. That is where the remaining wall is, and it is the
same `slice_elementwise` / `cse_slice` pair section 4 named, now working on the
adjoint's slices rather than the primal's.

### Where the 58,620 came from — the base budget, not the run structure

Counted on the emitted module (`mlirG2/ssp_vjp-direct-unopt.mlir`, the
`gather`-form emission at 72cbadc30): **640 reads reach a concatenate, and they
share only 37 DISTINCT PRODUCER SETS.** 240 of those reads have the same shape —
432 positions in 168 to 312 runs out of TWO producers, the 3744-slot extended
state beside a 2304-slot buffer, 6048 elements of base. The run-length test
wanted every one of them as a gather. `_de_gather_base` declined every one of
them, because it compared the base against ONE read (`tot > max(8 * n, 4096)`,
6048 against 4096) while the base is CACHED and that one concatenate would have
served all 240.

**43,368 of the 58,620 slices — 74% — are in reads that failed only that
clause.** The rest are single-producer reads whose average run is four or longer,
which is what the slice path exists for.

EarthSciAST 8433a50c3 replaces the per-read comparison with an absolute budget
on the COPY, charged only when there is one — a base over one producer with no
structural zero is not a concatenate at all, because `_de_concat` of a single
piece is that piece. That is the `budget 65536` column above.

### And it is not enough: `opt2b` survives every read form

Three more arms, same probe, same grid, one process each.

| arm | emitted `slice` | after `opt1` | `opt1` | `enzyme` | `opt2b` |
| --- | ---: | ---: | ---: | ---: | ---: |
| budget 65536 (the fix) | 15,252 | 9,932 | 10.5 s | 142.8 s | > 37 min |
| `ESM_DIRECT_EMIT_READ=always` | 8,800 | 8,726 | 49.1 s | 117.6 s | > 31 min |
| the fix, minus `slice_elementwise` | 15,252 | 10,015 | **2.9 s** | 102.7 s | **> 76 min** |
| the fix + emitter CSE (section below) | 11,476 | 9,932 | 4.5 s | 99.7 s | > 13 min |

**DO NOT READ THE `enzyme` COLUMN AS A LEVER.** Those four modules are within
1% of each other in slice count where it matters — 8,726 to 10,015 after
`opt1` — and the differentiation stage over them measured 99.7, 102.7, 117.6
and 142.8 s. That spread is the node, not the arm: this measurement ran through
a ninety-minute filesystem event on a shared node (section 6), and the 142.8 s
arm ran in the worst of it. What IS attributable is the 1297.6 s of the
4096-budget arm, whose slice count differs by 25x rather than by 3%.

None of the four `opt2b` numbers is a completion. The first two hit the probe's
own cap; the last two were killed from outside the probe at 76 and 13 minutes,
so the 76-minute figure is a floor on the `slice_elementwise`-excluded arm and
nothing in this table establishes an upper bound on any of them.

**Every read form converges to the same module after `opt1`** — 8,726 slices
against 9,932, for emitted counts that differ by 1.7x — and Enzyme turns that
into 23,988 / 27,610 slices plus 8,723 / 9,583 pads either way. `always` is the
CEILING on what the read lowering can buy (it gathers every read that
decomposes into more than one run, and lifts the budget entirely), and it moves
`opt2b` not at all. So the read lowering is finished as a lever, and the
paragraph in section 4 that guessed a materialization layout was right about
WHERE to look next even though it was wrong that the read lowering had nothing
left in it.

What is left after `opt1` is ~9,900 slices that no read form removes, because
they are not multi-run reads: they are single-position and short reads, one
slice each, with no concatenate to replace (`always` emits 8,800 slices against
20 concatenates). Merging those is a question about which slots sit next to
which — `src/tree_walk/oop_merge.jl`'s block layout — and not about
`_de_emit_runs`.

### Three quarters of the slices were the SAME slice

Counted on the same emitted module: 58,620 `stablehlo.slice`, **14,476 of them
distinct**. One span of the extended state is emitted a hundred times. That is
exactly the work `cse_slice` is doing pairwise, on a population the emitter knew
to be redundant before it wrote it, so EarthSciAST 5181e5c47 memoizes `_de_slice`
and `_de_concat` on the emission context (one straight-line block, so an earlier
value always dominates a later use).

It helps where it can and it stops where the measurement says it must:

| | the fix | the fix + emitter CSE |
| --- | ---: | ---: |
| emitted `slice` | 15,252 | 11,476 |
| emission | 3.4 s | 2.9 s |
| `opt1` | 10.5 s | **4.5 s** |
| after `opt1` | 9,932 | 9,932 |

**The module after `opt1` is identical to the op.** The pipeline was already
collapsing the duplicates — just expensively — so this buys emission and the
pre-Enzyme pass and cannot buy `opt2b`.

### And which pattern, once `slice_elementwise` is gone

`perf record -F 199 -g`, 150 s, 29,727 samples, taken live on `opt2b` in the
third arm above:

| share | symbol |
| ---: | --- |
| 19.8% | `mlir::OperationEquivalence::isEquivalentTo` (two frames) |
| 6.9% | `mlir::enzyme::failIfDynamicShape` (the `CheckedOpRewritePattern` guard) |
| 2.2% | `StaticSlice::get` |
| 2.0% | `CSE<stablehlo::SliceOp>::matchAndRewriteImpl` |

`cse_slice` is now the whole cost, and it is QUADRATIC: it compares each slice
against the others through `OperationEquivalence::isEquivalentTo`, and the
adjoint module carries 24,000 to 28,000 of them. Excluding it as well is the
combination section 4 already tried and it OOM-kills, because `cse_slice` is
what keeps the other slice patterns from multiplying an un-deduplicated set.
There is no pass exclusion that wins here; the slice population is the variable.

### Where that leaves the direct lane's adjoint

> **Overtaken at 6x6x8 on 2026-09-15 evening, and only there.** `@compile
> ssp_vjp` is 98.7 s in the demonstration preset on EarthSciAST `oop-delete`
> `87f4f2474` (`run_reseact_adjoint.jl`'s header has the full table, and the
> preset now runs end to end in 21 m 11 s with every acceptance check green).
> Two things moved at once and they are not separately measured: the emitter
> commits after `72cbadc30`, and the `make_tracer` opaque-leaf registration,
> which had never applied inside the adjoint driver because its guard tested
> `Main` and the driver runs in `Main.adjoint`. The section below is still the
> measurement of WHERE the time goes, and the ~9,900 slices are still the
> number the next lever has to move — at CONUS, where nothing has been
> measured.


`@compile ssp_vjp` at 6x6x8 is still the wall, against 115.4 s in the traced
lane, and the stage it is stuck in is now named: the FIRST `enzyme-hlo-opt`
over the differentiated module, in `cse_slice`. Everything before it is in
range — emission 2.9 s, `opt1` 2.9 to 4.5 s, the differentiation 102.7 to
142.8 s, which together are inside the traced lane's 115.4 s — and everything
the read lowering controls has been spent.

The target for whatever comes next is a number, not a direction: **the ~9,900
slices that survive `opt1` under every read form.** They are single-position and
short reads, one slice each, distinct from each other, with no concatenate to
replace and no duplicate to collapse; Enzyme's reverse of them is 24,000 to
28,000 slices plus ~9,000 pads, and `cse_slice` is quadratic in that. Four
levers were tried against them and all four are spent — the run-length rule, the
base budget, emitter-side CSE, and pattern exclusion. The one that is not is the
slot layout the reads address: if a stencil neighbour read were one contiguous
span, it would be one slice instead of several, and that is
`src/tree_walk/oop_merge.jl`.

## 6. The trap that cost this measurement three runs

**Every timing in section 5 was taken on a node whose page cache had lost the
3 GB `libReactantExtra.so`, and for ninety minutes that was indistinguishable
from a hung process.** Julia processes sat in `D`/`I` state at 0% CPU with 0.4 GB
resident and no output, for fifty minutes, while `dd` on the very same file
returned 204 MB/s and the cgroup reported no memory pressure at all. The
filesystem was not slow in bulk; individual memory-mapped PAGE FAULTS against it
were taking on the order of 200 ms each, and dynamic linking a 3 GB shared
object touches pages one at a time and out of order. `cat`-ing the library and
the depots' `.so` caches to `/dev/null` — 3 GB in 100 s — unblocked every
stalled process within seconds.

Three consequences to carry forward. A stalled Julia process here is worth ONE
check before it is worth a diagnosis: `cat` the artifact, and if it starts
moving the problem was never the code. Any wall-clock number taken on this node
while that is happening is worthless — the concurrent slurm adjoint's
`@compile ros_step` reads 2795.8 s against 69.4 s for the same program, and the
`enzyme` column of the table above spreads 1.4x across arms that are the same
module. And a probe that prints only on stage COMPLETION cannot tell a stall
from slow progress, which is why the split stage announces each pass before
running it.

## 6. The 9,900: not the reads, the WRITE

Section 5 ended on a number and a direction: **~9,900 slices survive `opt1`
under every read form**, they are single-position, and the next place to look
is the slot layout (`src/tree_walk/oop_merge.jl`). The direction was wrong, and
the reason it was wrong is that nothing had ever asked WHICH PART OF THE EMITTER
WROTE THEM.

### The attribution

EarthSciAST's direct emitter now tallies every emitted `slice` / `gather` /
`concatenate` a second time, under `<op>@<site>.<why>` — the emitter site that
was running (`_de_at!`: the scalar spine, a materialization level's scalars or
kernels or scans, an access kernel, a scan, the output assembly) and the read
form that asked for it. `RESEACT_CC_PROG=rhsT` prints the table. ONE emission
of the transport right-hand side, 6x6x8, before this section's fix:

| site | op | count |
| --- | --- | ---: |
| `assemble` | `slice`, ONE position | **1,728** |
| `kernel` | `gather` | 1,211 |
| `mat_kernel` | `slice`, wider | 698 |
| `assemble` | `slice`, wider | 432 |
| `mat_kernel` | `concatenate` | 89 |
| `mat_kernel` | `gather` | 31 |
| `mat_scan` | `slice`, wider | 8 |
| `kernel` | `concatenate` | 4 |
| `kernel` | `slice`, wider | 3 |
| `assemble` | `concatenate` | 1 |
| | **2,869 slices, 1,242 gathers, 94 concatenates** | |

2,869 x 4 SSPRK43 stages = 11,476, which is `ssp_vjp`'s emitted slice count to
the op. **Sixty per cent of every slice the emitter wrote is the OUTPUT
ASSEMBLY, and every one of those is a single position.** The scalar spine
emitted none at all.

### The lead that the attribution killed

The hypothesis this section was opened to test was the scalar spine: the `:oop`
build carries a lane-batching plan for it (`_OopScalarBatches`, the
`ESS_OOP_BATCH` mechanism, which on the traced lane cut dynamic slices from
64,834 to 3,682 at 7x7x16) and the direct emitter never referenced it — it
walked every scalar entry individually, which is exactly the shape that leaves
thousands of one-element reads.

It is a real emitter gap and it is now closed (EarthSciAST's
`ext/reactant_direct/batch.jl`), but **it moves nothing on this model**: run the
attribution with `ESS_OOP_BATCH=0` and with `ESS_OOP_BATCH=1` and the two
modules are identical op for op — 2,869 slices, 1,242 gathers, 94 concatenates
in both. ReSEACT's transport half has no per-cell scalar entries at all; every
read in it goes through an access kernel. The batched surface is kept because a
model that DOES decline the kernel path would otherwise emit O(cells) ops where
the interpreter emits O(1), but it is not what follows.

### `_de_assemble` had its own run-walk

Every read in the emitter goes through ONE form, `_de_emit_runs`: decompose the
positions into runs, then apply the cost model — one gather when the read
shatters, slices plus a concatenate when the runs are long, structural zeros
folded into either. Every read except one. `_de_assemble`, which turns the
finished `du` slot map into the returned vector, had its own local walk: merge
maximal runs of consecutive slots held at consecutive positions in the same
producer, emit one slice each, concatenate.

On a stencil model that walk finds almost nothing to merge. The output map is
INTERLEAVED — a kernel's result value holds its own cells, and the next slot in
ascending order usually belongs to a different value, or to a non-adjacent
position inside the same one — so the runs are mostly runs of ONE. 3,744 slots
decomposed into 2,160 pieces, 1,728 of them single positions.

The fix is one line of behaviour: the output is a read like any other.

```julia
_de_assemble(ctx, M, n) = _de_emit_runs(ctx, _DESlot[M.m[i] for i in 1:n])
```

2,160 pieces over 3,744 positions is an average run of 1.7, the cost model wants
a gather, and the producers concatenate once into a base well under the budget.
**The whole assembly becomes ONE `stablehlo.gather`.**

### What it costs and what it buys

One emission of the transport right-hand side, 6x6x8, `raw2` (warm emission, no
passes):

| | before | after |
| --- | ---: | ---: |
| pre-pass ops | 10,506 | **8,348** |
| `stablehlo.slice` | 2,869 | **709** |
| `stablehlo.gather` | 1,242 | 1,243 |
| `stablehlo.concatenate` | 94 | 94 |

and on `ssp_vjp`, the reverse-mode program that was the wall, the same twelve-
stage `RESEACT_CC_STAGES=split` run section 5 used (`ESM_DIRECT_EMIT_READ`
default, base budget default, emitter CSE on — the "the fix + emitter CSE"
column of section 5 is the `before`):

| stage | before | after |
| --- | ---: | ---: |
| emitted `slice` | 11,476 | **2,836** |
| emitted `gather` | 4,728 | 4,972 |
| after `opt1` (`slice`) | 9,932 | **1,292** |
| `opt1` | 4.5 s | 5.2 s |
| `enzyme` (the differentiation) | 142.8 s | **12.4 s** |
| adjoint `slice` / `pad` after `enzyme` | 27,610 / 9,583 | **3,862 / 943** |
| `opt2b` (`enzyme-hlo-opt` on the adjoint) | **> 37 min, capped** | **4.6 s** |
| twelve stages, total | — | **25.1 s** |

**`opt2b` is 4.6 seconds.** The stage that three separate arms could not get
through in half an hour, and that no read form and no pass exclusion moved, is
gone — because `cse_slice` is quadratic in the slice POPULATION and the
population is now a twentieth of what it was. The differentiation itself, which
section 5 measured as superlinear in the same variable, fell 11.5x on the same
evidence.

End to end:

| program | traced | direct, before | direct, after |
| --- | ---: | ---: | ---: |
| `@compile ssp_vjp` | 115.4 s | **> 6 h, unfinished** | **76.1 s** |

The direct lane's reverse-mode transport step now compiles FASTER than the
traced lane's.

### Why this was the last one standing

Four levers had been spent on the slice population and every one of them worked
on the READS: the run-length rule, the gather base budget, emitter-side CSE, and
pass exclusion. None of them could touch the assembly, because the assembly is
not a read of the state — it is the WRITE of the result, and it was the one
surface in the emitter that had been allowed to decide its own shape. Section 5
measured that "every read form converges to the same module after `opt1`" and
concluded the read lowering was finished as a lever. That was correct. What it
missed is that ~9,900 of the survivors were never reads at all.

The generalisable reading, for the next emitter and the next binding: **one read
form, no exceptions.** A surface that opens its own path around the cost model
will eventually be the surface that costs the most, and a flat op census cannot
tell you which surface that is. Attribute the ops to the code that wrote them.

## 7. The grid, 2026-09-15/16: three clauses that priced a stencil and not a model

Section 6 left `@compile ssp_vjp` — the reverse of the four-stage transport
step, and the program that had been the wall — at 76.1 s at 6x6x8, and the
emitter's read cost model looking settled. At the CONTINENTAL grid none of that held. A
three-grid census of ONE transport right-hand side (`RESEACT_CC_PROG=rhsT`,
`raw2`, i.e. emission alone) shows what grew:

| one `rhsT` emission | 6x6x8 | 13x7x16 | 13x7x72 (CONUS) |
| --- | ---: | ---: | ---: |
| cells | 288 | 1,456 | 6,552 |
| total ops | 8,348 | 37,086 | **105,474** |
| `stablehlo.slice` | 709 | 30,549 | **97,385** |
| `stablehlo.gather` | 1,243 | 563 | 721 |
| everything else | flat | flat | flat (1.33x for 22.75x the cells) |

The chemistry half is grid-independent in op count over the same span (13,619 →
18,192) with its data volume scaling correctly, so this is the transport read
form and nothing else. At CONUS `@compile reactant_ssp_step` never returned:
6 h 27 m, 99.3% of a live `perf` profile inside `xla::HloCSE::RunOnComputation`
(the pairwise `HloInstruction::IdenticalInternal`, one layer below the
`cse_slice` of section 5), and then `LLVM ERROR: Unable to allocate section
memory!` out of the CPU backend's contiguous section allocator — slurm
10567298, MaxRSS 48.4 GiB (52.0 GB), on a node with 160 GB requested.

### The two clauses, and why each is a grid cap in disguise

Both live in EarthSciAST `ext/reactant_direct/values.jl` and both compare a
property of the STENCIL against a cost that grows with the GRID.

**The average-run test.** `_de_gather_is_cheaper` priced one gather against a
slice per run by AVERAGE RUN LENGTH. That length is fixed by how a read walks
its axis — a column read runs the model's level count and nothing else — while
the NUMBER of runs is proportional to cells. At 8 levels the vertical runs are
2 to 4 and the reads gather; at 16 levels and above the dominant run is exactly
7, past the break-even of 4, so the same reads shatter into thousands of slices
at every grid from there up. It declined 1,246 reads at CONUS and accounts for
72,405 of the 97,385 slices (74%).

**The absolute base budget.** `_DE_GATHER_BASE_MAX` refused a cross-producer
gather whose base concatenate would copy more than 65,536 elements. The CONUS
extended state is 85,176 slots, so every cross-producer read was refused *by
construction* — including the output assembly, which spans every producer there
is: one gather at 6x6x8, 24,000 slices at CONUS (26% of the total).

`ESM_DIRECT_EMIT_READ=always` bounds from above what the read form can buy:
10,201 ops and 222 slices at CONUS, flat across grids, but 38.3 M i64 index
elements (293 MB) per right-hand side. That is the ceiling, not the fix.

### The fix: a piece cap, a model-relative budget, a canonical base

1. **A piece cap** (`_DE_GATHER_MAX_PIECES = 64`). Past it a read gathers
   whatever its runs look like, so no read costs more than the cap in ops;
   below it the average-run test decides as before, which keeps a short affine
   read on the slice path where it carries no index data.
   `ESM_DIRECT_GATHER_MAX_PIECES` overrides it.
2. **A model-relative base budget**, `max(65_536, 4 * n_extended_state)`, with
   the old constant as a floor. `ESM_DIRECT_GATHER_BASE_MAX` still overrides it.
3. **One canonical base per slot map, in slot order.** The map's distinct
   producers are concatenated once, in first-slot order, and reordered once if
   that concatenation is not already in slot order; every read that would cost
   more than the piece cap then reads that ONE value at plain slot indices. The
   concatenate is emitted once per map per write epoch rather than once per
   producer set, a read affine in slot space is an ordinary slice with no index
   constant, and `_de_assemble` — a read of slots 1..n — IS the base, so the
   output assembly emits nothing at all.

EarthSciAST `retire-oop` ee0606b0d, dc5782cae, 6cc2e807b.

### What the census says now

Same probe, same three grids, Reactant 0.2.285, EarthSciAST `retire-oop`
6cc2e807b. `before` is `oop-delete` a3f9e1ba1:

| one `rhsT` emission | 6x6x8 | 13x7x16 | 13x7x72 |
| --- | ---: | ---: | ---: |
| ops, before | 8,348 | 37,086 | 105,474 |
| **ops, after** | **8,334** | **8,456** | **11,893** |
| `slice`, before | 709 | 30,549 | 97,385 |
| **`slice`, after** | **686** | **562** | **2,412** |
| `gather`, before | 1,243 | 563 | 721 |
| `gather`, after | 1,248 | 1,293 | 1,474 |
| `concatenate`, before | 94 | 233 | 629 |
| `concatenate`, after | 93 | 130 | 515 |
| i64 constant elements, before | 1.57 M (12.0 MB) | 2.98 M (22.7 MB) | 13.4 M (102.3 MB) |
| i64 constant elements, after | 1.61 M (12.3 MB) | 8.47 M (64.6 MB) | 39.6 M (301.8 MB) |
| emission wall, before | 1.1 s | 3.8 s | 8.7 s |
| emission wall, after | 1.2 s | 2.9 s | 11.5 s |

**The op count is now flat in the grid** — 8,334 / 8,456 / 11,893 over a 22.75x
span of cells, against 8,348 / 37,086 / 105,474 — and the 6x6x8 module, where
none of the three clauses bites, is unchanged. The chemistry half is unchanged
too (13,619 → 13,628 ops at 6x6x8).

**The index data is not flat, and it is the cost of the trade.** 301.8 MB per
CONUS transport right-hand side is above the 293 MB that
`ESM_DIRECT_EMIT_READ=always` spends and well above the ~160 MB this work aimed
at. It buys the op count: a read of 6,552 positions in 936 runs is one gather
with 52 KB of indices instead of 936 operations, and it is the OPERATION count
that the quadratic deduplication passes — Enzyme-JAX's `cse_slice` and XLA's
`HloCSE` — are quadratic in. The consequence shows up in memory rather than in
time; see the peak RSS row below.

The site attribution says exactly which surface moved (one CONUS emission):

| site | op | before | after |
| --- | --- | ---: | ---: |
| `kernel` | `slice` | 66,895 | **5** |
| `kernel` | `gather` | 498 | 1,211 |
| `assemble` | `slice` | 24,000 | **0** |
| `assemble` | `gather` / `concatenate` | 0 / 1 | **0 / 0** |
| `mat_kernel` | `slice` | 6,418 | 2,335 |
| `mat_kernel` | `gather` / `concatenate` | 223 / 537 | 256 / 508 |
| `canon` | `concatenate` / `gather` | — | 7 / 7 |

Seven canonical bases over the whole emission, and the output assembly emits
nothing.

### `ssp_step` and `ssp_vjp` at 13x7x16

The four-stage step and its reverse, one program per process, production
compile options:

| 13x7x16 | emission (`raw2`) | `@compile` | pre-pass ops |
| --- | ---: | ---: | ---: |
| `ssp_step` | 9.3 s | 49.1 s | 33,903 |
| `ssp_vjp` | 10.0 s | 302.6 s | 34,162 |

### The 48 h CONUS gradient on the fixed emitter

`tools/diag/adjoint_conus_48h_direct.sbatch`, slurm **10575494**, scavenger
partition, one 40-core node, **59 m 08 s all in**, exit 0. Beside it, the
traced record it is measured against, slurm 10372969 (2026-09-05, 1 h 15 m 37 s):

| 13x7x72, 576 macro steps, 48 h | traced record 10372969 | direct 10575494 |
| --- | ---: | ---: |
| emitter | traced, `ESS_OOP_SSA=1` | direct StableHLO |
| Reactant | 0.2.280 (see below) | 0.2.285 |
| EarthSciAST | main of 2026-09-05 | `retire-oop` 6cc2e807b |
| BUILD | 230.4 s | 155.7 s |
| `prepare_jacobian` | 65.3 s | 33.7 s |
| plan vs the JacobianEvaluator | 0.000e+00 PASS | 0.000e+00 PASS (509.9 s) |
| `@compile ssp_step` | 159.5 s | **27.2 s** |
| `@compile ssp_vjp` | 345.4 s | **358.6 s** |
| forward pass | 692.6 s (1.202 s/macro step) | **432.9 s (0.751)** |
| backward sweep | 2,106.7 s (3.657 s/macro step) | **1,127.2 s (1.957)** |
| loop total | 47 min | **26 min** |
| cost ratio backward/forward | 3.04 (VJP-only 2.42) | 2.60 (VJP-only 1.72) |
| accepted inner steps | 27,973 (1,859 T, 26,114 C) | 27,970 (1,859 T, 26,111 C) |
| flaky-reverse retries | 0 / 27,973 | 0 / 27,970 |
| fixed-sequence replay | — | 0.000e+00 at every checkpoint |
| MaxRSS | 42.2 GiB (45.4 GB) | **116.4 GiB (124.9 GB)** |
| J (ppb mean surface O3) | 30.1943301698531 | 30.1943304387531 |

**The loop-cost datum**, per device call:

| | traced record | direct | ratio |
| --- | ---: | ---: | ---: |
| transport step, forward | 18.94 ms | 9.92 ms | 1.9x |
| transport step, replay | 19.82 ms | 9.56 ms | 2.1x |
| **transport VJP** | **326.86 ms** | **82.18 ms** | **4.0x** |
| chemistry step | 13.65 ms | 11.26 ms | 1.2x |
| chemistry VJP | 30.11 ms | 15.55 ms | 1.9x |
| forcing refresh | 3,866 ms | 1,298 ms | 3.0x |

The transport VJP is the program the whole read form was about, and it is 4.0x
the traced lane's per call. The chemistry and refresh gains are not this work's:
they come with the same environment and are reported so the loop total is not
attributed entirely to the emitter.

**J IS NOT A REPRODUCTION AND SHOULD NOT BE READ AS ONE.** It differs from the
record at 8.9e-9 relative, where the traced lane's own variants reproduced each
other to 13 digits. Three things differ between the two runs and only one of
them is this work, so the difference is not attributable:

* the emitter (traced against direct), which is the point of the run;
* Reactant against Reactant. The record's log does not print a version; the
  production traced environment of that date is Reactant 0.2.280 (AGREEMENT.md),
  and this run prints 0.2.285 in its own provenance block;
* the MODEL. The record carried 160 runtime scalars, this run carries 162:
  `Transport3D.dlat_deg` and `Transport3D.dlon_deg` became runtime scalars in
  reseact.esm since 2026-09-05, and both are nonzero in the gradient (19
  nonzero components then, 21 now).

The window, macro step count, objective, base point and jitter are identical
(576 x 300 s = 172,800 s, `SuperFast.O3:surf`, default initial condition,
`ujitter=0`), and the configuration is otherwise the record's value for value.

**What the ladder says about WHERE the difference is.** The TRANSPORT
accept/reject ladder is identical to the record — 1,859 accepted transport
steps in both — and the chemistry ladder differs by three accepted steps out of
26,114. The half this work changed reproduced the record's step sequence
exactly; the divergence is in the chemistry half, which is the adaptive
controller converting ulp-level differences into a different accept/reject
decision, the mechanism DIFFERENTIABILITY_PLAN.md section 5 documents.

All nineteen gradient components the two runs share agree; sixteen of them to
**5.2e-7 or better**, and the three largest in magnitude to 1.7e-7 or better:

| parameter | traced record | direct | rel |
| --- | ---: | ---: | ---: |
| `NEIRegrid.scale` | -2.1143633041322 | -2.1143633236492 | 9.2e-9 |
| `DryDepositionGas.kappa` | -1.6165013604182 | -1.6165010844056 | 1.7e-7 |
| `Transport3D.g_acc` | +0.66916827272938 | +0.66916823753613 | 5.3e-8 |
| `Transport3D.dlat_deg` | — (not a scalar then) | -0.67857169538 | — |
| `NEIRegrid.g0` | -0.21560505413492 | -0.21560505612510 | 9.2e-9 |
| `DryDepositionGas.g_const` | -0.51290882583830 | -0.51290880329965 | 4.4e-8 |
| `Transport3D.Rd_air` | -0.022861170673268 | -0.022861169470941 | 5.3e-8 |
| `Transport3D.lat0_deg` | +0.017701150585183 | +0.017701155404131 | 2.7e-7 |
| `Transport3D.lon0_deg` | +1.2220503711785e-03 | +1.2221790333493e-03 | **1.1e-4** |
| `Transport3D.dlon_deg` | — (not a scalar then) | +0.025480057104 | — |
| `GEOSFP.dt_interp_A1` | +2.4065195865537e-03 | +2.4065029990172e-03 | 6.9e-6 |
| `Transport3D.tau_pblmix` | -4.1311275049680e-04 | -4.1311270234881e-04 | 1.2e-7 |
| `SuperFast.CH4` | +9.9893465281987e-05 | +9.9893466413848e-05 | 1.1e-8 |
| `GEOSFP.t_interp_ref_A1` | +5.4698172131827e-05 | +5.4697814366183e-05 | 6.5e-6 |
| `GEOSFP.dt_interp_A3` | +4.0513025933446e-05 | +4.0513017577183e-05 | 2.1e-7 |
| `GEOSFP.dt_interp_I3` | +2.1403473012459e-05 | +2.1403473090578e-05 | 3.7e-9 |
| `GEOSFP.t_interp_ref_A3` | +3.5750021994023e-06 | +3.5750015426635e-06 | 1.8e-7 |
| `GEOSFP.t_interp_ref_I3` | +1.9464571995344e-06 | +1.9464572155646e-06 | 8.2e-9 |
| `WetDeposition.Vdr` | +1.0197593211775e-08 | +1.0197593147603e-08 | 6.3e-9 |
| `WetDeposition.rho_water` | -5.0987966058876e-11 | -5.0987965738013e-11 | 6.3e-9 |
| `DryDepositionGas.theta` | -1.0376715638196e-40 | -1.0376710202812e-40 | 5.2e-7 |

Three components fall outside 1e-6: `Transport3D.lon0_deg` at 1.1e-4 and the
`GEOSFP` A1-interpolation pair `dt_interp_A1` and `t_interp_ref_A1` at 6.9e-6
and 6.5e-6. All three are among the smallest sensitivities in the table and all
three are differences of nearly-cancelling contributions, which is what a
three-step change in the chemistry ladder moves first. The components that
carry the calibration signal — `NEIRegrid.scale`, `DryDepositionGas.kappa`,
`Transport3D.g_acc`, `NEIRegrid.g0`, `DryDepositionGas.g_const` — agree to
1.7e-7 or better, well inside the 1e-6 this work targets.

### The cost, stated plainly

**Peak memory nearly tripled: 42.2 GiB for the traced record, 116.4 GiB here.**
The gather index constants are the bulk of it — 301.8 MB per transport
right-hand side, four stages per step, forward and reverse — and the 160 GB the
sbatch asks for is now margin rather than headroom. That is the price of moving
the transport step from 105,474 operations to 11,893, and at this grid it is
worth paying, because the 105,474-operation program does not compile at all.
The next lever, if one is wanted, is the index data rather than the op count:
an affine run in slot space costs no index constant, and the canonical base has
made slot space the addressing space in which that is now expressible.

> **The sentence above about peak RSS is wrong and section 7.1 retracts it.**
> The index data WAS nine tenths redundant and is now a fifteenth of what it
> was; peak RSS did not move, because the gather index constants were never the
> bulk of it. 7.1 has the measurement and names what is.

### 7.1 The index data, 2026-09-16: it was nine tenths redundant, and it was not the peak

The paragraph above names the index data as the next lever and the peak RSS as
what it would buy. Half of that is right. The index data was almost all
redundant and is now a fifteenth of what it was — and **peak RSS did not
move**, because the gather index constants were never the bulk of it. The
retraction is the more useful half of this measurement and is stated in full
below.

**What the index vector actually cost.** An emission was writing one index
constant per gather, and most of the gathers were repeats. A gather's index
vector is a property of the READ and of nothing else, while the VALUE a read
addresses changes under it: a write to a slot map retires the map's canonical
base, so the next read of the same slots is a different gather over a different
value at exactly the same indices, and on a stencil the map is written between
almost every pair of reads. Counting the distinct contents of every
`tensor<Lx1xi64>` constant in the dumped CONUS module says how much of it was
repeat:

| one `rhsT` emission, index constants | 6x6x8 | 13x7x16 | 13x7x72 |
| --- | ---: | ---: | ---: |
| emitted | 1,248 | 1,293 | 1,474 |
| DISTINCT contents | 123 | 146 | 155 |
| bytes emitted | 12.3 MB | 64.6 MB | 301.8 MB |
| bytes if each distinct vector were emitted once | 1.5 MB | 8.2 MB | 41.6 MB |

And the element type was the widest available. `stablehlo.gather` takes start
indices of any integer type; a 32-bit index addresses a base far longer than a
slot map that fits in memory (the largest index the CONUS emission uses is
247,422), so the wide form was buying nothing.

**The two commits.** EarthSciAST `gather-index-memory` 8b8ac53f9 interns the
gather on `(value, positions)` — the emitter-side common-subexpression rule the
slice and concatenate forms already followed, which subsumes the canonical
base's own memo — and the index constant separately on its CONTENTS, so a
module carries one copy of each distinct index vector however many gathers read
at it. dda88a330 emits that vector as `i32` whenever every position into the
base is representable in one, `i64` otherwise;
`ESM_DIRECT_GATHER_INDEX_BITS` pins the width as the negative control.

#### What the census says now

Same probe, same three grids, Reactant 0.2.285. `before` is `retire-oop`
0b216f783, the emitter slurm 10575494 ran:

| one `rhsT` emission | 6x6x8 | 13x7x16 | 13x7x72 |
| --- | ---: | ---: | ---: |
| total ops, before | 8,334 | 8,456 | 11,893 |
| **total ops, after** | **6,086** | **6,162** | **9,256** |
| `gather`, before | 1,248 | 1,293 | 1,474 |
| **`gather`, after** | **125** | **146** | **156** |
| `constant`, before | 1,273 | 1,317 | 1,498 |
| `constant`, after | 148 | 170 | 179 |
| `slice` | 686 → 686 | 562 → 562 | 2,412 → 2,412 |
| `concatenate` | 93 → 93 | 130 → 130 | 515 → 515 |
| index data, before | 12.3 MB | 64.6 MB | **301.8 MB** |
| **index data, after** | **0.7 MB** | **4.1 MB** | **20.8 MB** |
| module text, before | 25.5 MB | 130.5 MB | 606.3 MB |
| module text, after | 2.0 MB | 9.1 MB | 44.0 MB |
| emission wall, before | 0.8 s | 2.4 s | 7.5 s |
| emission wall, after | 0.4 s | 1.2 s | 3.5 s |

**Nothing about the READ FORM moved.** `slice`, `concatenate`, `reshape`,
`broadcast_in_dim`, `select` and `transpose` are identical at all three grids;
only `gather` and `constant` fall. What the emitter decided about every read is
what it decided before — the change removes redundancy and narrows a type, and
takes no cost-model decision at all. The op count also stays flat in the grid
(6,086 / 6,162 / 9,256 over 22.75x the cells), which was the property the
previous round bought and the one a fix here had to keep.

The index-data row is counted twice and independently: from the emitter's own
`stats[:gather_index]` tally, and by parsing every integer constant out of the
dumped module text and summing its real width. 20.8 MB at CONUS is 14.5x below
the 301.8 MB the previous round spent and well below the ~160 MB this work
aimed at.

#### The 48 h CONUS gradient, and the retraction

`tools/diag/adjoint_conus_48h_direct.sbatch` with its `cd` and its
`RESEACT_RXENV` default pointed at this worktree and its environment, and
nothing else changed — slurm 10575494's configuration value for value, on
EarthSciAST `gather-index-memory` dda88a330 — slurm **10586921**, scavenger,
one 40-core node, **1 h 05 m 30 s all in**, exit 0:

| 13x7x72, 576 macro steps, 48 h | direct 10575494 | this run 10586921 |
| --- | ---: | ---: |
| EarthSciAST | `retire-oop` 6cc2e807b | `gather-index-memory` dda88a330 |
| J (ppb mean surface O3) | 30.19433043875315 | **30.19433043875315** |
| all 21 gradient components | — | **bit-identical** |
| accepted inner steps | 27,970 (1,859 T, 26,111 C) | 27,970 (1,859 T, 26,111 C) |
| clamp bits on the tape | 51,308 | 51,308 |
| fixed-sequence replay | 0.000e+00 | 0.000e+00 |
| flaky-reverse retries | 0 / 27,970 | 0 / 27,970 |
| plan vs the JacobianEvaluator | 0.000e+00 PASS | 0.000e+00 PASS |
| forward pass | 432.85 s | 411.08 s |
| backward sweep | 1,127.23 s | 1,049.58 s |
| transport step, per call (exec) | 9.92 ms | 10.55 ms |
| transport VJP, per call | 84.88 ms | 89.98 ms |
| chemistry step, per call | 11.26 ms | 10.26 ms |
| forcing refresh, per call | 1,298.08 ms | 374.76 ms |
| `@compile ssp_vjp` | 358.6 s | 336.9 s |
| per-shard `compile step` | 179.7 s | 59.6 s |
| BUILD | 155.7 s | 192.1 s |
| `prepare_jacobian` | 33.7 s | 126.8 s |
| **MaxRSS** | **116.4 GiB** | **118.7 GiB** |

**The numerics are unchanged in the strongest sense available.** J is
bit-identical to all 17 digits, every one of the 21 gradient components is
bit-identical, both accept/reject ladders are identical step for step, and the
tape's clamp-bit count is identical. That is the reproduction the previous
round could not claim, and it is what an emitter change that removes duplicate
constants and narrows an index type should produce.

**And MaxRSS did not fall.** 118.7 GiB against 116.4 GiB, having cut the index
data of one transport right-hand side from 301.8 MB to 20.8 MB. The arithmetic
says why, and it should have been done before the run rather than after: four
SSPRK stages of forward plus reverse is at most ~2.4 GB of index constants in
the whole job, which is 2% of a 116 GiB peak. Section 7's claim that "the
gather index constants are the bulk of it" was an attribution by plausibility
and it is **wrong**.

**Where the peak actually is**, from the numbers both runs already printed:

| at the peak | 10575494 | 10586921 |
| --- | ---: | ---: |
| chemistry shard workers, each | 10.0-10.2 GB | 10.0-10.1 GB |
| worker RSS, max / total (forward) | 10.5 GB / 83.8 GB | 10.6 GB / 84.6 GB |
| worker RSS, max / total (backward) | 10.9 GB / 86.5 GB | 11.0 GB / 87.2 GB |
| job MaxRSS | 116.4 GiB | 118.7 GiB |

**Eight chemistry shard worker processes at ~10.6 GB each are ~87 GB of the
peak, three quarters of it, and they are the same size in both runs.** The
remainder is the driver, whose own peak is the `ssp_vjp` compile. The traced
record this is all measured against (10372969, 42.2 GiB) ran the same eight
shards at ~4 GB each, so the regression to chase is the SHARD WORKER's
footprint under the direct lane — a chemistry program, whose reads are not
shattered and which emits almost no index data at all. The 2.3 GiB between the
two runs here is inside the run-to-run spread of this partition: the three
stages that precede any emission at all moved by more (BUILD +23%,
`prepare_jacobian` +276%, the JacobianEvaluator check +22%), and the emitter
cannot touch any of them.

**What the levers bought, then, is compile time and module size, not peak
memory.** The per-shard chemistry `compile step` is 3.0x faster (179.7 s →
59.6 s) because the chemistry module carries a ninth of the gathers it did, the
forcing refresh is 3.5x faster per call, emission of the CONUS transport right-
hand side is 2.1x faster, and the module the pipeline has to hold is 44.0 MB
of text rather than 606.3 MB. Execution cost is unchanged within noise.

#### Two levers measured and NOT taken

**Generating the affine lattices in-program.** The standing finding was that
the merged-class slot vectors are affine lattices and could be produced from an
`iota` and a handful of arithmetic ops instead of a dense constant. Counting
the distinct index vectors of the dumped modules says how far that reaches: at
CONUS **18 of the 155** distinct vectors are exact multi-level affine lattices,
and they are 0.54 MB of the 20.8 MB. The other 137 are not lattices under any
uniform decomposition (the common shapes decompose into 168, 98, 504, 6,720 and
6,440 arithmetic runs, and the run lengths are not uniform). Generating them
would cost ops — the thing the previous round bought — for 3% of the bytes.

**Interning across emissions.** An explicit Runge-Kutta step emits the same
right-hand side once per stage into ONE MLIR block, so a module-scoped intern
would fold four copies into one. It is not taken, and not because it would not
work: the cache would have to be keyed on the MLIR block, a freed block's
address can be reused by a later one, and a stale hit is a dangling SSA
reference rather than a wrong number. With the per-emission interning in place
the whole four-stage step carries ~83 MB of index data, so the remaining 4x is
not worth a use-after-free hazard in the emitter.

### 7.2 The chemistry shard worker, 2026-09-16/17: two compiles nobody calls

Section 7.1 ended by naming the eight chemistry shard workers as three
quarters of the 48 h CONUS peak — about 10.6 GB each and about 87 GB together,
against about 3.4 GB each on the traced lane. This is what was in them.

#### Profiling one worker

A shard worker's build depends on its cell count `C` and on the model, not on
the domain it was cut from, so ONE worker at the CONUS shard size can be
profiled in a session: a 13x7x9 grid is 819 cells, which is exactly 6552 / 8.
The driver's own external sampling of that worker's `/proc` agrees with the
CONUS job's per-shard figure to within a tenth of a gigabyte, so it is the same
worker. `RESEACT_SHARD_MEMLOG=1` (ReSEACT `gather-index-memory` 6ca0475) prints
resident size, the process high-water mark and Julia's live heap at every phase
boundary of `build!`.

| phase, one worker at C=819 | resident, before | resident, after |
| --- | ---: | ---: |
| worker start, libraries loaded | 1.64 | 1.60 |
| model loaded and split | 1.79 | 1.67 |
| capacity document | 1.66 | 1.67 |
| `build_evaluator` | 2.47 | 2.31 |
| **base-point finiteness check** | **4.74** | — |
| `prepare_jacobian` | 4.65 | 3.04 |
| **gather-plan check** | **7.05** | 5.10 / 3.09 |
| `@compile` the step | 7.58 | 5.80 / 3.65 |
| base point steps finite | — | 5.81 / 3.67 |
| `@compile` the VJP | 9.32 | 7.56 / 5.50 |
| after the build's release | 8.83 | **7.06 / 5.00** |
| process high-water mark | **9.54** | **7.92 / 5.60** |

Two columns after the fix because the gather-plan check now runs on one shard
per capacity size: the first figure is that shard, the second is every other
one.

**Almost none of it is Julia.** Live heap after a full collection is 0.57 GB
against 9.4 GB resident, and `summarysize` over everything the worker holds for
the rest of the run — the split document, the capacity document, the capacity
right-hand side, the band model, the block-Jacobian plan, the lane buffers and
the two compiled programs — comes to about 1.3 GB, of which the two compiled
programs are more than half (each `Thunk` keeps its own MLIR module text). The
loaded documents, the first candidate anyone would reach for, are eight and
three megabytes. The footprint is native, and it arrives one compile at a time.

#### The cause: four XLA:CPU compiles, of which the worker calls two

Each `@compile` in this worker retains between one and two gigabytes after a
full garbage collection and a `malloc_trim`, and it retains about the same
amount whatever the program is: the two guards below cost as much as the step
and the VJP the shard exists to run.

The worker performs four. Two are the step and the VJP. The other two are
BUILD-TIME GUARDS that each compile a whole program to evaluate one point and
then drop it:

* the base-point finiteness guard, which catches a lane the forcing gather did
  not reach (an unfilled lane is a zero pressure and NaNs through
  `log(PS/Pc)`);
* `validate_plan`, which checks that the block-Jacobian plan's padded gather
  and per-block slot lists reproduce the band evaluator's own `umap` and
  `scatter`.

**Both are there because the host-callable out-of-place build was retired.**
They used to evaluate the build directly on `Vector{Float64}`s. An `:oop` build
product is now the compiled IR a backend lowers and raises
`E_TREEWALK_OOP_NOT_EVALUABLE` if called, so each guard reaches for
`rx_host_eval`, which compiles. That is the whole of the regression against the
traced lane, and it is in the shard driver rather than in the emitter.

Bounding it: with both guards removed the worker's high-water mark is 5.65 GB
against 9.54 GB, so **the two throwaway compiles are 41% of it**.

**A cheaper compile is not available.** Running the two guards with
`optimization_passes = false` was tried and does not compile at all: XLA:CPU
fails on the unoptimized module of this program. The lever is the number of
compiles, not their cost.

#### The fix

ReSEACT `gather-index-memory` a17c715. Neither guard is weakened.

1. **The finiteness guard runs on the step the shard is about to run.** NaN
   propagates through the ROS23 stage solves and through the per-cell error
   norm, so a non-finite right-hand side at the base point is a non-finite step
   at the base point; the guard is one device call on a program the worker
   compiles anyway. The price is that it fires after that compile rather than
   before it.
2. **The gather-plan check runs once per distinct capacity size.** What it
   validates is index algebra, and every shard built at the same `C` builds the
   identical capacity document and therefore the identical tables; the state,
   the parameters and the lane data are the probe for that comparison and not
   its subject. The driver names the shards that run it.
3. **`build!` collects and returns what the build borrowed** before the worker
   settles, since a scheduler records resident size and not live size.

#### The 48 h CONUS gradient

`tools/diag/adjoint_conus_48h_direct.sbatch`'s configuration with the `cd` and
`RESEACT_RXENV` pointed at this worktree — slurm 10586921's configuration value
for value — slurm **10597002**, scavenger, one 40-core node, **47 m 23 s all
in**, exit 0:

| 13x7x72, 576 macro steps, 48 h | 10586921 | 10597002 |
| --- | ---: | ---: |
| ReSEACT | f820518 | `gather-index-memory` a17c715 |
| J (ppb mean surface O3) | 30.19433043875315 | **30.19433043875315** |
| all 21 gradient components | — | **bit-identical** |
| accepted inner steps | 27,970 (1,859 T, 26,111 C) | identical |
| clamp bits on the tape | 51,308 | identical |
| replay / flaky retries | 0.000e+00 / 0 | identical |
| per-shard worker resident | 10.0-10.2 GB | **5.3-5.5 GB, one at 7.3 GB** |
| worker RSS, max / total | 10.6 / 84.6 GB | **7.9 / 46.4 GB** |
| **job MaxRSS** | **118.7 GiB** | **78.0 GiB** |
| forward / backward | 411.1 / 1,049.6 s | 393.0 / 989.7 s |
| BUILD / `prepare_jacobian` | 192.1 / 126.8 s | 125.0 / 32.9 s |
| `@compile ssp_step` / `ssp_vjp` | 87.1 / 336.9 s | 27.1 / 338.3 s |
| wall | 1 h 05 m 30 s | **47 m 23 s** |

**Peak memory falls by 40.7 GiB, a third of the job**, with J and every
gradient component bit-identical and both accept/reject ladders identical step
for step. The shard workers account for 38.2 GB of that, which is what the
three changes above predicted from the single-worker profile.

#### The 6x6x8 acceptance, both shard settings

The demonstration preset, unsharded (`RESEACT_ADJ_SHARDS=0`, all four
validation stages), reproduces the recorded table exactly: J =
38.84667055571979 and all 21 nonzero gradient components BIT-IDENTICAL,
structural identity 1.213e-15 PASS, frozen replay bit-identical, `fdtape` all
three parameters PASS, 0 flaky retries. That arm does not run a shard worker at
all, and it is the control that says the merge of `main` into this branch moved
no number.

The same preset at the driver's default eight shards (slurm 10597384, on a node
of its own — eight workers plus the driver do not fit the interactive cgroup)
gives the same J to the last bit and 20 of the 21 components bit-identical
against that unsharded record, the twenty-first (`Transport3D.dlon_deg`, the
smallest transport component) at 1.5e-14 relative. That difference is between
SHARDED and UNSHARDED, not between before and after: the sharded path sums each
shard's partial dJ/dp in shard order and the unsharded path does not. The
before/after comparison at eight shards is the CONUS pair above, and it is
bit-identical in every component.

Its workers come out at 4.6-4.7 GB each with one at 4.7 GB, a total of 37.6 GB
— the same shape as at CONUS, and a reminder that most of a worker is the model
and its two compiled programs rather than its cell count: 36 cells per shard
here against 819 there.

#### What is left, and where it is

The traced lane's own record at this grid and shard count (slurm 10386109,
2026-09-08, EarthSciAST v0.1.1, Reactant 0.2.280 — the resolution-scaling
arm's 4x5 48 h leg) had workers at 3.3-3.5 GB, a worker total of 27.7 GB and a
job MaxRSS of 40.9 GiB. Against that:

| | traced 10386109 | direct 10586921 | direct 10597002 |
| --- | ---: | ---: | ---: |
| worker, each | 3.3-3.5 GB | 10.0-10.2 GB | 5.3-5.5 GB |
| worker total | 27.7 GB | 84.6 GB | 46.4 GB |
| job MaxRSS | 40.9 GiB | 118.7 GiB | 78.0 GiB |
| implied driver | ~15 GiB | ~40 GiB | ~35 GiB |

The workers are now within about 1.6x of the traced lane, and what remains
there is the two compiled programs themselves, which the shard genuinely runs.
**The rest of the gap is the DRIVER**, whose peak is the `ssp_vjp` compile —
the transport half, not the chemistry shards — and which this work did not
touch.
