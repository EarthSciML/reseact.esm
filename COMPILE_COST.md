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
