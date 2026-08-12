# Upstream report: reverse-over-forward AD in Reactant/Enzyme-MLIR

Written to be filed by a human, as **two separate issues** (plus a third, minor,
if wanted). They were all found trying to put an exact (AD) block Jacobian
inside a reverse-mode adjoint, but they are independent bugs with different
owners, and each section below is self-contained enough to paste on its own —
copy the Environment block into each.

| | what | file to | isolated? |
|---|---|---|---|
| **Bug B** | `concat_broadcast_slice` merges `concatenate` operands along the wrong axis and emits a malformed module | EnzymeAD/**Enzyme-JAX** | **yes** — five-line reproducer, no autodiff, root cause quotable |
| **Bug A** | reverse-over-forward segfaults on a NULL `FuncOp` in `AutoDiffCallRev::createReverseModeAdjoint` | EnzymeAD/**Enzyme** | **crash mechanism yes, trigger no** — reproduces only on a real model |
| Bug C | batched forward mode does not lower, by either available route | EnzymeAD/**Reactant.jl** | yes, but minor |

Everything attributed to *this machine* is measured, not inferred. Where
something is a hypothesis it says so.

## Environment (copy into each issue)

| | |
|---|---|
| Julia | 1.12.6, x86_64-linux-gnu |
| Reactant | 0.2.274 |
| Reactant_jll | 0.0.395+1 |
| Enzyme | 0.13.190 (reached as `Reactant.Enzyme`; not a direct dep) |
| backend | CPU (PJRT) |

**Caveat on the C++ quoted below.** Enzyme-JAX and Enzyme reach this machine
only as the prebuilt `Reactant_jll` 0.0.395+1 binary; nothing was built from
source here, so **the upstream commit the jll was built from cannot be pinned**.
The C++ was read from the `main` branches of `EnzymeAD/Enzyme-JAX` and
`EnzymeAD/Enzyme` on the day of writing: treat the file paths and function names
as reliable and any line numbers as indicative only. Everything attributed to
*this machine* — error text, pass names, timings, module contents — is measured
on the versions in the table.

---

# ISSUE 1 (Bug B) — `concat_broadcast_slice` produces a malformed `stablehlo.concatenate`

**File to: EnzymeAD/Enzyme-JAX** (the pattern lives in
`src/enzyme_ad/jax/Utils.cpp`); worth cross-referencing from
**EnzymeAD/Reactant.jl**, which enables the pattern by default.

**Severity: silent wrong-shape miscompile.** Here it failed the verifier loudly,
but the rewrite itself is a shape error in a rewrite that is on by default, and
nothing about it is autodiff-specific.

## Reproducer

No Enzyme, no autodiff, no gradients — plain `vcat` of reshaped slices.
Full script: `tools/diag/rof_concat_repro.jl`.

```julia
using Reactant
const NBLK, BLK = 13, 5
rowstack(x) = vcat([reshape(x[(k*BLK+1):((k+1)*BLK)], 1, BLK) for k in 0:(NBLK-1)]...)
X = Reactant.ConcreteRArray(collect(1.0:(NBLK*BLK)))
@compile rowstack(X)
```

Observed, verbatim (log: `tools/diag/logs/concat_repro_13x5.log`):

```
=== concat repro: 13 blocks x 5  (N=65) ===
┌ Error: Compilation failed, MLIR module written to /tmp/reactant_cqSpts/module_000_nxc9_post_all_pm.mlir
└ @ Reactant.MLIR.IR .../Reactant/PMrNV/src/mlir/IR/Pass.jl:146
  default                    FAIL error: type of return operand 0 ('tensor<65x1xf64>') doesn't match function result type ('tensor<5x13xf64>') in function @rowstack
  no concat_broadcast_slice  OK   (values correct)
```

Excluding the one pattern fixes it and the values are correct:

```julia
@compile compile_options=Reactant.CompileOptions(;
    excluded_passes=["concat_broadcast_slice"]) rowstack(X)   # OK
```

Measured for `NBLK` = 2, 3 and 13 — all fail by default, all correct with the
pattern excluded.

**Note on where the concatenate comes from.** The input IR contains no
`stablehlo.concatenate` at all: `vcat` lowers to a chain of
`stablehlo.dynamic_update_slice` into a `13x5` buffer (verified with
`@code_hlo optimize=false`). The pipeline itself forms the concatenate
(`dynamic_update_to_concat` / `dus_dus_concat`) and then `concat_broadcast_slice`
mangles it. So this is entirely an optimizer-internal miscompile; nothing about
the user program has to mention `concatenate`.

The whole function collapses to the merged slice — post-pipeline body of
`@rowstack` in the dumped module:

```mlir
"func.func"() <{function_type = (tensor<65xf64>) -> (tensor<5x13xf64>, tensor<65xf64>),
                sym_name = "rowstack"}> ({
^bb0(%arg0: tensor<65xf64>):
  %0 = "stablehlo.reshape"(%arg0) : (tensor<65xf64>) -> tensor<65x1xf64>
  "func.return"(%0, %arg0) : (tensor<65x1xf64>, tensor<65xf64>) -> ()
```

All 13 operands were merged into one full-extent slice and the result shape was
computed as `65x1` instead of `13x5`.

## What the rewrite does wrong

A larger module (the same rewrite, but only a *run* of the operands merged)
shows the shape error directly and is the easiest form to read:

```mlir
// before: 13 operands, each  reshape(slice(%4902[k*5 : (k+1)*5])) : tensor<1x5xf64>
%12441 = "stablehlo.concatenate"(%12429, ..., %12429) <{dimension = 0 : i64}>
       : (tensor<1x5xf64>, ... x13 ...) -> tensor<13x5xf64>

// after: operands 3..12 merged into ONE slice, taken along the WRONG axis
%12445 = "stablehlo.reshape"(%4902) : (tensor<65xf64>) -> tensor<1x65xf64>
%12446 = "stablehlo.slice"(%12445) <{start_indices = array<i64: 0, 15>,
          limit_indices = array<i64: 1, 65>}> : (tensor<1x65xf64>) -> tensor<1x50xf64>
%12447 = "stablehlo.concatenate"(%12442, %12443, %12446, %12444)
          <{dimension = 0 : i64}>
        : (tensor<1x5xf64>, tensor<1x5xf64>, tensor<1x50xf64>, tensor<1x5xf64>)
       -> tensor<13x5xf64>

error: shapes of operand (0) and (2) are not compatible at non-concat index 1:
       (1, 5) != (1, 50)
error: 'stablehlo.concatenate' op failed to infer returned types
```

Ten adjacent operands, each a `1x5` view of rows `k` of the source, were merged
into a single `1x50`. That merge is only sound when the axis along which the
slices are adjacent is **the concatenation axis**. Here the slices advance along
operand dim 1 while the concat is along dim 0, so the merged operand should have
been `10x5`.

## Root cause, from the source

`src/enzyme_ad/jax/Utils.cpp` (main, and matching the behaviour of the pinned
Reactant_jll 0.0.395):

```cpp
static Value mergeConcatSlicedElems(PatternRewriter &rewriter,
                                    int64_t concatDim, ConcatSlicedElem a,
                                    ConcatSlicedElem b) {
```

`concatDim` is **never referenced in the body** — `grep -n concatDim
Utils.cpp` returns only the two declaration lines. The function checks that the
two elements share a source (`a.src != b.src`), that they are adjacent and
unit-stride (`a.limit != b.start || a.stride != b.stride || a.stride != 1`),
and then merges. Nothing checks that the sliced dimension maps onto `concatDim`
under `a.perm`.

Suggested fix (not implemented or measured here): in `mergeConcatSlicedElems`,
bail out unless the sliced dimension of the merged operand corresponds to
`concatDim` — i.e. unless `a.perm[<index of a.dim>] == concatDim` — or else
reshape/transpose the merged slice into the shape the concatenate requires.

## Why it matters here

`vcat` of per-species blocks of one state vector is exactly
`concatenate(reshape(slice(x)))`, so a block-diagonal chemistry Jacobian walks
into it. In our case it made every reverse-mode compile at >= 5 species fail
and blocked an unrelated bisection outright. It has nothing to do with
autodiff — it just showed up first inside a gradient, because that is where the
`reshape` that turns the rank-1 concatenate into a rank-2 one appeared. (A
rank-1 concatenate of rank-1 adjacent slices is a *valid* merge, which is
presumably why this has not been hit more often.)

---

# ISSUE 2 (Bug A) — segfault on a NULL `FuncOp` in `AutoDiffCallRev::createReverseModeAdjoint`

**File to: EnzymeAD/Enzyme** (`enzyme/Enzyme/MLIR`). The *crash* is
unambiguously an unchecked null there. Whether the *failure* that produces the
null is also an Enzyme bug or a Reactant lowering problem is open — and cannot
be answered from outside, for the reason given under "which callee".

**The ask, in one line:** null-check `revFn` in `edetail::callReverseHandler`
and `emitError()` instead of dereferencing it, so the failing callee names
itself. Everything else in this issue is context for *why* that check is the
thing that unblocks a real diagnosis.

## Symptom

Compiling a Reactant function whose body is

```
Enzyme.gradient(Reverse, f, ...)          # outer reverse
  -> f calls Enzyme.autodiff(Forward, Const(g), Duplicated, Duplicated(u, v))
     once per colour (13 colours), to build a block Jacobian by JVPs
```

segfaults inside `DifferentiatePass`:

```
AutoDiffCallRev::createReverseModeAdjoint
  -> func::CallOp::build
  -> mlir::Operation::getAttr        <-- on a NULL FuncOp
```

It is a hard crash: the process dies, no MLIR diagnostic is printed, and a
Julia `try`/`catch` cannot see it.

## Mechanism of the crash (certain)

`AutoDiffCallRev<OpTy>::createReverseModeAdjoint` is a one-line forwarder
(`CoreDialectsAutoDiffImplementations.h:243`) to
`detail::callReverseHandler`, which is registered for `func::CallOp` by
`registerAutoDiffUsingCallInterface`. In
`CoreDialectsAutoDiffImplementations.cpp` (read from `main`; the pinned
Reactant_jll 0.0.395 was not built from source here, so line numbers may drift
but the code path is the one the reported frames name):

```cpp
auto revFn = gutils->Logic.CreateReverseDiff(
    fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, returnShadow,
    mode, freeMemory, gutils->AtomicAdd, width, /*addedType*/ nullptr,
    type_args, overwritten_args, /*augmented*/ nullptr, gutils->omp,
    gutils->postpasses, gutils->verifyPostPasses, gutils->strongZero,
    /*markReadonly=*/false);

SmallVector<Value> revArguments;
...
auto *revCallOp = cast<AutoDiffFunctionInterface>(revFn.getOperation())
                      .createCall(builder, orig->getLoc(), revArguments);
```

**`revFn` is used without being checked.** Every other failure path in the same
function returns `orig->emitError() << ...` and produces a diagnostic; this one
does not. When `CreateReverseDiff` fails it hands back a null
`FunctionOpInterface`, `revFn.getOperation()` is `nullptr`, and the
`createCall` -> `func::CallOp::build` -> `getAttr("sym_name")` chain
dereferences it. That is exactly the reported frame sequence.

**Minimum ask, independent of the underlying cause:** check `revFn` and
`emitError()` instead of crashing, e.g.

```cpp
if (!revFn)
  return orig->emitError()
         << "failed to create reverse-mode adjoint for callee "
         << fn.getNameAttr() << "\n";
```

The same unchecked pattern appears for `forwardFn` in
`edetail::callForwardHandler` a few hundred lines above. Turning this crash
into a diagnostic is what would let a user report the real bug — right now the
information about *which* callee failed to differentiate dies with the process.

## Reproducer status — HONEST ACCOUNT

**We do not have a model-free reproducer.** The crash reproduces on the ReSEACT
atmospheric model (13 species, 288 cells, `tools/rx_adjoint_check.jl` stage
`jacrev`, ~600 s build — see "What DOES reproduce it" below), and a substantial
bisection *failed* to reproduce it on a toy that carries the same nesting. What
was ruled out is listed below so that whoever picks this up does not re-walk it.

`tools/diag/rof_repro.jl` runs one configuration per process (a segfault cannot
report its own death, so `tools/diag/rof_sweep.sh` records the verdict from
outside) and `tools/diag/rof_results.tsv` is the raw log. The differentiated
program is always the same: a Rosenbrock23 step whose block Jacobian is built
by `NS` coloured forward-mode JVPs, reverse-differentiated as a whole.

| axis moved from the working toy | value | reverse-over-forward |
|---|---|---|
| baseline | NS=3, NC=5, `theta::Vector` | **OK** |
| colour count (= nested `enzyme.fwddiff` ops) | NS=4 | OK |
| colour count | NS=5,6,7,8,10,13 | fails, but with **Bug B**, not the segfault |
| colour count, Bug B excluded from the pipeline | NS=13 | **OK** (59.6 s) |
| block width | NC=288 (ReSEACT's) | **OK** |
| `theta` shape | `(p = NamedTuple of 13 traced scalars, bufs = Tuple of traced arrays)` | **OK** |
| helper-minting RHS constructs, all at once: `log10`, `floor`, `sum` reduce, `Int` cast, `scatter` (`setindex!` with an index vector), gather, `ifelse`, `x^1.5` | NS=3 | **OK** |
| the same, plus a `stablehlo.dynamic_slice` at a **traced** `Int` index (what a forcing interpolation emits) and a `Reactant.@trace if` region | NS=3 | **OK** |
| without `rx_native_patch.jl`, i.e. every broadcast a minted `func.func` reached by `func.call` | NS=3 | **OK** |
| everything above simultaneously (NS=13, `nt` theta, all constructs, Bug B excluded) | | **OK** (77.6 s) |
| **exactly ReSEACT's dimensions**: NS=13, NC=288, N=3744, `nt` theta, Bug B excluded | | **OK** (96.7 s) |

So: **not** the number of nested `enzyme.fwddiff` calls, **not** the `theta`
NamedTuple carrying mixed traced scalars and arrays, **not** the block width,
**not** any single helper-minting construct nor their union, and **not** the
presence of `func.call`s inside the differentiated region. `tools/rx_adjoint_toy.jl`
had already ruled out `log10`/`floor`/`sum`-reduce individually.

What is *left* between the toy and the model is the emitted ReSEACT RHS itself
(EarthSciAST `:oop` emitter: 13 species of SuperFast gas-phase chemistry with
photolysis, emissions, deposition, and forcing-buffer interpolation). The size
gap is the whole remaining gap, and it is large.

**Provenance of the numbers below, stated exactly.** They are a census of one
ROS23 chemistry-step VJP module as Reactant hands it to Enzyme
(`@code_hlo optimize=false`, 6x6x8 grid, N=3744), and that module is the
**`jac=:fd`** one — it was dumped before the `ros_vjp` call-site bug (below) was
found, so the file labelled `_ad` was in fact FD. They are quoted here only to
bound the SCALE of the emitted RHS, which is common to both Jacobian flavours
and is the only thing the argument rests on. An equivalent census of the genuine
`jac=:ad` module is what stage `addump` now produces.

| | toy at the same NS/NC | ReSEACT, ROS23 VJP (`jac=:fd`) |
|---|---|---|
| module size | ~0.1 MB | **34.8 MB** (323 435 lines) |
| `func.func` | 0–13 | **1346** |
| `stablehlo.gather` | 0 | 3648 |
| `stablehlo.dynamic_slice` | 0 | 2016 |
| `stablehlo.scatter` | 0–1 | 48 |
| `stablehlo.while` | 0 | 0 |

The helper functions the emitted RHS mints — a property of the RHS, so common to
both flavours (from the tracer's own instrumentation):
`TypeCast{Float64}_broadcast_scalar` 439, `reduce_fnadd_sum` 196,
`log10_broadcast_scalar` 54, `update_computation` (the `stablehlo.scatter`
region) 19, out of 745 total. Every one of those is a `func.call` the reverse
pass must cross, and `AutoDiffCallRev` is the handler for exactly that — so
"one of the callees comes back null from `CreateReverseDiff`" is the shape of
the answer. **Which one, we do not know**, and the crash destroys the only place
that information exists.

## What DOES reproduce it: the coloured Jacobian alone, on the model

Reproduced on 2026-08-12 with the stage algebra removed entirely. The
differentiated program is now just

```julia
# tools/rx_adjoint_check.jl, stage `jacrev`
function _jdot_ad(g, u, th, t, ::Val{K}) where {K}
    Jb = ad_block_jac(uu -> g(uu, th, t), u, Val(K), NC)   # K coloured JVPs
    sum(sum(Jb[r, s]) for r in 1:K, s in 1:K)
end
Enzyme.gradient(Reverse, _jdot_ad, Const(gC), u, theta, Const(t), Const(Val(13)))
```

— no Rosenbrock stages, no `blocksolve`, no `sum(lambda .* unew)`, no linear
algebra of any kind. **Reverse mode over the coloured forward-mode Jacobian is
by itself sufficient.**

* grid 6x6x8, NS=13 species, NC=288 cells, N=3744 states
* **default pass pipeline — NO `excluded_passes`.** Bug B does not fire here and
  did not contaminate this verdict.
* build 601.9 s, then SIGSEGV (`EXIT=139`) during `@compile`
* log: `tools/diag/logs/reseact_jacrev13b.log`

**Why this verdict cannot be a mislabel** (it matters — a sibling probe *was*
mislabelled today, see the provenance note at the end). Two independent checks:

1. **Structural.** `_jdot_ad` calls `RTI.ad_block_jac(...)` *directly*. There is
   no `jac` kwarg anywhere in this path and therefore no default to inherit
   silently — which is exactly the failure mode that bit `ros_vjp`. The AD
   Jacobian is not selected here, it is the only thing called.
2. **Observed in the IR.** `enzyme.fwddiff` op counts in the pre-pipeline module
   (`ROF_HLO`, toy at NS=4 so it is cheap and non-crashing):

   | | `enzyme.fwddiff` | `enzyme.autodiff` |
   |---|---|---|
   | `jac=:ad` | **4** (= NS colours) | 1 (the outer reverse) |
   | `jac=:fd` | **0** | 1 |

   So "the AD path" is directly visible as `enzyme.fwddiff` ops in the module,
   one per colour, and the FD path emits none. The nesting the crash needs is a
   thing you can count, not a thing you have to trust.

The stack is the mechanism above, frame for frame:

```
DifferentiatePass::runOnOperation
  DifferentiatePass::lowerEnzymeCalls
    MEnzymeLogic::CreateReverseDiff
      MEnzymeLogic::differentiate
        MEnzymeLogic::visitChild
          ReverseAutoDiffOpInterface::createReverseModeAdjoint
            AutoDiffCallRev::createReverseModeAdjoint
              mlir::func::CallOp::create(OpBuilder&, Location, FuncOp, ValueRange)
                mlir::func::CallOp::build(OpBuilder&, OperationState&, FuncOp, ValueRange)
                  mlir::Operation::getAttr(llvm::StringRef)   <-- SIGSEGV
signal 11 (1): Segmentation fault
```

`func::CallOp::build(..., FuncOp callee, ...)` does `callee.getNameAttr()`,
i.e. `getAttr("sym_name")`, on the `FuncOp` it is handed. It is handed the
unchecked `revFn`. So the null comes from `CreateReverseDiff` returning failure
for one of the callees inside the reverse pass — silently, with no diagnostic
emitted before the crash.

### Which callee — and why we cannot tell you

**We could not identify the callee whose `CreateReverseDiff` returned null, and
we do not think it can be identified from outside the compiler.** Three
independent reasons, all of them checked rather than assumed:

1. The information exists only inside the process that dies. A segfault takes
   the process with it and a Julia `try`/`catch` cannot see it, so every probe
   here had to be one configuration per process with the verdict recorded by the
   parent.
2. `Reactant_jll` ships a stripped `libReactantExtra.so` — the backtrace above
   is mangled symbol names with `(unknown line)` throughout. There is nothing to
   set a breakpoint on and no way to print `fn.getNameAttr()` at the crash site.
3. **No MLIR diagnostic precedes the fault.** `CreateReverseDiff` returned
   failure silently; the log goes straight from the last tracer heartbeat to
   `signal 11`.

That is the whole argument for the null check. It is not a nice-to-have: with
`emitError()` in place this report would have named the op in one run, and
without it no amount of work on our side can.

### Corroboration: the original form crashes identically

The question as originally posed — reverse-differentiate the **whole `jac=:ad`
ROS23 step**, not just the Jacobian — gives the same verdict. So the minimal
form above is a genuine reduction of the real bug, not a different one:

```
---- ros_advjp : jac=:ad REVERSE only, excluded_passes=String[] ----
[1469164] signal 11 (1): Segmentation fault
```

* same grid, build 644.4 s, `EXIT=139` (core dumped)
* again the **default pipeline, no `excluded_passes`**
* the same frame sequence: `getAttr` <- `func::CallOp::build` <-
  `func::CallOp::create` <- `AutoDiffCallRev::createReverseModeAdjoint` <-
  `visitChild` <- `differentiate` <- `CreateReverseDiff` <- `lowerEnzymeCalls`
* log: `tools/diag/logs/reseact_ros_advjp.log`

`ros_advjp` is also the stage to re-check a fix with: it compiles only the
reverse pass, so it does not first pay for the `jac=:ad` JVP.

### Probes left running at close-out (no verdict)

Two further on-model probes were still compiling when this was written and are
reported as unfinished rather than guessed. Their logs are in `tools/diag/logs/`
and each ends in `EXIT=<code>`; 139 is the segfault.

| probe | question it would answer | status |
|---|---|---|
| `reseact_jacrev1b.log` | does a **single** colour (NCOL=1) also segfault? If yes it is a materially smaller reproducer and should replace the NCOL=13 one above | reached the reverse compile, no verdict |
| `reseact_addump3.log` | op census of the genuine `jac=:ad` module | building, no verdict |

`addump3`'s purpose — confirming the crash verdict is on the AD path — was met
by the two cheaper checks in "Why this verdict cannot be a mislabel" above, so
it is corroboration rather than a dependency. **The NCOL=1 question is genuinely
open** and is the one thing that would still improve this report.

## Files in this repo

| path | what |
|---|---|
| `tools/diag/rof_repro.jl` | parametrised bisection harness (one config per process) |
| `tools/diag/rof_sweep.sh` | records the verdict from outside the process |
| `tools/diag/rof_results.tsv` | every configuration run and its verdict |
| `tools/diag/rof_bisect_pattern.sh` | binary search that named `concat_broadcast_slice` |
| `tools/diag/rof_concat_repro.jl` | the five-line Bug B reproducer |
| `tools/diag/rof_batchfwd.jl` | batched-forward-mode probe (Issue 3) |
| `tools/rx_adjoint_toy.jl` | the model-free variant harness that predates this |
| `tools/rx_adjoint_check.jl` | the on-model harness; stages `jacrev`, `ros_advjp`, `addump` |

## Provenance note, for anyone re-checking these verdicts

Two probes were mislabelled during this work and both are corrected above rather
than quietly dropped:

1. **`rx_adjoint_check.jl`'s `ros_vjp` never passed `jac=:ad`.** It omitted the
   kwarg and inherited `ros23_step_vjp`'s `:fd` default, from the first commit
   on — so the harness stage named "jac=:ad" had been reverse-differentiating
   the FD Jacobian all along. Proven, not inferred: the two dumped modules are
   byte-identical after renaming the function. Fixed; the census table above is
   relabelled accordingly, and a new `ros_advjp` stage compiles only that
   reverse pass so the question can be asked without first paying for the
   `jac=:ad` JVP.
2. **An `EXIT=137` was briefly recorded as evidence about the `jac=:ad` JVP.**
   It was a memcg OOM-kill caused by running five model builds at once in a
   40 GiB SLURM step cgroup — self-inflicted, and retracted. Note for anyone
   running these: `/sys/fs/cgroup/memory.stat` is the whole node; the cap that
   matters is `system.slice/slurmstepd.scope/<job>/memory.max`, and its `anon`
   is the number to threshold on.

---

# ISSUE 3 (Bug C, minor) — batched forward mode does not lower, by either route

**File to: EnzymeAD/Reactant.jl.**

Found while evaluating the standing suggestion to *remove* the nesting by
emitting ONE width-NS forward derivative instead of NS width-1 ones.
`tools/diag/rof_batchfwd.jl` compares three ways to get the same NS Jacobian
columns of the same tiny RHS (NS=4, N=12). The serial form works; **both
batched forms fail**, so the suggestion is not currently actionable.

**(a) serial, what `ad_block_jac` does today — works**

```julia
ntuple(Val(NS)) do s
    Enzyme.autodiff(Forward, Const(rhs), Duplicated, Duplicated(u, seed_s))[1]
end
```
compiles in ~15 s and emits **NS separate `enzyme.fwddiff` ops** (measured: 4
for NS=4). That op count is the thing an outer derivative then has to cross.

**(b) `BatchDuplicated` — fails at XLA export**

```julia
Enzyme.autodiff(Forward, Const(rhs), BatchDuplicated, BatchDuplicated(u, (v1,v2,v3,v4)))
```
```
error: 'enzyme.extract' op unsupported op for export to XLA
note: see current operation:
  %2 = "enzyme.extract"(%0) <{index = 0 : i64}> : (tensor<4x12xf64>) -> tensor<12xf64>
```
`enzyme.extract` survives the pipeline with no lowering.

**(c) `Reactant.Ops.batch` wrapped around a width-1 `fwddiff` — fails in the
batch lowering**

```julia
Ops.batch([umat, seedmat], [NS]) do uu, vv
    Enzyme.autodiff(Forward, Const(rhs), Duplicated, Duplicated(uu, vv))[1]
end
```
```
error: TransposeOp operand rank 1 does not match permutation size 2
error: 'stablehlo.transpose' op failed to infer returned types
  note: %3 = "stablehlo.transpose"(%2) <{permutation = array<i64: 1, 0>}>
        : (tensor<12xf64>) -> tensor<12x4xf64>
```

## The compile-cost problem is NOT the colour count

The exact Jacobian is separately blocked by compile cost: a forward JVP of the
`jac=:ad` step on ReSEACT 6x6x8 has been reported not to finish in 85 minutes,
against 141 s for the same JVP of the `jac=:fd` step. That is a different
problem from the segfault and it survives a fix to it.

Scaling the colour count on the toy at fixed block width (NC=64) does **not**
reproduce it — compile seconds, `tools/diag/logs/cost_*`:

| NS | primal :ad | primal :fd | JVP of :ad | JVP of :fd |
|---|---|---|---|---|
| 3 | 32.7 | 28.9 | 37.7 | 36.7 |
| 6 | 35.7 | 31.1 | 38.1 | 38.8 |
| 9 | 37.4 | 32.2 | 41.6 | 39.5 |
| 13 | 39.9 | 35.5 | 74.3 | 65.1 |

At ReSEACT's NS=13 the AD JVP is 14% dearer than the FD one, not 35x. So the
cost is not in the number of nested derivatives per se; it scales with what is
*inside* each of them. `ad_block_jac` emits NS separate `enzyme.fwddiff` ops
(measured, section (a) above), and on ReSEACT each one wraps a 3000-op RHS with
745 minted helper functions — so the differentiated program is ~13 copies of
that before the outer derivative starts, while the FD Jacobian is 13 *calls* to
one copy. That is a plausible mechanism and it is consistent with the numbers
above, but **it is a hypothesis: it was not measured on the model here.**

The other half of the suggestion — "build the Jacobian columns directly against
`Reactant.Ops` instead of through `Enzyme.autodiff`" — has no target to aim at:
the only Ops-level primitive that produces a derivative *is* `enzyme.fwddiff`,
which is what `Enzyme.autodiff(Forward, ...)` already emits. Dropping to Ops
therefore changes nothing unless it can emit ONE fwddiff of width NS, i.e.
unless (b) or (c) works.
