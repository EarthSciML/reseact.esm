# Upstream report: reverse-over-forward AD in Reactant/Enzyme-MLIR

Written to be filed by a human. Three INDEPENDENT bugs were found while trying
to put an exact (AD) block Jacobian inside a reverse-mode adjoint; they are
reported separately because they have different owners, different severities,
and only two of them are fully isolated.

* **Bug A — SEGFAULT.** `Enzyme.gradient(Reverse, ...)` over a function that
  contains `Enzyme.autodiff(Forward, ...)` calls dies in
  `AutoDiffCallRev::createReverseModeAdjoint` on a NULL `FuncOp`.
  **Root cause of the CRASH is identified and is a one-line unchecked-null in
  Enzyme-MLIR. What makes the underlying differentiation FAIL is NOT identified**
  — see "what we could not establish".
* **Bug B — MISCOMPILE.** The `concat_broadcast_slice` HLO rewrite merges
  adjacent `concatenate` operands along the wrong axis and produces a malformed
  module. **Fully isolated, 8-line reproducer, no autodiff involved, root cause
  quotable from the source.**

Everything below is measured on this machine, not inferred.

## Versions

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

# Bug B — `concat_broadcast_slice` produces a malformed `stablehlo.concatenate`

Filing target: **EnzymeAD/Enzyme-JAX** (the pattern lives in
`src/enzyme_ad/jax/Utils.cpp`), cross-referenced from **EnzymeAD/Reactant.jl**
because Reactant enables the pattern by default.

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

# Bug A — segfault in `AutoDiffCallRev::createReverseModeAdjoint`

Filing target: **EnzymeAD/Enzyme** (`enzyme/Enzyme/MLIR`). The *crash* is
unambiguously an unchecked null there. Whether the *failure* that produces the
null is also an Enzyme bug or a Reactant lowering problem is open.

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

**We do not have a model-free reproducer.** The crash is reproducible on the
ReSEACT atmospheric model (13 species, 288 cells, `tools/rx_adjoint_check.jl`
stage `ros_ad`, ~600 s build), and a substantial bisection *failed* to
reproduce it on a toy that carries the same nesting. What was ruled out is
listed below so that whoever picks this up does not re-walk it.

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
gap is the whole remaining gap, and it is large. Measured on the module as
Reactant hands it to Enzyme (`@code_hlo optimize=false`, 6x6x8 grid, N=3744):

| | toy at the same NS/NC | ReSEACT |
|---|---|---|
| module size | ~0.1 MB | **34.8 MB** (323 435 lines) |
| `func.func` (minted helpers) | 0–13 | **1346** |
| `stablehlo.gather` | 0 | 3648 |
| `stablehlo.dynamic_slice` | 0 | 2016 |
| `stablehlo.scatter` | 0–1 | 48 |
| `stablehlo.while` | 0 | 0 |

The helper functions the emitted RHS mints, by name and count (from the tracer's
own instrumentation): `TypeCast{Float64}_broadcast_scalar` 439,
`reduce_fnadd_sum` 196, `log10_broadcast_scalar` 54, `update_computation`
(the `stablehlo.scatter` region) 19, out of 745 total. Every one of those is a
`func.call` the reverse pass must cross, and `AutoDiffCallRev` is the handler
for exactly that — so "one of these 1346 callees comes back null from
`CreateReverseDiff`" is the shape of the answer. **Which one, we do not know**,
and the crash destroys the only place that information exists.

<!-- RESEACT_RESULTS -->

## Files in this repo

| path | what |
|---|---|
| `tools/diag/rof_repro.jl` | parametrised bisection harness (one config per process) |
| `tools/diag/rof_sweep.sh` | records the verdict from outside the process |
| `tools/diag/rof_results.tsv` | every configuration run and its verdict |
| `tools/diag/rof_bisect_pattern.sh` | binary search that named `concat_broadcast_slice` |
| `tools/diag/rof_concat_repro.jl` | the 8-line Bug B reproducer |
| `tools/diag/rof_batchfwd.jl` | batched-forward-mode probe (see below) |
| `tools/rx_adjoint_toy.jl` | the model-free variant harness that predates this |
| `tools/rx_adjoint_check.jl` | the on-model harness; stages `ros_ad`, `jacrev`, `addump` |

---

# Bug C — batched forward mode does not lower, by either route

Filing target: **EnzymeAD/Reactant.jl**.

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
