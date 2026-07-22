# Results: splitting ReSEACT and tracing the halves under Reactant.jl

Continues `README.md`. Run on a **188 GB / 20-core Linux** box (the earlier handoff was
blocked purely by an 8 GB laptop's RAM). Env: `run-model-jl` with `Reactant` added
(`JULIA_DEPOT_PATH=/projects/illinois/eng/cee/ctessum/ctessum/.julia`; ~7.5 GB XLA
artifacts). Reactant defaults to a GPU backend and there is none here, so the scripts
call `Reactant.set_default_backend("cpu")`.

## TL;DR

1. **The splitter needs no change to emit a traceable RHS.** `split_system` /
   `build_split_evaluator` are representation-agnostic and forward `form=:oop` straight
   through `build_evaluator`, so each split half is a genuine out-of-place RHS. The split
   reconstructs the full model **exactly**: `(du_transport + du_pointwise) − du_full = 0.0`.
2. **The chemistry (pointwise) half compiles and runs under Reactant, bit-exact.**
   `@compile rhs_with_buffers(fo)(u,p,t,dev)` ≈ 90–110 s at 7×7×72 (45,864 states);
   the compiled program matches the host RHS to `maxabs(dev−host) = 1.4e-19`.
3. ~~The transport half does NOT compile~~ **SOLVED 2026-07-21**: with native broadcast
   lowering + the kernel-class merge, the transport half compiles too (§2026-07-21), and
   the merge is now landed in EarthSciAST's build (§2026-07-21 later).
4. **End-to-end simulation runs with BOTH halves on XLA** (§2026-07-21 later). The earlier
   hybrid (chemistry-only XLA) section below is retained for history.

## The transport blocker (concrete, fileable)

`@compile` of the transport half fails with:

```
Could not find unique name for *_broadcast_scalar
```

Cause, exactly:
- Reactant's `__lookup_unique_name_in_module` (`src/TracedUtils.jl:1050`) uniquifies a
  symbol by appending `_1 … _10000` and **errors past 10,000**.
- Reactant's `elem_apply` (`src/TracedUtils.jl:1224`) mints a **fresh** named helper
  (`<op>_broadcast_scalar`) via `make_mlir_fn` **per broadcast site**, with no
  memoization by (op, arg-types). The transport half's traced MLIR contains **25,945**
  such helpers — **10,002 of them the identical scalar-multiply body** — so the multiply
  helper alone crosses the 10,000 cap.
- Why so many: EarthSciAST's transport RHS is inlined with **no CSE sharing**
  (perf-gap-closure-plan.md: one species re-inlines to 57,688 `index` ops), and the
  `enzyme.batch` lowering batches over the **49 horizontal cells** but **inlines the 72
  vertical levels** — so helper count scales with `NLEV`.

It is **size-driven, not a semantic incompatibility** (see the compile-scaling test).

### It is Reactant, not EarthSciAST
EarthSciAST's `:oop` path is correctly **whole-array vectorized** — the transport MLIR has
4,197 `stablehlo.slice` (stencil windows as shifted whole-array reads), native tensor
`multiply`/`add`, and a **grid-independent op count** (identical at NLEV=8 and NLEV=72), i.e.
exactly the `_oop_run_acc_vec` design (whole-array gather → broadcast ladder → one scatter).
The helpers come from **Reactant's broadcast lowering**: every elementwise broadcast over
traced arrays goes `TracedRArray.jl:_copyto! → elem_apply → make_mlir_fn("<op>_broadcast_scalar")`
(TracedUtils.jl:1224) — **one named helper per broadcast op, no native-`stablehlo` fast path,
no body dedup**. 10,002+ distinct multiplies ⇒ 10,002+ identical `multiply(x,y)` functions ⇒
past the hardcoded `0:10000` cap. This is **known open Reactant issue #1616** (same error;
maintainers: the cap "did not expect anyone to exceed this limit"). The proper fix —
function-call insertion with cached defs (PR #523, jumerckx) — is a **draft, currently broken,
unmerged** as of 2026-02; the caching part was *removed* from the merged #1665. There is **no
user-facing flag** and the cap is not configurable (checked `CompileOptions.jl` /
`Configuration.jl` on 0.2.274 and `main`).

### Fixes, in order of leverage
1. **Reactant:** memoize/dedup identical batched-helper bodies (or lower primitive broadcasts
   straight to `stablehlo`). Collapses the helper set to ~10; the cap is never approached AND
   the module stays small. The only real fix.
2. Raising the 10,000 cap — **tested, see below; insufficient on its own.**

### Experiment: raising the cap (rx_raise_cap_experiment.jl)
Monkeypatched `__lookup_unique_name_in_module`'s `0:10000` → `0:10_000_000` (non-destructive,
runtime `@eval`) and re-ran the transport `@compile` at 7×7×8:
- ✅ The trace **passed 10,000** (every prior run died there) and kept going — **50,000+**
  naming calls, no cap error. So the cap is genuinely the first blocker and is removable.
- ❌ Then it **OOM-killed at 41.4 GB anon-rss, still tracing** (never reached XLA backend). Our
  Slurm step is capped at `SLURM_MEM_PER_NODE≈40 GB`; building the un-deduplicated MLIR module
  (25,945+ `func.func`, and growing) exhausts it before the trace even completes.

**Conclusion:** raising the cap is necessary but **not sufficient** — the module's sheer size
(from un-deduplicated helpers) is a second, memory wall. This is exactly why the real fix is
Reactant-side dedup / function-call caching, not the cap. A larger-memory job (≫40 GB) might
let the trace complete, but the pathological module size would remain.

## 2026-07-21: TRANSPORT COMPILES — native lowering + kernel merge

Two composable fixes, both prototyped here with **no EarthSciAST or Reactant source
edits**, take the transport half from "hard-blocked" to **compiled, verified, and
20× faster than the host RHS**:

### Fix 1 — native broadcast lowering (`rx_native_patch.jl`)
Runtime `@eval` methods on `Reactant.TracedUtils.elem_apply` intercept ~20 primitive
ops (`* + - / ^ min max muladd ifelse comparisons & | identity abs sqrt exp log
tanh sin cos`) and emit `Reactant.Ops.*` (native stablehlo) directly — legal because
`_copyto!` (TracedRArray.jl:394-404) `broadcast_to_size`s every arg *before*
`elem_apply`, so array args always share one shape. Zero `<op>_broadcast_scalar`
helpers ⇒ the 10k name cap is unreachable and the module stays small.
Microbench (`rx_native_bench.jl`): correct at every size, ~1-1.5 ms/site vs stock
5-8 ms/site. This alone let the FULL (unmerged) transport trace complete (~30 min,
7.3 GB, 3.5M+ sites) — revealing the true blowup was ~100× worse than the 25,945
helpers the stock runs died at — but `@compile` then sat for hours in Reactant's
closure-capture walk (`make_tracer`) over the captured 8M-node IR (killed after
~4 h, never finished). Necessary, not sufficient.

### Fix 2 — post-hoc kernel merge (`rx_merge_kernels.jl`)
The real volume: the :oop transport IR is **4,119 top-level `_AccKernel`s (mostly
1-lane = per-cell) + per-kernel template-body sub-kernel copies = 7.93M unique
nodes**, in only ~346 lockstep-mergeable classes (each class = one formula × 12
species). EarthSciAST's interning/CSE/acc-merge machinery is form-agnostic and
working — the fragmentation is upstream of it (per-cell instantiation of the same
stencil formulas). The prototype groups kernels by a lockstep signature
(shape/ops/acc-kind-families/buffer ids/cached tiers/fn-spec content), then clones
each class representative with varying leaves redirected to concatenated per-lane
tables using STOCK descriptors (`_AccStateTblBox(tbl,1,0,0,1)` with 0=ghost,
`_AccConstBox`, `_AccArrTblBox`; `_outs_cells` lane addressing m1=ordinal), subs
merged recursively (their plans are parent-lane-aligned), sub inv-CSE tier folded
into the cell tier. Evaluation is stock `_build_oop_acc_plan` + `_oop_run_acc_vec`.

### Results (7×7×8, 5,096 states)

| metric | before | after merge |
|---|---|---|
| kernels | 4,119 (+ sub copies) | **346** |
| unique IR nodes | 7,933,282 | **1,063,563** |
| host RHS, bit-identical? | — | **yes: maxabs=0.0, 5096/5096 exact** |
| host RHS time | 1,895 ms | **200 ms (9.5×)** |
| trace: broadcast sites | 3.5M+ (unfinished) | **311,027** |
| helpers minted | 25,945+ → cap error | **348** (343 `update_computation` regions, 3 TypeCast) |
| `@compile` transport | never succeeded | **SUCCEEDED, 393 s total** |
| compiled vs host | — | maxabs = 1.1e-15 ✓ |
| compiled RHS eval | — | **9.69 ms/call (20× vs merged host, ~195× vs original)** |

Debug war story: a nested closure assigning `r` silently REBOUND the enclosing
function's `s, r, av, sc = ...` destructuring variable (Julia closure capture),
corrupting every cloned node's op field to the last-visited leaf's — symptom was
`E_TREEWALK_UNSUPPORTED_OP: ` (empty op). Renamed to `msp_/mrc_/mav_/msc_`.

### What this means
- The 10k-cap question is closed: **yes, the :oop side can get under it** — native
  lowering makes helper count ~0 regardless, and the merge makes the whole program
  small enough that every downstream wall (trace time, closure walk, XLA size,
  memory) collapses at once.
- The durable home for Fix 2 is EarthSciAST's build (merge across arrayop
  instances at build time, where the per-cell entries already flow through
  `acc_merge.jl`); the durable home for Fix 1 is Reactant (issue #1616 / PR #523
  territory) or the EarthSciASTReactantExt.
- Remaining 346-vs-~30 class split is NOT boundary-kind driven (coarsening
  access-kind families changed nothing); prime suspect is per-class interp-spec
  content in the signature. Collapsing it needs table-ized interp specs — an
  optimization, not a blocker.

## Compile-scaling test (answers the README's open question)

The README asked: *is the traced program grid-independent in size, or O(#cells)?* and
suggested shrinking the grid. Finding: **7×7 is essentially the minimum valid horizontal
grid** — 5×4, 4×3, etc. fail to even *load* (`makearray_region_inverted`: the transport
stencil's interior region `[2, N-1]` goes below the scheme's minimum extent, esm-spec
§9.6.8). So the smallest testable grid is ~7×7×(small NLEV), and:

| grid (NLON×NLAT×NLEV) | states | transport `@compile` | `_broadcast_scalar` helpers |
|---|---|---|---|
| 7×7×8  (smallest valid) | 5,096  | **FAILS** — `Could not find unique name for *_broadcast_scalar` | **25,945** (10,002 are `*`) |
| 7×7×72 (full)           | 45,864 | FAILS — same | **25,945** (10,002 are `*`) |
| < 7 in either horizontal dim | —  | won't even load (stencil below minimum extent) | — |

**The helper count is identical (25,945) at NLEV=8 and NLEV=72** — it is *grid-independent*.
This actually **inverts the README's worry**: the traced transport program did **not**
degrade to O(#cells); the vertical levels are fully batched (`enzyme.batch`), so the ~10k
scalar-multiply helpers are an *intrinsic, fixed* property of the model's species×stencil
structure. The program is grid-independent in size **as the extension intends** — but that
fixed size carries 10,002 identical `*_broadcast_scalar` helpers, which alone exceed
Reactant's 10,000-name cap. **So no valid grid escapes the blocker**, and shrinking the grid
cannot help. (The **chemistry** half compiles at every grid; verified bit-exact at 7×7×8
too: `maxabs(dev-host)=1.1e-19`.)

## End-to-end hybrid simulation

`run_split_reactant.jl` — chemistry on XLA, transport interpreted on CPU (a half that
fails/skips XLA falls back to interpreted host `:oop` eval, reading the same live forcing
buffers). Same operator-split scheme as `tools/run_scale.jl`: SSPRK43 transport ⊕
Rosenbrock23 with a 13×13 block-diagonal FD Jacobian for chemistry; GEOS-FP forcing
refreshed on the host each cadence and mirrored to the device with `sync_forcing!`.

60 s window at 7×7×72:

```
RESULT label=reseact.esm cells=3528 nstates=45864 NS=13 transport=CPU chem=XLA
       solve_s=52.35 solve_secs=60 nT=2 nC=109 rcT=Success rcC=Success
       m=[9.986e-01,5.017e+03] O3=[3.9379e+01,3.9459e+01] ok=true
```

Both sub-solves return Success, air density stays positive, O3 ≈ 39 ppb (physical).

**Cross-check vs the all-CPU reference** (`tools/run_scale.jl`, same 60 s window) — identical
to printed precision, confirming the XLA chemistry gives the same full-solve answer as the
interpreter, not just a matching single RHS eval:

```
RESULT label=cpu_baseline ... nT=2 nC=109 rcT=Success rcC=Success
       m=[9.986e-01,5.017e+03] O3=[3.9379e+01,3.9459e+01] ok=true
```

Note the hybrid solve (52.35 s) is slightly **slower** than pure CPU (41.36 s): each
chemistry eval pays a host→device→host round-trip (fresh `ConcreteRArray` + XLA dispatch +
`Array` copy), which on a **CPU-backend** XLA outweighs any compiled-kernel win. The value
here is a correct, portable traced chemistry RHS; a **GPU** backend (where the round-trip is
amortized and the block-chemistry is massively parallel) is where this pays off.

## 2026-07-21 (later): both halves on XLA; merge landed in EarthSciAST; IIP findings

**The kernel-class merge is now IN EarthSciAST** (branch `feat/oop-kernel-class-merge`,
commit `4b6df081`): `src/tree_walk/oop_merge.jl` (`_merge_oop_acc_kernels`), run by
`_make_rhs_oop` after plan build, default-ON, `ESS_OOP_MERGE_DISABLE=1` restores the
unmerged build. Safety posture: every ineligible scope (overlapping out-slots, reduce
segments, unmergeable descriptor kinds, nested sub-subs, a failed group, an unvectorizable
merged plan) falls back to the original kernels; closed-fn payloads guarded by
`_check_fn_group_specs` (loud error on hash collision, never silent wrong numbers). Tests:
`test/oop_merge_test.jl` (pass fires; bit-identity vs disabled build and vs `f!`; live
forcing stays live through merged `_AccArrTblBox` tables; ForwardDiff bit-identical).
Full suite: **8,553 pass / 4 broken (pre-existing) / 0 fail** with the pass default-on.

**Second Reactant fix discovered and added to `rx_native_patch.jl`**: Reactant's
`make_tracer` capture walk recurses the whole object graph of every closure capture —
O(IR size) per `@compile`. The merged transport closure (1.06M `_Node`s) cost **~90+ min**
of walk (the prototype's 393 s compile had accidentally dodged it by keeping the IR in
module globals). Registering `_Node` / `_AccKernel` / `_OopAccPlan` / `_AccScratch` as
opaque leaf types for `make_tracer` (they hold only host data; precedent:
`make_tracer(::Union{ExceptionStack,MethodInstance})`) removes the walk: transport
`@compile` **456 s** in the driver. Durable home: EarthSciASTReactantExt.

**End-to-end, BOTH halves on XLA** (`run_split_reactant.jl`, now defaults
`RESEACT_XLA_PARTS=1,2`; per-half flow = in-package merge at build → driver-side merge
verify (now a no-op: 346→346, 13→13) → native-lowering patch → `@compile`), 60 s window
at 7×7×8:

```
BUILD 157.8 s   transport @compile 456.2 s   chem @compile 21.4 s
RESULT label=7x7x8 cells=392 nstates=5096 NS=13 transport=XLA chem=XLA
       solve_s=49.85 solve_secs=60 nT=2 nC=111 rcT=Success rcC=Success
       m=[1.514e+03,1.549e+03] O3=[3.9379e+01,3.9403e+01] ok=true
```

and at the FULL grid 7×7×72 (45,864 states):

```
BUILD 176.5 s   transport @compile 497.4 s   chem @compile 22.6 s
RESULT label=7x7x72 cells=3528 nstates=45864 NS=13 transport=XLA chem=XLA
       solve_s=70.47 solve_secs=60 nT=2 nC=109 rcT=Success rcC=Success
       m=[9.986e-01,5.017e+03] O3=[3.9379e+01,3.9459e+01] ok=true
```

Two things worth naming: the transport `@compile` is **essentially grid-independent**
(456 s at 8 levels → 497 s at 72; merged kernel count 346 at both, trace 311k sites at
both — the merge preserved the N-independence the build promises), and the full-grid
solve **matches the all-CPU reference to printed precision** (`m`/`O3` ranges and step
counts `nT=2 nC=109` identical to the `cpu_baseline` RESULT in the hybrid section) — the
both-halves-XLA solve computes the same answer as the interpreter, end to end. As with
the hybrid, wall-clock on a CPU-backend XLA (70.5 s) does not beat pure CPU (41.4 s):
each eval pays a host→device→host round-trip; the GPU backend is where this pays.

**Does the merge pay on the production `:inplace` path too?** (`rx_iip_merge_probe.jl`,
transport 7×7×8, `ESS_XCSE_DISABLE=1` both arms so kernels are self-contained):

- Merge applies cleanly to the IIP kernel list: 4,119 → 346, 0 blocked/failed, ~10 s,
  output **BIT-IDENTICAL** (maxabs=0.0, 5096/5096).
- Interpreted (codegen-off) IIP RHS: **678 → 313 ms/call (2.2×)**.
- Codegen (B1) source *generation*: 7.8 → 2.3 s (3.4×).
- The codegen tier's FIRST-CALL native compile needs **>35 GB** on this transport half in
  BOTH configs (stock OOM-killed at 34.6 GB, merged at 37.6 GB, under a ~40 GB Slurm
  cgroup shared with co-running jobs) — so that arm is unrankable here, and the cost looks
  lane-count-driven rather than kernel-count-driven. Follow-up question for EarthSciAST.
- **Constraint for the IIP landing**: the merge must run BEFORE xcse (B4) — xcse rewrites
  kernel invariant tiers into SCALAR-cache reads (`_NK_CACHED` with a foreign payload)
  which the merge signature does not model. Landing in progress on branch
  `feat/iip-kernel-class-merge` (hoist to `build.jl` before `_share_kernel_invariants!`).

## Files
- `rx_split_probe.jl` — split + build both `:oop` halves + `@compile` each (pointwise
  first). `RESEACT_MODEL` picks the grid; `RESEACT_SIGMA_CHECK=1` adds the Σ vs full check.
- `run_split_reactant.jl` — the hybrid end-to-end solve. `RESEACT_XLA_PARTS` (default `2`)
  chooses which halves attempt XLA; `RESEACT_SOLVE_SECS` the window.

## 2026-07-22: full-loop tracing — the entire adaptive solve as ONE XLA program

The purpose-built traced integrator (`rx_traced_integrator.jl`, ~330 lines) puts the
WHOLE adaptive solve loop — stages, FD block Jacobian, batched pivot-free 13×13 block
solves, hairer error norm, PI step controller, accept/reject — inside one Reactant
trace, with the time loop as a `stablehlo.while` (`Reactant.@trace while`). The stage
algebra (Rosenbrock23 ConstantCache, SSPRK43 ConstantCache), the error residual, and
the PIController formulas are copied from the installed OrdinaryDiffEq sources with
their queried per-algorithm defaults; the new code is the batched block solver, the
masked FD block Jacobian (species-major layout, asserted aligned), and the traced loop
skeleton. `rx_traced_smoke.jl` is a seconds-fast toy-ODE regression test.

**Chemistry half validated at 7×7×8** (`run_traced_chem.jl`, dt0=0.5, rtol=1e-4,
abstol=1e-9, same merged RHS in all arms):

```
HOST   solve:  10.14 s  nacc=109 nrej=5   (OrdinaryDiffEq Rosenbrock23 + BlockDiag LU)
REF    solve:  36.52 s  nacc=1884         (host at rtol=1e-7 — accuracy yardstick)
TRACED @compile: 135.9 s
TRACED solve:   1.60 s  nacc=103 nrej=4
maxrel(traced,host)=2.06e-05   errT(ref)=4.96e-04   errH(ref)=4.94e-04   ok=true
```

Same accuracy as the host solver (4.96e-4 vs 4.94e-4 against the tight reference), and
**6.3× faster than the host chemistry solve on the CPU backend** (1.60 s vs 10.14 s).
The earlier hybrid (forward pass on XLA, loop on host) was *slower* than pure CPU
because every RHS eval paid a host↔device round-trip; with the loop inside XLA that
overhead is gone — this is the payoff of full-loop tracing, before even touching a GPU.

Three `@trace while` lessons, each regression-covered by the smoke test:
1. **No host `if` on Number-ish values in the body** — `track_numbers` promotes every
   host number the body references (Bool ⊂ Number included) to a traced value, so a
   host branch throws "non-boolean (TracedRNumber{Bool}) used in boolean context".
   Resolve choices (e.g. clamp on/off) at trace time, outside the loop.
2. **The body/cond must capture NO traced values.** `Ops.while_loop` builds the while
   operands only from the loop variables the macro collects syntactically; traced
   values captured inside a closure the body calls (the RHS's params + forcing
   buffers) become extra region block args with no matching operands and the verifier
   rejects the module. Fix: thread every traced dependency through the loop-carried
   `aux` argument (`adaptive_solve(..., ctrl, aux)`; the body does `aux = aux`).
3. **One broadcast op per statement** in integrator code — Julia fuses broadcasts
   within an expression into a composed function that misses the native `elem_apply`
   fast path; single-op statements guarantee native stablehlo lowering.

(Host-reference gotcha: the BlockDiagonal `jac_prototype` only reaches `jac!` via
`SplitODEProblem(f, zerof!, ...)` — with a plain `ODEProblem` the solver hands `jac!`
a dense `Array`.)

`run_traced_window.jl` composes BOTH halves — transport SSPRK43 adaptive window then
chemistry ROS23 adaptive window, nonneg clamp standing in for PositiveDomain — into
ONE @compile'd program per Lie-Trotter window (each half's RHS appears exactly once in
the module regardless of step count).

**Both halves in ONE XLA program — validated at 7×7×8** (`run_traced_window.jl`,
transport SSPRK43 adaptive + chemistry ROS23 adaptive per Lie-Trotter window,
nonneg clamp in place of PositiveDomain):

```
HOST   window: 15.82 s  transport nacc=2 nrej=0  chem nacc=109 nrej=5
TRACED @compile: 3070.2 s  (~51 min, one-time; transport body = 4x 311k-site RHS trace)
TRACED window:  3.95 s  transport nacc=2 nrej=0  chem nacc=103 nrej=4
m=[1.514e+03,1.549e+03]  O3=[3.9379e+01,3.9403e+01]   (identical to the baseline RESULT)
maxrel_vs_host=2.06e-05  ok=true
```

The traced window is **4.0× faster than the host Lie-Trotter step on the CPU backend**
(3.95 s vs 15.82 s) and reproduces the baseline m/O3 ranges exactly. The 51-min compile
is dominated by TRACE time (linear in RHS-calls-per-body: 4× transport + 16× chem);
if it matters later, restructuring SSPRK43 around a single shared RHS call site would
cut the transport trace ~4×. Compile is per-grid but amortizes across every window and
every timestep thereafter.

**Full grid 7×7×72 (45,864 states) — same driver, one env var:**

```
HOST   window: 36.57 s  transport nacc=2 nrej=0  chem nacc=110 nrej=5
TRACED @compile: 3259.5 s  (vs 3070.2 s at 7×7×8 — compile is grid-independent)
TRACED window:  4.52 s  transport nacc=2 nrej=0  chem nacc=104 nrej=4
m=[9.986e-01,5.017e+03]  O3=[3.9379e+01,3.9459e+01]   (identical to the baseline RESULT)
maxrel_vs_host=4.87e-05  ok=true
```

**8.1× faster than the host window at full grid** (4.52 s vs 36.57 s), up from 4.0× at
7×7×8 — the traced window barely slows down as the grid grows 9× (3.95 → 4.52 s)
because XLA batches across cells, while the host RHS+LU cost scales with cell count.
Together with the grid-independent compile, the trend is exactly what you want for
production grids: pay ~54 min once per grid shape, then every 60 s window costs ~4.5 s
on CPU (and the GPU backend is untried headroom on top).
