# Why the build scales with the grid, and what to do about it

**Question (2026-09-07):** the 0.25°×0.3125° box arm (slurm 10395008) spent
**1932.7 s** in `BUILD` against **330.5 s** for the 2x2.5 arm (10386111) at an
**identical** model grid — 25×13×72, `nstates=304200` in both. And the full
0.25° CONUS probe (10395007) never printed `BUILD` at all: it was OOM-killed at
174 GB MaxRSS. Why does the build scale with the grid?

**Answer:** it scales along **two** axes, and neither is the compiled program.

* **A — the native FORCING grid**, at a fixed model grid. All of it is
  **decode**, and specifically one line of the build. Nothing downstream of the
  decode costs anything: `build_evaluator` at the box325 model grid is *faster*
  at 0.25° than at 2x2.5. Fix: windowed / projected reads
  (`INDEX_RANGE_READS.md`).
* **B — the MODEL grid.** ~0.0057 s per cell. A tier-ordering bug sends 14
  column sums to a per-cell tier; fixing it removes an O(#columns) IR term
  (EarthSciML/EarthSciAST#273, RHS bit-identical) but **costs no measurable wall
  time** (§B3). The term it does NOT explain is now **found and fixed**: the
  affine box processor treats a state gather from an array of a DIFFERENT SHAPE
  than the output (a latitude column, a staggered face, a surface field) as
  non-affine, so it cuts the grid into thousands of boxes and materialises a
  dense per-box slot table — 4.5 M entries, 289 per cell, at 18x12x72. §B4, and
  EarthSciML/EarthSciAST#278.

The XLA trace is innocent on both axes. Between 2x2.5 and 0.25° the traced
helper census is character-for-character identical (`fast=30426 helpers=776`,
same op histogram, same `top_helpers`), and `prepare_jacobian` is flat (141.8 s
vs 138.4 s).

**And the same answer holds for the LOOP, which matters more than the build.**
10395008's forward pass is 80,141 s against 2,354 s at the identical cell count,
and **98.3% of it is `refresh`** — 1,231.34 s per call, measured, against the
~800 s `SCALE_025.md` projected. `C.step` is resolution-independent to 0.7%; the
per-step transport RHS is flat to 2% across a 2,200× change in forcing array
size; and `T.step`'s 1.86x is memory pressure on fixed-size transfers, worth
0.35% of the pass. The refresh's own superlinearity (91.3x for a 63.4x globe) is
decode plus that same residency premium, measured at **+18.8%** by holding 55 GB
of ballast against a byte-identical decode. §C.

> **A ~590 s residual reported in the first version of this document DOES NOT
> EXIST.** It was the difference between a probe run on a warm cache and a
> driver run whose BUILD overlapped the 37.6 GB download of the same files.
> §A2 is the retraction and the measurement. Two mechanisms were refuted on the
> way — `DiscreteMaterializer.materialize!` and gather locality — and both
> refutations are recorded, because they were the leading hypotheses.

---

## A. The native forcing grid

### A1. One line of the build is ALL of the tax

`tools/adjoint_gradient.jl:317` seeds the live forcing buffers by sampling every
discrete provider once:

```julia
fld = EA._provider_const_field(EA.provider_sample(prov, T0), k)
```

Each provider decodes its netCDF file **whole** — every variable, all eight
records — and keeps two records of one variable. `tools/diag/decode_cost.jl`
times exactly that loop, with every file already cached, so the number is decode
and nothing else:

| res | native grid | one sample of all 15 providers | kept |
|---|---|---|---|
| 4x5 | 72×46 | 7.17 s | 31.0 MB |
| 2x2.5 | 144×91 | 18.69 s | 122.7 MB |
| 0.25° | 1152×721 | **1052.87 s** | **7774.3 MB** |

(0.25° measured by slurm 10400641; the two coarse rows on the login node.)

And measured *inside a faithful replica of the driver's own BUILD region*
(`tools/diag/build_driver_decomp.jl`, slurm 10404359 — same `ndays`, same
`inspect`, same statement order, warm cache, one process per arm):

| res | ndays | doc | ctor | **sample** | build1 | build2 | matrlz | **BUILD** |
|---|---|---|---|---|---|---|---|---|
| 2x2.5 | 1 | 54.72 s | 0.57 s | 23.08 s | 169.41 s | 57.26 s | 0.00 s | **329.98 s** |
| 2x2.5 | 4 | 53.91 s | 0.57 s | 23.13 s | 169.63 s | 56.25 s | 0.00 s | **328.47 s** |
| 0.25° | 1 | 54.07 s | 0.61 s | **1018.22 s** | 153.38 s | 54.41 s | 0.00 s | **1306.25 s** |
| 0.25° | 4 | 55.02 s | 0.60 s | **1027.00 s** | 157.43 s | 54.58 s | 0.00 s | **1319.95 s** |

The 2x2.5 row reproduces the driver's own `BUILD 330.46 s` to **0.15%**, which
is what makes the 0.25° rows admissible as the driver's number. Comparing the
two `ndays = 4` rows — the driver's exact configuration on both arms — the delta
is **991.5 s**, of which the provider sample is **1003.9 s**: **decode is 101%
of the whole resolution tax**, and everything downstream is slightly *negative*.

`ndays` is free (328.47 vs 329.98 s at 2x2.5; 1319.95 vs 1306.25 s at 0.25° —
within run-to-run scatter for a 4× longer cadence), so the driver's
`NDAYS = forcing_days_for(T0, T_END) = 4` is not a cost. Provider construction
fetches nothing; only `provider_sample` does.

### A2. RETRACTION: the ~590 s residual, and the two mechanisms it did not have

The first version of this section said ~590 s of the 0.25° build was unexplained
and needed both a fine native grid and a real model grid to appear — pointing at
something per-cell reading the global arrays, where gathers into a 7.77 GB array
lose the locality the same gathers into a 122 MB array keep. That was a good
hypothesis. It is wrong three times over.

**(i) `materialize!` is 0.01 s.** `DiscreteMaterializer.materialize!()` was the
prime suspect: the driver runs it inside the timed `BUILD` region and it is the
only build phase that is both per-cell and a reader of the forcing buffers.
Slurm **10404098** re-ran `build_res_window.jl` at `BRW_GRID=25x13x72` with that
phase timed. It costs **0.01 s** at every resolution and every arm. Refuted.

**(ii) `build_evaluator` is FLAT in the native grid at the real model grid.**
Same job, model grid fixed at the box325 grid where the residual was supposed to
live:

| res | arm | sample | param MB | doc | build1 | build2 | matrlz | total |
|---|---|---|---|---|---|---|---|---|
| 2x2.5 | full | 23.86 s | 122.7 | 54.24 s | 168.21 s | 56.59 s | 0.01 s | 303.52 s |
| 2x2.5 | win | 22.53 s | **3.5** | 53.82 s | 167.30 s | 54.89 s | 0.01 s | 299.40 s |
| 0.25° | full | 1025.91 s | 7774.3 | 56.91 s | **159.60 s** | 54.17 s | 0.01 s | 1297.24 s |
| 0.25° | win | 1037.42 s | **3.5** | 55.18 s | **155.59 s** | 52.80 s | 0.01 s | 1301.91 s |

With 63× the native cells and 63× the resident forcing, `build_evaluator` part 1
is **159.60 s against 168.21 s** — five percent *faster*, not slower. It is
faster for a mundane reason visible in the same lines: with 7.8 GB live, Julia's
GC fires less often (`gc 10.74 s` at 0.25° against `22.95 s` at 2x2.5). Carrying
whole-globe arrays through the build does not cost; it very slightly pays.

The windowing stays exact — the RHS hash is identical between `full` and `win`
in both split parts at both resolutions (`f23e0d933740ffae` / `cea04733f5da7671`
at 0.25°) and correctly *differs* between resolutions.

**(iii) The residual was the download.** The driver run that produced 1932.7 s
started at 20:13:22 on 2026-09-06. The cache manifests
(`$EARTHSCIDATADIR/v1/meta/*.json`, field `fetched_at`) say every one of the
twenty 0.25° blobs — 4 days × 5 collections, 37.6 GB — was fetched between
**20:12:04 and 20:48:44 CDT**, i.e. across that job's entire BUILD window; the
day-1 files its BUILD actually reads landed at 20:26:29 (A3dyn), 20:29:31
(A3cld) and 20:29:56 (A3mstE). Slurm **10395007** started 20:12:52 on another
node and spent the next 110 minutes in *its* build against the same
`/projects` cache before being OOM-killed. So that BUILD was competing with, and
partly blocked on, a 37.6 GB fetch into a shared cache with a second consumer.

On a warm, uncontended cache the same BUILD — same resolution, same model grid,
same `ndays = 4`, same `inspect` — is **1319.95 s** (§A1). 1932.68 − 1319.95 =
**613 s of download and contention**, and it is not a property of the model, the
grid, or the arrays. `build_driver_decomp.jl` prints a cache census
before and after and says explicitly whether a blob was written inside the timed
region, so this cannot recur silently.

**Lesson worth keeping:** a `BUILD` number measured on a run that is also
populating its own cache is a network measurement wearing a build's clothes.
Two independent probes (6x6x8 and box325) both said "flat downstream"; the
disagreement was with an un-reproduced single number, and the single number was
the one that was wrong.

### A3. Isolating array size from work — the controlled ladder

Because a resolution change moves three things at once (decode cost, resident
array size, model geometry), `tools/diag/forcing_size_ladder.jl` was written to
move exactly one. It decodes **once**, always at 2x2.5, then EMBEDS that same
halo window into a synthetic native array of whatever shape the rung asks for
and rebases `LON0`/`LAT0` onto the embedding offset. Across the ladder the data,
the equations, the geometry and the number of gathers are identical and **only
the size of the array being gathered from moves** — which is exactly "hold the
access count fixed and scale the array", the experiment that separates a
locality mechanism from a work mechanism. The RHS hash is the acceptance test
and must not move; it does not.

At the box325 model grid, one process per rung, transport RHS (`gT`) compiled
with the driver's own options and timed (slurm **10404593**):

| rung | synthetic native | forcing MB | build1 | RHS sha | @compile gT | **gT median** | **gT min** | `sync_forcing!` |
|---|---|---|---|---|---|---|---|---|
| win | 27×14 | 3.5 | 175.09 s | 18480c62f94b7dc5 | 2239.01 s | 15.194 ms | **14.296 ms** | 0.35 s |
| r1 | 144×91 | 122.7 | 173.81 s | **same** | 2406.32 s | 15.196 ms | **14.612 ms** | 0.42 s |
| r8 | 1152×721 | **7774.3** | 161.75 s | **same** | **8820.35 s** | 16.657 ms | **14.611 ms** | 1.88 s |

`18480c62f94b7dc5` is the same hash `build_res_window.jl` gets from the real
2x2.5 forcing, so the embedded ladder is bit-for-bit the model the driver runs —
and it is unchanged across a **2,200× change in the size of the array being
gathered from**.

Read off it:

* **The per-call RHS is flat.** `gT` min is 14.296 / 14.612 / 14.611 ms — 2% and
  no trend. The median moves +9.6% at r8, which is occasional allocation stalls
  in a 21.5 GB process, not a slower kernel; the floor does not move.
* **The gathers do not miss more.** `perf stat` over each whole process:
  `dTLB-load-misses` 2.637e9 / 2.659e9 / 2.845e9 (**+7.9% for 2,200× the
  array**) and the cache-miss RATE flat at 37.98% / 38.57% / 38.37%. That is the
  direct locality discriminator, and it says there is no locality effect to find.
  `page-faults` DO jump (40.6M / 43.8M / 296M) — that is first-touch of the
  7.78 GB allocation, paid once, not per call.
* **`build_evaluator` is flat here too** (175.09 / 173.81 / 161.75 s), reproducing
  §A2(ii) under full control.
* **What DOES scale is the COMPILE**: `@compile gT` 2239 → 8820 s, 3.9×, for a
  byte-identical program. XLA:CPU compilation slows down in a process holding
  7.78 GB of buffer operands (296M page faults ≈ 1.2 TB of page touching). This
  is the same effect the driver sees more mildly (`@compile ssp_step`
  228.3 → 281.0 s, `ssp_vjp` 932.8 → 1052.1 s) and it is a memory-pressure cost,
  not more program.

Smoke run at 6x6x8 agrees (one process, so only r1 vs r2 are comparable — the
first rung pays the JIT cold start): build1 36.33 / 35.06 s and gT 0.164 /
0.159 ms between 122.7 MB and 490.6 MB, RHS bit-identical.

### A4. The chemistry half already proves the fix works

Half the model already runs on windowed data. `tools/capacity_chem.jl:376` gives
each chemistry shard `zeros(Float64, dims[1], 1, 2, C + 1)`: its own C cells,
gathered per lane, never the globe. From the same two runs:

| | 4x5 | 2x2.5 | 0.25° |
|---|---|---|---|
| shard cells | 819 | 2925 | 2925 |
| shard build | 46.4 s | 54.4 s | **49.6 s** |
| shard RSS | 3.5 GB | 5.5 GB | **5.5 GB** |

The chemistry build is **flat in resolution** — 0.25° is *faster* than 2x2.5 at
the same shard size.

### A5. What to do

**Windowed and projected reads**, `INDEX_RANGE_READS.md` steps 1–2 — but note
that §A2 changes *why*. Slicing the arrays after decode (`win` arm above) buys
**2.6% of `build_evaluator`** and nothing else, because residency was never the
problem. The whole prize is on the reader side: 1003.9 s of the 991.5 s build
delta, and — far more importantly — the loop's refresh term (§C), which is 109 h
of a five-day 0.25° run.

**The decode PEAK survives, and it is what OOM-killed 10395007.** `provider
sample` reaches **30.3 GB RSS** at 0.25° to keep 7.77 GB — the whole-file decode
transient is ~4× what it retains. Slicing after decode drops RSS to 8.4 GB but
pays the 30 GB peak on the way. Only a reader that honours a selection avoids it.

---

## B. The model grid

With the decode subtracted, the driver's own `BUILD` numbers at NLEV=72 give
~**0.40 s per model column**, which carries to ~2.1 h for full 0.25° CONUS
(18,721 columns) with memory growing alongside — which is what OOM-killed
10395007 at 174 GB before it reached the `BUILD` line. `tools/diag/build_res_window.jl`
independently sees ~0.0058 s/cell between its 6x6x8 and 25x13x72 arms.

That per-column term is real. **The explanation this document gave for it was
wrong**, and the correction is worth more than the original claim.

### B1. What the 14 equations actually are

The cascade tally reports `percell_loop=14` for the transport half and **zero**
for chemistry. `ESS_STENCIL_DEBUG=1` names all fourteen:

* `Transport3D.dp_col`
* `Transport3D.pbl_wt`
* `Transport3D.pbl_sum_{CH2O, CH3O2, CH3OOH, CO, H2O2, HNO3, HO2, ISOP, NO, NO2, O3, OH}`

Every one is a plain mass-weighted column sum — `pbl_sum_O3` entire:

```json
{"lhs": "pbl_sum_O3",
 "rhs": {"op": "aggregate", "output_idx": ["gi", "gj"], "reduce": "+",
         "ranges": {"gi": {"from": "lon"}, "gj": {"from": "lat"},
                    "gk": {"from": "lev"}},
         "expr": dp[gi,gj,gk] * pbl_f[gi,gj,gk] * SuperFast.O3[gi,gj,gk]}}
```

**Two corrections to the earlier account.**

*They were never offered the affine lane, rather than declining it.*
`_compile_arrayop_equation!` computes `use_contraction_loop` **before** the
affine block and gates that block on `!use_contraction_loop`, so any reduction at
or above the length-8 floor bypasses the affine tier outright. The gather-wrapping
story (`_materialized_fill_equation` synthesizing `index(<def>, gi, gj)`, the
unwrap firing only for prefix scans) described the snapshot pinned in this repo's
checkout, which is ~80 commits behind upstream. Upstream `c5c6b7089` (2026-08-04)
already renamed `_scan_unwrap_identity_gather` to `_unwrap_identity_gather` and
widened it to any contracting producer, and closed the const-lane hazard
structurally — `LANE_CONST` now derives invariance from the resolved linear
index's stride vector, never from sampled corner values. That commit left a
"NOTE for later" saying the tier ordering was still wrong.

*The tier is a LOOP, not an unroll.* `percell_loop` is the contraction-loop
tier: one resolve + `_compile` per **output cell**, each O(1) in the reduction
length. So the term is **O(#columns) and independent of NLEV** — not the
`14 × 72 = 1,008 unrolled lowerings per column` this document previously
asserted. Measured removal is ~**63 node lowerings per column at both NLEV=8 and
NLEV=72**, which is what an NLEV-independent term looks like.

### B2. The fix works, and it does not buy what I claimed

`EarthSciAST-axisB`, branch `perf/contraction-tier-order` (commit `7f7964b33`,
based on `upstream/main`): let the contraction loop preempt the affine tier only
when it is the cheaper of the two — `#output cells < ∏|k…|` — and otherwise offer
affine first with the loop still behind it as the fallback.

| | pristine | patched |
|---|---|---|
| tally, part 1 | `affine=87 percell_loop=14` | `affine=101` (no `percell_loop`) |
| nodes, 216 columns | 145,249 | 131,111 |
| nodes, 640 columns | 175,353 | 133,231 |
| nodes, 325 cols × 72 lev | 153,628 | 133,128 |
| RHS `sha256(du)`, both parts | — | **identical at six grids** |

Node lowerings lose their O(#columns) term and go flat in the grid, and the RHS
is bit-identical at 6x6x8, 9x9x8, 12x12x8, 18x12x8, 40x16x8 and 25x13x72.

**But build wall time does not move**: 133.76 s → 130.90 s at 640 columns, and
117.05 s → 123.77 s at 216 columns (different nodes, so the sign there is noise).
Forty-two thousand removed lowerings cost no measurable time.

So the fix is correct, upstreamable and makes the IR volume grid-independent —
and it is **not** the ~0.40 s/column term. On an isolated synthetic shape it is
the whole build (node lowerings `4·#output cells` → a constant 9, build
allocation −12%), but on ReSEACT it is worth seconds.

### B3. What was therefore still open (and is now answered in §B4)

**The real per-cell build term was unattributed.** It is ~0.0057 s/cell,
resolution-independent (§A2 measured `build_evaluator` flat across a 63x native
grid at both 6x6x8 and 25x13x72), and it is what projects to ~2.1 h and the
174 GB OOM at full 0.25 CONUS. It is not the fourteen column sums.

Two things follow that do not depend on finding it:

1. **Tile the transport half across processes**, as `SCALE_025.md` 6 step 4
   argues for the loop. Whatever the per-cell term turns out to be, tiling
   divides the cells per process, so build and compile become O(tile) — which
   A3's shard table shows already working for the chemistry half.
2. Do not re-derive the per-column story from the cascade tally. `percell_loop`
   counts equations that took a per-cell TIER; it does not measure what that tier
   costs. This document asserted the connection and measurement refuted it.

### B4. FOUND: a state gather on its own grid is affine, and the affine tier did not know

**The term is not per-column and it is not the `var_map` name plumbing.** Two
things had to be measured to get there, and both refute a standing hypothesis.

**(i) It is per CELL, not per column.** Holding the columns fixed at 18x12 = 216
and moving only NLEV (`tools/diag/aktbl_ladder.jl`, one process per grid, part 1):

| grid | cells | columns | build |
|---|---|---|---|
| 18x12x8 | 1 728 | 216 | 65.58 s |
| 18x12x24 | 5 184 | 216 | 70.42 s |
| 18x12x72 | 15 552 | 216 | 93.68 s |

0.00203 s/cell along an axis on which the column count never moves. The column
axis at fixed NLEV (18x12x8 -> 40x16x8) gives 0.00445 s/cell. Both are real; the
"~0.40 s per column" of the first version of this document was 0.0058 s/cell x 72
levels read as a per-column law, from two points that confounded the two axes.

**(ii) `var_map` is REFUTED.** The name-keyed state plumbing — one constructed,
hashed `"Transport3D.m[13,7,72]"` string per state — was the leading hypothesis
for "O(cells) with a large constant and no IR". Phase timers put
`_build_state_layout` (which builds every cell name, `var_map`, and `u0`) at
**0.159 s at 1 728 cells, 1.72 s at 15 552 and 2.74 s at 23 400** — real, linear,
and 2.5% of the build. It is not the term.

**What it is.** `_BENCH_PHASE` (added in EarthSciML/EarthSciAST#278) attributes
the 25x13x72 build phase by phase, and `ESS_AK_TBL_DEBUG=1` names the lanes:

| phase | 18x12x8 | 18x12x72 | 25x13x72 |
|---|---|---|---|
| `affine_tier` | 33.0 s | 60.2 s | 91.9 s |
| ‥ `affine_box` (n boxes) | 25.3 s (8 360) | 41.5 s (7 170) | 60.5 s (7 760) |
| ‥ `affine_cuts` | 7.5 s | 18.4 s | 31.1 s |
| ‥ `mat_state_tbl` (n tables) | 0.2 s (14 784) | 3.6 s (36 375) | 5.1 s (40 142) |
| `class_merge` | 6.7 s | 10.1 s | 13.2 s |
| `oop_plan` | 2.2 s | 4.4 s | 6.2 s |
| `state_layout` | 0.16 s | 1.7 s | 2.6 s |
| **per-box slot-table ENTRIES** | **198 144** | **4 491 936** | **6 855 624** |

Every one of those entries is a STATE gather, and the log names them:
`Transport3D.dS_lat[j]`, `dphi_lat[j±n]`, `coslat_e[j]` (latitude columns read
from an (i,j,k) loop), `Mx[i,j,k]` / `My[i,j+1,k]` (staggered faces on
(NLON+1, NLAT, NLEV)), `pbl_wt[i,j]` / `PS[i,j]` (surface fields).

`_AK_STATE_AFFINE` models `u[oln + Δ]` — a gather from an array laid out exactly
like the output. All of the above are affine in the loop indices with the
ARRAY's own strides, so Δ = slot − output_slot moves at every cell. The affine
tier therefore (a) read a signature transition at every cell, so its
edge-inward cut scan never stabilised and it cut every axis up to the
`_AFFINE_MAX_DELTA_SEGS` cap — the box count grew with the grid — and (b) could
not prove the lane uniform over its box, so it materialised a dense per-box slot
table, one `_eval_recipe` and one stored `Int` per box cell per lane.
`_AK_CONST_BOX` and `_AK_FORCING_BOX` exist precisely because a const array or a
forcing buffer lives on its own grid; **there was no state counterpart.**

**The fix** (EarthSciML/EarthSciAST#278, stacked on #273) does two things. It
keys the cut signature on the lane SUBSCRIPTS rather than on Δ: a subscript that
is structurally affine in the loop indices can never open a cut and is skipped
outright, and any other one contributes its per-dim BACKWARD DIFFERENCE
`v(loop) − v(loop − e_d)` — constant wherever the subscript is affine, changing
exactly where a clamp / fold / wrap engages. (A first attempt keyed the deviation
from an affine model derived at a probe cell; that silently poisons the scan when
the probe sits on the clamped side of the transition, keys a change at every cell,
and past `_AFFINE_MAX_DELTA_SEGS` falls back and tables the lane anyway. The
pgather-table test caught it at N=32 and not at N=8 — worth remembering.) And it
lowers a loop-affine state lane through the existing `_AccStateTblBox` descriptor
addressed by an IDENTITY table over the variable's own slot block, shared
build-wide. No new descriptor kind, so no runner, codegen or Reactant-extension
change; the interpreted RHS is unchanged (18x12x8, 200 reps: min 31.85 -> 31.03
ms, median 43.08 -> 42.84 ms).

| grid | cells | cols | build (pre -> post) | boxes | table entries |
|---|---|---|---|---|---|
| 18x12x8 | 1 728 | 216 | 65.58 -> **43.94 s** | 8 360 -> **4 219** | 198 144 -> **0** |
| 25x13x8 | 2 600 | 325 | 64.30 -> **44.50 s** | 9 041 -> **4 220** | 342 064 -> **0** |
| 40x16x8 | 5 120 | 640 | 80.69 -> **53.33 s** | 11 084 -> **4 223** | 786 688 -> **0** |
| 18x12x24 | 5 184 | 216 | 70.42 -> **52.90 s** | 7 170 -> **4 219** | 1 364 256 -> **0** |
| 18x12x72 | 15 552 | 216 | 93.68 -> **76.21 s** | 7 170 -> **4 219** | 4 491 936 -> **0** |
| 25x13x72 | 23 400 | 325 | 131.15 -> **101.75 s** | 7 760 -> **4 220** | 6 855 624 -> **0** |

Note what the PRE column does between 12x12x8 and 18x12x8: **92.4 s at 1 152
cells against 70.7 s at 1 728** — the old build was not even MONOTONE in the
grid, because each axis was cut per index only while its length stayed under
`_AFFINE_MAX_DELTA_SEGS = 16`, and at 12x12 both horizontal axes qualify. After
the fix it is monotone.

Build ALLOCATION (which, unlike RSS, does not depend on when GC last ran) falls
32.4 -> 21.0 GiB at 12x12x8, 19.9 -> 15.8 GiB at 9x9x8, 36.0 -> 30.9 GiB at
18x12x8. Peak RSS moves the same way but noisily (6.8 -> 6.0 GB at 25x13x72 in
one paired run, 6.8 -> 7.6 GB in another): it is a high-water mark taken after
the build, so it reports where GC happened to be, not what the build needed.

Box count is **flat in the grid** afterwards, `_cell_ckey!` calls **saturate**
(20 817 / 25 371 / 25 371 / 26 329 over a 13.5x cell range), and no per-box table
is materialised at all. The RHS `sha256(du)` is **identical to the kill-switch
build for both split parts at ten grids** (6x6x8, 9x9x8, 12x12x8, 18x12x8,
25x13x8, 40x16x8, 6x6x72, 18x12x24, 18x12x72, 25x13x72) with the node-lowering
count unchanged.

**The residual, honestly.** The build is still not O(1) in the grid: 101.75 s at
23 400 cells against 43.94 s at 1 728 — 0.00267 s/cell over the whole ladder,
against 0.00303 before. The per-cell SLOPE improved only 12%; what improved is
the constant (−20 to −44 s), the MONOTONICITY, and the character of what is
left. Box count, `_cell_ckey!` count, spine-template count and node lowerings
are all flat now, so the residual is not a per-cell algorithm; it is the cost of
the same fixed work in a process whose live heap grows with the grid.
`_BENCH_PHASE_GC` measures **27% of it as GC** (19.9 s of the 73.8 s affine tier
at 25x13x72), and what grows the heap is the `:oop` emitter's per-lane
descriptor vectors (`_build_oop_desc_vectors` materialises one `Int` or
`Float64` per lane per descriptor, even for a pure arithmetic progression) —
which is exactly the "per-lane DATA may grow, at most linearly" carve-out that
`pkg/EarthSciAST.jl/test/grid_invariance_test.jl` writes into the pinned
property. Making those descriptors lazy is the next lever and was not attempted.

Note on stale baselines: `logs/bldscale-10120474.out` (2026-08-24) shows 245–440 s
part-1 builds where the same ladder now gives 50–86 s. Do not mix it with current
numbers.

---

## C. The loop at 0.25° is refresh, and refresh is bigger than anyone priced it

Slurm 10395008 finally printed its forward-pass decomposition. 576 macro steps,
0.25°, box325 grid, against the 2x2.5 arm (10386111) at the **identical** cell
count:

| term | 2x2.5 | 0.25° | ratio |
|---|---|---|---|
| forward pass | 2354.65 s | **80141.39 s** | 34.0x |
| `refresh` | 862.77 s / 64 = 13.48 s/call | **78806.05 s / 64 = 1231.34 s/call** | **91.3x** |
| `C.step` | 44.44 ms/call | **44.77 ms/call** | **1.01x** |
| `T.step` | 89.26 ms/call | 165.60 ms/call | 1.86x |
| ‥ `T.step.exec` | 87.23 ms | 130.81 ms | 1.50x |
| ‥ `T.step.read` | 1.23 ms | 11.89 ms | 9.7x |
| ‥ `T.step.upload` | 0.81 ms | 22.90 ms | 28.3x |
| `host.ctrl+tape` | 0.53 ms/call | 0.73 ms/call | 1.4x |
| GC | 49.53 s, 903 GB alloc | 2582.70 s, **39,908 GB alloc** | 44x |

**Refresh is 98.3% of the forward pass.** Everything else together is 1,335 s of
80,141 s. Whatever else is true, this is the only term with a cost model.

### C1. `C.step` is the control, and it passes

44.77 vs 44.44 ms/call — **resolution-independent to 0.7%** at identical cells.
The chemistry shards receive per-lane *gathered* forcing
(`capacity_chem.jl:414 gather_forcing!`, C cells not the globe), so this is the
same computation reading a small array in both arms. It is the clean control the
whole-globe question needed: same cells, same chemistry, no penalty.

### C2. `T.step`'s 1.86x is memory pressure, NOT gather locality

Transport reads the global forcing arrays directly, so its 1.86x is the obvious
candidate for the loop-side version of the build hypothesis. It is not, and the
decomposition says so before any probe does:

* `T.step.upload` is `RX.ConcreteRArray(u)` and `T.step.read` is `Array(r[1])`
  (`adjoint_gradient.jl:734-744`) — moving the **state vector**, 304,200 × 8 B =
  2.4 MB, **the same 2.4 MB in both arms**. They cost 28.3x and 9.7x more at
  0.25°. A fixed-size allocate-and-copy cannot become 28x slower because of
  anything in the forcing arrays; it becomes 28x slower because the process
  around it holds 62–88 GB (four resident copies of the 7.77 GB forcing: host,
  `dev_bufs[1]`, `dev_bufs[2]`, `dev_bufsJ`) with eight 5.5 GB workers beside it.
  `read + upload` is 34.79 vs 2.04 ms — **43% of the whole T.step delta**, and it
  never touches a forcing array.
* The remaining `exec` 1.50x is measured directly and **is not in the RHS**: the
  §A3 ladder compiles the transport RHS at the box325 grid against 3.5 MB,
  122.7 MB and 7.78 GB of forcing and gets **14.296 / 14.612 / 14.611 ms**, with
  `dTLB-load-misses` flat to 7.9%. The gathers are fine; `ssp_step`'s four stages
  allocate their own intermediates, and that allocation pays the same residency
  tax as the state transfer.

And the size of the prize: T.step is **278.20 s of 80,141 s = 0.35%**. Removing
the whole 1.86x would buy 128 s of a 22.3 h pass. It is worth understanding, not
worth optimising.

### C3. The refresh superlinearity — 91.3x for a 63.4x globe

The pieces, all measured:

| | value | source |
|---|---|---|
| standalone decode of all 15 providers, 0.25°, warm | 1052.87 s | `decode_cost.jl`, slurm 10400641 |
| the same sample inside a driver-faithful build, quiet node | 1018.22 / 1027.00 s | `build_driver_decomp.jl`, slurm 10404359 |
| **in-run `refresh`** | **1231.34 s** | 10395008's own timer |

so the decode is ~1,020 s and the refresh costs **~210 s (+21%) more than the
decode it contains**. What a `refresh` does beyond `provider_sample`
(`adjoint_gradient.jl:850`), with what each is worth:

* `merged_param[k] .= …` — the sample returns a fresh 7.77 GB which is then
  broadcast into the live buffers and dropped: ~15.5 GB of traffic, ~3–8 s, and
  7.77 GB of immediate garbage.
* `materialize!` — **0.01 s** (§A2).
* `push_forcing!` → `sync_forcing!` for part 1, part 2 **and** the band Jacobian
  = three host→device copies of the whole forcing set. The §A3 ladder times one
  such set at 7.78 GB: **1.88 s**, so ~5.6 s.
* `shard_refresh_all!` → `gather_forcing!` per shard: 2,925 lanes × 2 records ×
  15 arrays × 8 shards ≈ 700k scattered reads out of 7.77 GB. This *is* a
  per-cell gather into whole-globe arrays — and at ~100 ns a miss it is ~0.07 s.
  Real, and irrelevant.

That names ~10–15 s of the 210 s. The rest is **residency**, and that is now
measured rather than inferred. `tools/diag/build_driver_decomp.jl` grew a
`BDD_BALLAST_GB` knob: a deliberately sized `Vector{Float64}`, touched so its
pages are really resident, held live for the whole process. Nothing else moves —
same node, same files, same 15 providers, byte-identical decode (slurm
**10407717**, two arms, one process each):

| ballast | RSS during sample | **provider sample** | GC in that phase | BUILD |
|---|---|---|---|---|
| 0 GB | 8.7 GB | 1046.98 s | 20.60 s | 1341.51 s |
| **55 GB** | **73.3 GB** | **1244.14 s** | **43.23 s** | 1535.43 s |

**+18.8% on the identical decode, from resident memory alone**, with GC in that
phase more than doubling. And the absolute number lands on top of the driver's:
the ballasted sample is **1244.14 s against the in-run refresh's 1231.34 s —
1.0% apart**, at a residency (73.3 GB) inside the driver's own 62–88 GB.

So the refresh decomposes, with no term left over:

| | s | how |
|---|---|---|
| decode of 15 providers, quiet process | ~1047 | measured, `build_driver_decomp` ballast=0 |
| + residency premium at driver-like RSS | ~197 | measured, ballast=55 GB |
| + `.=` copies, 3× `sync_forcing!`, shard gather | ~10–15 | measured / bounded above |
| = | **~1257** | vs **1231.34** measured in-run |

The mechanism is the same one behind §C2: the forward pass allocates
**39,908 GB** — ~620 GB per refresh, the decode's own Float32→Float64 and
`permutedims` churn — and every allocation in a 62–88 GB process is dearer.
Julia's GC pays 2,582.70 s of it directly.

**What a live `refresh` is made of**, from `perf record` on the running process
(400 s, 20,356 samples, all on one thread; refresh is 98.3% of the pass, so this
IS the refresh):

| share | where |
|---|---|
| 33.95% | `libz` — `inflate_fast` 29.17%, `adler32_z` 4.13%, `inflate_table` 0.43% |
| 53.40% | Julia JIT, 40.2% in ONE loop: `Array(v)` + `_to_file_order` — the Float32→Float64 convert and the file-order `permutedims` of `read_native(::NetCDFReader, …)` |
| 8.77% | Julia GC collecting that |
| 2.9% + 0.7% | kernel, `memmove` |

No XLA anywhere. Corroborated without instrumentation: over 300 s the driver
burned 32,056 CPU ticks and the eight shard workers burned **0**; cumulative
`rchar` is 1.639 TB, and one refresh re-reads 29.05 GB of file (2 providers on
I3 2.26 GB, 4 on A3dyn 3.80 GB, 6 on A1 1.06 GB, 1 on A3cld 1.59 GB, 2 on
A3mstE 0.69 GB).

### C4. The five-day projection gets worse, not better

`SCALE_025.md` projected ~800 s/refresh; a later revision cut the five-day
refresh term from ~94 h to ~67 h by carrying the 0.721 in-run/decode ratio
measured at 2x2.5. **Neither survives.** At the measured 1231.34 s/call, 320
refreshes is **~109 h**, worse than either estimate, and it is ~98% of a
five-day 0.25° forward pass rather than 18.8%.

---

## D. What the fix is worth, now that the term is measured

Refresh at 0.25° decodes **327 variable-days to keep 15**:

| collection | file | data vars | providers on it | variable-decodes per refresh |
|---|---|---|---|---|
| A3dyn | 3.80 GB | 5 | 4 (U, V, OMEGA, RH) | 20 |
| A1 | 1.06 GB | 47 | 6 (PBLH, TS, Z0M, USTAR, SWGDN, HFLUX) | **282** |
| A3cld | 1.59 GB | 7 | 1 (CLOUD) | 7 |
| A3mstE | 0.69 GB | 5 | 2 (PFLCU, PFLLSAN) | 10 |
| I3 | 2.26 GB | 4 | 2 (PS, T) | 8 |

~27.4 G decoded cells ≈ 219 GB per tick, independently reproducing the 211 GB of
`INDEX_RANGE_READS.md` §1. The cause is confirmed in the pinned checkout:
`read_native(::NetCDFReader, path)` (`EarthSciIO/julia/src/readers.jl:122`)
declares **no keyword arguments at all**, so `reader_option_keys` reports none,
`Provider._load` cannot push `variables` down, and every provider decodes its
whole file and `_select`s afterwards.

With `variables` pushdown alone — **step 1**, the twenty-line one already open as
a PR — each provider decodes its own variable: ~4.0 G cells ≈ 32 GB, i.e.
**6.9x off the single largest term in the whole run**. Refresh 1231 s → ~180 s;
the 48 h forward pass 21.9 h → ~3.2 h; the five-day refresh term 109 h → ~16 h.
No spec change, no window, no rechunk. It also cuts the ~620 GB/refresh of
allocation by the same factor, which is where the 2,582 s of GC and a large part
of the residency tax in §C2/§C3 come from.

Revised ranking at 0.25°:

1. **step 1 (`variables` projection)** — 6.9x of the largest term in the run,
   already written. Do this first, before any of `SCALE_025.md` §6's steps 2–5:
   at 0.25° they are optimising the 1.7% of the pass that is not refresh.
2. **repo-side: one provider per collection, fanned out to variables** — most of
   step 1's win with *no* upstream change (six A1 providers each decoding 47
   variables become one). `INDEX_RANGE_READS.md` §5 already names it.
3. **step 3a (netCDF→zarr transcode on ingest)** — the only lever that touches
   the 34% of the profile that is `inflate`, because the 0.25° chunk layout
   `[1, 1, 721, 1152]` means a horizontal window cannot reduce decompression.
4. **step 2 (decode-time window)** — buys retention, the 30.3 GB decode peak that
   OOM-killed 10395007, and the residency that §C2 shows taxing every fixed-size
   transfer in the process. Less wall time at 0.25° than at 4x5.

---

## Reproducing

```bash
# the driver's OWN BUILD region, decomposed, warm cache, one process per arm.
# Prints a cache census before/after so a download inside the timed region
# cannot masquerade as build time -- which is exactly what happened to 10395008.
sbatch tools/diag/build_driver_decomp.sbatch            # slurm 10404359

# the resolution axis at a FIXED model grid, full vs windowed forcing arrays.
# Use the _split form: one process per arm, so no arm inherits another's JIT
# warm-up. BRW_GRID=25x13x72 is the box325 grid (slurm 10404098).
BRW_GRID=25x13x72 sbatch --export=ALL tools/diag/build_res_window_split.sbatch

# array size vs work, controlled: same data, same gathers, same equations,
# only the size of the array being gathered from moves. RHS hash must not move.
FSL_GRID=25x13x72 FSL_STAGES='build,loop' FSL_RUNGS='win,r1,r8' \
  sbatch --export=ALL tools/diag/forcing_size_ladder.sbatch     # slurm 10404593
# NB: pass these through the environment with --export=ALL, not inside
# --export=...,FSL_RUNGS=win,r1,r8 -- sbatch splits that list on the comma.

# does the decode slow down purely because the process is big? Same node, same
# files, same decode, only resident memory moves (arm = res:ndays:ballast_GB).
BDD_ARMS='0.25x0.3125:4:0,0.25x0.3125:4:55' \
  sbatch --export=ALL tools/diag/build_driver_decomp.sbatch     # slurm 10407717

# decode cost per provider, per resolution
julia --project=run-model-jl tools/diag/decode_cost.jl            # 4x5, 2x2.5
sbatch --export=ALL,RESEACT_DECODE_RES=0.25x0.3125 tools/diag/decode_cost.sbatch

# the model-grid axis, and which equations fall off the affine lane
ESS_STENCIL_DEBUG=1 BS_GRIDS=6x6x8 BS_PARTS=1 \
  julia --project=run-model-jl tools/diag/build_scaling.jl 2>&1 | grep DECLINED

# WHERE the model-grid seconds go: per-phase wall + GC timers, the cascade
# tally, and the per-box slot-table attribution log, on one grid ladder.
# Needs an EarthSciAST carrying `_BENCH_PHASE` (EarthSciML/EarthSciAST#278);
# `envs/buildperf` points at the clone that does. NB pass the grid list through
# the ENVIRONMENT -- `--export=ALL,AKT_GRIDS=a,b` is split on the comma by
# sbatch and silently runs only the first rung.
AKT_GRIDS='6x6x8,18x12x8,18x12x24,18x12x72' \
  sbatch --mem=48G --export=ALL tools/diag/aktbl_ladder.sbatch   # slurm 10423120
# the same ladder with the fix off, i.e. the pre-fix baseline:
ESS_LANE_AFFINE_KEY_DISABLE=1 ESS_STATE_BOX_DISABLE=1 \
  AKT_GRIDS='6x6x8,18x12x8,18x12x24,18x12x72' \
  sbatch --mem=48G --export=ALL tools/diag/aktbl_ladder.sbatch   # slurm 10423122

# RHS bit-identity across the fix, both split parts, six grids
RESEACT_RXENV=$PWD/envs/buildperf AB_ARMS=pre,base AB_PARTS=1,2 \
  AB_GRIDS=6x6x8,6x6x8,9x9x8,12x12x8,18x12x8,6x6x72 \
  julia --project=envs/buildperf tools/diag/axisb_ab.jl

# what a live run is actually doing (no instrumentation, no restart)
perf record -F 49 -p <driver pid> -o /scratch.local/$USER/p.data -- sleep 400
perf report -i /scratch.local/$USER/p.data --stdio --no-children --sort dso,symbol
awk '/rchar/{print $2}' /proc/<driver pid>/io    # /29.05 GB = refresh passes
```
