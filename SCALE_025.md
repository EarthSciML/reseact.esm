# Making a five-day 0.25° CONUS gradient tractable on CPU

**Question (2026-09-06):** the 0.25°×0.3125° CONUS adjoint projects to ~16–23
days of loop for a five-day window. What makes it tractable without GPUs?

**Answer:** three of the four dominant terms are already-identified levers, and
**two of them were priced and rejected at 4x5 for a reason that does not hold at
0.25°** — a fixed ~3 ms per-call floor that is 23% of a 4x5 chemistry call and
1.4% of a 0.25° one. The lever ranking inverts with resolution. Combined, the
ladder below reaches ~1 day of loop, inside the 3-day partition limit, and makes
compile time and memory O(tile) rather than O(domain), which is what running
past CONUS requires at all.

Everything here is measured at 4x5 and 2x2.5 (slurm 10386109–12, five-day and
48 h arms) and extrapolated on fitted exponents; jobs 10395007 (full 0.25°
CONUS, 3 macro steps) and 10395008 (0.25° spacing, 325-column box, 48 h) are
running to replace the extrapolated per-step constants with measured ones.

---

## 0. MEASURED CORRECTIONS (2026-09-07, slurm 10395008)

10395007 was OOM-killed; **10395008 completed its forward pass** and its
decomposition changes this document's ranking. Full working in
`BUILD_SCALING.md` §C.

**(a) `refresh` is the whole run, and it is 54% dearer than projected.**
Measured **1231.34 s/call** (78,806.05 s over 64 calls) against the ~800 s below.
That is **91.3x the 2x2.5 refresh for a 63.4x globe** — superlinear, and dearer
than the 56.3x that standalone decode predicts. Refresh is **98.3% of the 0.25°
forward pass**: 78,806 s of 80,141 s, everything else together 1,335 s. The
five-day term (320 refreshes) is therefore **~109 h**, not the 80 h in §1 and
not the ~67 h a later revision assumed by carrying the 0.721 in-run/decode ratio
measured at 2x2.5. That ratio does not carry: it is 1.17 at 0.25°.

**(b) The per-step cost is resolution-INDEPENDENT at fixed cell count.** At an
identical 25x13x72 grid, `C.step` is **44.77 ms/call at 0.25° against 44.44 ms
at 2x2.5** — 1.01x. The chemistry shards read per-lane *gathered* forcing
(`capacity_chem.jl:414`), never the globe, and it shows. The transport RHS,
compiled and timed against 3.5 MB / 122.7 MB / 7.78 GB of forcing at that same
grid, is **14.296 / 14.612 / 14.611 ms** with `dTLB-load-misses` flat to 7.9%
(`tools/diag/forcing_size_ladder.jl`, slurm 10404593). So the `e` exponents in
§1 are exponents in CELLS alone; refining the native grid at fixed cells costs
the model nothing.

**(c) `T.step`'s 1.86x is residency, not the grid, and is not worth chasing.**
165.60 vs 89.26 ms/call, but 43% of the gap is `T.step.read` + `T.step.upload` —
which move the **same 2.4 MB state vector** in both arms and cost 9.7x and 28.3x
more because the process holds 62–88 GB (four resident copies of the 7.77 GB
forcing) with eight 5.5 GB workers beside it. T.step is 0.35% of the forward
pass; the whole 1.86x is 128 s of 22.3 h.

**(d) The ordering in §6 inverts.** At 0.25° refresh is ~98% of the run, so
steps 2–5 optimise the other 1.7%. `read_native(::NetCDFReader, path)` in the
pinned EarthSciIO declares **no keyword arguments at all**, so `variables` never
reaches the reader and every one of the 15 providers decodes its whole
collection file: **327 variable-days decoded to keep 15** per refresh. The
`variables` projection alone is **6.9x** off the largest term in the run
(refresh 1231 s → ~180 s, the five-day term 109 h → ~16 h), with no window and
no rechunk.

---


## 1. Where the time goes, and how each term scales

Per-call cost between the two measured resolutions (3.571x the cells), fitted as
cost ∝ cells^e, then carried to 0.25° CONUS (18,721 columns = 1,347,912 cells =
57.6x the 2x2.5 window; the globe at 0.25° is 63.4x the 2x2.5 globe):

| term | 4x5 | 2x2.5 | e | 0.25° projection |
|---|---|---|---|---|
| C.step | 13.2 ms | 45.4 ms | 0.97 | 2.3 s/call |
| T.step | 18.6 ms | 89.3 ms | 1.23 | 13.1 s/call |
| T.vjp | 332 ms | 1072 ms | 0.92 | 44.8 s/call |
| refresh | 3.74 s | 13.48 s | 1.01 (of the **globe**) | ~800 s/call — **measured 1231.34 s, §0(a)** |

Five-day loop (1440 macro steps), from the 2x2.5 five-day arm's 6 h 41 m:

| component | 2x2.5 5-day | share | 0.25° 5-day |
|---|---|---|---|
| chemistry (step + replay + VJP) | 13,052 s | 54.3% | **209 h** |
| transport (step + replay + VJP) | 6,136 s | 25.5% | **98 h** linear / **249 h** at e=1.23 |
| forcing refresh | 4,526 s | 18.8% | **80 h** → **109 h measured** (§0a) |
| host (tape, masks, ckpt) + GC | ~310 s | 1.3% | ~5 h |
| **total** | **24,039 s** | | **392–543 h (16–23 days)** |

The transport exponent is the one real uncertainty. 1.23 is a cache-residency
artefact of the 4x5→2x2.5 transition (a 4x5 state is 680 kB and L3-resident; at
2x2.5 the intermediates are not), so the marginal exponent should fall back
toward 1.0 once everything is memory-resident. Job 10395007 settles it.

Setup is a separate problem: fitted exponents give build 18 min,
`prepare_jacobian` 30 min, `@compile ssp_step` 23 min and **`@compile ssp_vjp`
8.3 h** — about 10 h before the loop starts, and `ssp_vjp` alone at e=0.85 is
what makes iteration at this scale impossible today.

## 2. Why the per-call floor inverts the lever ranking

`DIFFERENTIABILITY_PLAN.md` §6 records two negatives, both correct as measured:

* **more chemistry shards** — "N=13 gains nothing (each call is near the ~3 ms
  per-call floor, round-trip 30%)" at 4x5;
* **per-bucket adaptive stepping** — "0.66x wall at CONUS, priced ceiling 1.69x",
  paused.

Both lost to a *fixed* per-call cost, not to anything about the algorithm. That
cost is ~3 ms per program execution regardless of grid. Against it:

| | cells/shard-call | C.step per call | floor share |
|---|---|---|---|
| 4x5, 8 shards | 819 | 13.2 ms | 23% |
| 2x2.5, 8 shards | 2,925 | 45.4 ms | 6.6% |
| 0.25°, 8 shards | 168,489 | 2,318 ms | 0.13% |
| 0.25°, 100 shards | 13,479 | 209 ms | 1.4% |

At 0.25° a shard can be split 12.5x further and still carry 4.6x the work per
call that a 2x2.5 shard carries today. The same arithmetic revives bucketing:
K=32 buckets raise the number of executions ~7.3x (sum over buckets of their own
step demand, vs one global ladder) while each execution shrinks 32x, so the
floor costs ~1.4 s/window against a chemistry budget of ~10.7 s/window — 13%
overhead for a measured 3.41x. At 4x5 the same split cost more than the whole
chemistry budget, which is exactly the 0.66x that was observed.

**Bucketing is already priced at the target grid**, by `tools/diag/cell_stiffness.jl`
and the diurnal replay report in `logs/stiffdiurnal/replay_report.txt` (288
windows, full diurnal cycle, CONUS):

| policy | K=8 | K=16 | K=32 | K=64 | K=128 |
|---|---|---|---|---|---|
| prev (use last window's demand) | 2.28x | 2.87x | **3.41x** | 3.91x | 4.36x |
| oracle | 3.28x | 4.38x | 5.15x | 5.61x | 5.86x |
| spatial (geographic buckets) | 1.61x | 1.77x | 1.86x | 1.91x | 2.09x |

per-cell ideal (infinite buckets, perfect prediction) 6.09x; log-demand
window-to-window autocorrelation median 0.998, so `prev` is a near-free
predictor (median penalty vs oracle 1.063, worst 4.77x at terminator crossings).
Note `spatial` is much weaker than `prev` — bucket by *demand*, not geography.

Bucketing also removes a superlinearity: the global controller takes the
min-over-cells step, so adding cells can only lower dt. The 209 h chemistry
projection above assumes the 2x2.5 step count carries over unchanged, which is
optimistic by exactly this effect.

## 3. Transport is running on one core out of forty

Two measurements, both already in the repo:

* `tools/diag/profile_threads.sbatch` / slurm 10104808 vs 10105170 — the ros23
  step is **1.43x faster on 16 cores than on 1** (~9% parallel efficiency): the
  programs are thousands of elementwise passes over slabs too small for XLA:CPU's
  loop emitter to split.
* `logs/execovl-10105948.out` — concurrent executions *inside one process* do
  not overlap either (8 concurrent = 1.34x). PJRT:CPU serialises them.

Together those say the only unit of CPU parallelism available is **a separate
process running its own execution of the same program** — which is precisely
what `tools/shard_chem.jl` does for chemistry, and why it works. Transport
currently has none of it: one program, one process, effectively one core, while
39 sit idle. That is the 98–249 h term.

The fix is the same trick with a halo. PPM is a stencil, SSPRK43 is explicit, so
cutting the window into equal tiles and exchanging halo faces on the host between
stages is exact for the state (the only reassociation is the error-norm
reduction, summed on the host — the identical argument that made chemistry
sharding bit-identical). Equal tiles means **one** compiled program executed on
N buffers, so it reuses the shard machinery wholesale, and the emitter already
routes stencil reads through an extended buffer (`ue`), so a halo is not a new
concept.

The secondary benefit is larger than the speedup: the transport program is then
compiled at *tile* size. `@compile ssp_vjp` stops scaling with the domain — an
8.3 h projected compile becomes a fixed ~15 min, and stays there at any grid.

## 4. Refresh is 80 h of decoding data the run never reads

At 0.25° the CONUS window is **2.25% of the globe**, and refresh reads the globe:
~800 s per refresh × 320 refreshes (160 forward + 160 backward) = 80 h. This is
the lever already in flight — see `INDEX_RANGE_READS.md` and the two open
EarthSciIO PRs (decode-time `select` in all three tracks, projection pushdown in
the Julia netCDF reader). The measured chunk layout (`[1, 1, 721, 1152]` — one
whole global level per chunk) caps what decode-time selection alone can win on
the horizontal, which is why that document's step 3 (rechunk/transcode) is
conditional on the arm now running as job 10395008.

**Newly found while prefetching (2026-09-06):** `EarthSciIO`'s HTTP transport has
a 90 s *total* per-request cap (`EARTHSCIIO_HTTP_TIMEOUT`, transport.jl:90), and
0.25° A3dyn is 3.80 GB/day — the fetch dies at 3.49 GB. Any 0.25° run must set
`EARTHSCIIO_HTTP_TIMEOUT` (a longer cap is enough; upstream the cap should be
idle-based, since the low-speed abort already handles stalls). Data volume itself
is not a constraint: 9.4 GB/day, 38 GB for four days, 61 TB free on /projects.

## 5. Memory

1440 checkpoints × 17,522,856 states × 8 B = **188 GB** in RAM against a 184 GB
node — the five-day run does not fit as written. Three options, in order of
preference:

1. **Fall out of decomposition for free.** Once transport is tiled and chemistry
   sharded across processes, each process checkpoints only its own slice: 188 GB
   / 100 processes = 1.9 GB each, 38 GB per node across 5 nodes. No disk, no
   extra passes, and O(tile) at any grid — the property the standing rule about
   scale-hostile storage actually asks for.
2. **Spill to `/scratch.local`** (752 GB free, node-local): 188 GB written once
   and read once, ~7 min total, O(1) RAM. Simplest thing that works today.
3. Two-level Revolve — costs an extra forward pass (~+25%); only if 1 and 2 are
   both unavailable.

## 6. The ladder

Applied to the pessimistic 543 h baseline (transport at e=1.23):

| step | what | C | T | R | H | total |
|---|---|---|---|---|---|---|
| baseline | | 209 | 249 | **109** | 5 | **572 h** |
| +1 | windowed/projected reads (§0d) | 209 | 249 | 3.6–16 | 5 | **466–479 h** |
| +2 | shards 8 → 100 across 5 nodes | 19.7 | 249 | 3.6 | 5 | **277 h** |
| +3 | bucketed dt, K=32 `prev` (3.41x) | 5.8 | 249 | 3.6 | 5 | **263 h** |
| +4 | transport tiles (20x of 40 lanes) | 5.8 | 12.4 | 3.6 | 5 | **27 h** |
| +5 | host/GC terms threaded | 5.8 | 12.4 | 3.6 | 1.2 | **23 h** |

Ordering note: 1–3 are worth little on their own because transport dominates
after them. **Step 4 is the one that decides whether this is possible**, and it
is also the step that fixes the 8.3 h compile and the 188 GB of checkpoints. Do
it first, or at least concurrently with the reader work.

> **Amended 2026-09-07 (§0).** That ordering was written against a ~800 s
> refresh. At the measured 1231.34 s it is **step 1 that decides whether this is
> possible** — at the box325 grid refresh is 98.3% of the forward pass and
> transport is 0.35%. Step 4 is still what makes compile and memory O(tile), and
> still what the full CONUS column count needs; but on wall time at 0.25° the
> reader comes first, and it is the cheapest of the five to land.

Capacity for it exists: partition `ctessum` is 5 nodes × 40 cores × 193 GB with
a 3-day limit (200 cores, 965 GB). Cross-node cost is negligible — a shard
exchanges ~304 kB per call each way at 2x2.5, ~7 MB/s per shard.

## 7. What is NOT worth doing

* **The transport VJP's 12x-over-primal codegen cost** (root-caused 2026-09-05,
  see `[[transport-vjp-22x-root-cause]]` and §6 lever 2). Real, but tiling
  divides the same 12x by 20; chasing a face-major custom reverse rule *before*
  tiling buys 249 h → ~90 h, where tiling alone buys 249 h → 12 h. Revisit after.
* **More XLA:CPU flags.** Exhausted and documented; every candidate measured
  ≤ 7% or negative.
* **Any epoch/tape cache.** Struck by standing rule (`[[no-scale-hostile-caches]]`);
  and note that levers 1–5 above are all O(1) or O(tile) in memory.
* **Loosening `rtol`** (1e-4 → 1e-3 would cut chemistry steps ~2.15x) — a real
  knob, but it changes the answer rather than the cost of computing it. Price it
  against bucketing, which buys 3.41x at fixed accuracy.
* **GPUs** — ruled out (`[[cpu-only-30min-target]]`).

## 8. Honest position on the 30-minute target

The standing target is a five-day CONUS gradient in 30 min of loop on CPU, and
that target was set at 4x5, where it stands at ~47 min for 48 h / ~2 h for five
days. At 0.25° CONUS the same window is 206x the columns; 23 h on 5 nodes is
~46x off the 30-minute figure. Reaching 30 min at 0.25° needs ~2,000 more cores
than the owned partition has, i.e. a different machine, not a different code
path. The tractable statement is: **five-day 0.25° CONUS in about one day of
loop on the 5-node partition, with compile and memory made grid-independent.**
