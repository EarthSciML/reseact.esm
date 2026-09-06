# Index-range reads for the GEOS-FP forcing — plan

**Status:** plan, not implementation. Written 2026-09-05, after resolution
became a first-class argument (`native_slice(res = ...)`, `reseact_forcing(;
res = ...)`) and the mirror survey showed that finer CONUS meteorology means
slicing the **global** GEOS-FP files — there is no nested North-America met
product on either mirror (only `soil`).

The short version: **the pushdown machinery already exists, end to end, and is
already exercised** — by the zarr/ISRM path. What is missing is one reader
capability (the netCDF reader is whole-file and takes no decode options at all
in the Julia track) and the wiring that turns this model's slice into a
selection. Steps 1 and 2 below are small, are useful at 4x5 today, and are what
makes 0.25x0.3125 arithmetically possible; step 3 is the one that stops
downloading the globe, and it is already an upstream charter item.

---

## 1. Why this is needed

The model reads the **whole native array** for every forcing variable and
indexes it with `LON0+gi` / `LAT0+gj`. That was free at 4x5 and is not free
above it. Measured on the cached 2016-01-06 4x5 files (raw netCDF sizes; the
decode is Float64, so double the on-disk float32):

| collection | data vars | decoded whole, Float64 | providers this model puts on it |
|---|---|---|---|
| A1     | 47 | 29.9 MB | 6 (PBLH, TS, Z0M, USTAR, SWGDN, HFLUX) |
| A3dyn  |  5 | 76.3 MB | 4 (U, V, OMEGA, RH) |
| A3cld  |  7 | 106.8 MB | 1 (CLOUD) |
| A3mstE |  5 | 77.4 MB | 2 (PFLCU, PFLLSAN) |
| I3     |  4 | 46.0 MB | 2 (PS, T) |

Each provider decodes its file **whole** — every variable, all 8 records — and
keeps two records of one variable. That is ~840 MB of decode per cadence tick at
4x5 to retain a few MB, and it scales with the native grid: **×4.0 at 2x2.5**
(~3.3 GB/tick) and **×251 at 0.25x0.3125** (~211 GB/tick). The forcing refresh
is already 0.87 s of the 4.86 s 48-h CONUS window budget (18%) at 4x5, where the
files are small.

What the model actually needs is the **halo-inclusive window**: lon
`LON0 .. LON0+NLON+1`, lat `LAT0+1 .. LAT0+NLAT`, all levels, the two bracketing
records. As a fraction of the native grid that is 120/3,312 = **3.6% at 4x5** and
19,110/830,592 = **2.3% at 0.25x0.3125**.

That it needs *nothing else* is measured, not assumed:
`tools/diag/native_window_probe.jl` writes NaN into every native cell outside
that window and the transport RHS comes back **bit-identical** — at 4x5 (3.8 M
cells masked) and at 2x2.5 (15.2 M masked). The same probe answers the question
that has to be settled before any windowed read: the emitted indexing is driven
by the runtime array shape, not by the declared `gf_lon`/`gf_lat` index-set
sizes, so a pre-sliced array is a rebasing problem and nothing more.

Two independent costs follow, and they need different fixes:

* **decode + resident memory** — fixed by a selection the *reader* honours
  (steps 1–2). This is what makes the 0.25° arithmetic land: ~211 GB/tick → under
  1 GB/tick, and ~4x better again if the two-record bracket is also pushed down.
* **download** — the 0.25x0.3125 A3dyn file is **3.8 GB per day** (vs 28 MB at
  4x5), so a five-day run pulls ~100 GB of which it reads ~2%. Only step 3 fixes
  that.

---

## 2. What already exists (the user's premise, verified)

**EarthSciAST** — the model-side seam is complete and provider-neutral:

* `provider_sample(provider, t; selection = nothing)` — one entry per NATIVE
  axis: `Colon()`, an `Integer` (1-based), or an ordered
  `AbstractVector{<:Integer}`. The returned gated axis is ordered exactly as the
  index vector. `src/data_refresh.jl:225`.
* `provider_supports_selection(provider)` — the gate; handing a `selection` to a
  provider that answers `false` is a clear error, and the fall-back-to-full-read
  belongs to the build side. `src/data_refresh.jl:89`.
* **Gated deferral** with a written axis vocabulary — `"all"`,
  `{"fixed": [i]}`, `{"range": {"start", "stop", "step"}}`,
  `{"gated_by": "<set>"}` — declared as `data_sources.<n>.metadata.x_esd.gated_select`,
  and explicitly "ONE vocabulary with the loader `select` of esm-spec §8.9.2".
  `src/data_refresh.jl:114`. **`range` is exactly a lon/lat window.**
* `src/pushdown_rewrite.jl` generates that record from ordinary model math for
  the emissions case, so the model does not have to spell it by hand.

**EarthSciIO** — the reader-side seam is complete for store-backed readers:

* `Provider` takes `select` baked in `reader_kwargs` **and** as a per-call
  override; `materialize(p, t; select = ...)`. `julia/src/provider.jl:166`.
* `supports_selection(reader)` / `store_backed(reader)` — an additive,
  default-off capability pair. `julia/src/registries.jl:121-135`.
* `array_shape(p, var)` — the full native shape for an honour/refuse decision,
  read from metadata without fetching a chunk. `julia/src/provider.jl:230`.
* The **zarr** reader implements the real thing: `select = {axes: [...]}` with
  `"all"` / `{indices}` / `{slice: [start, stop, step]}`, mapping each requested
  index to its chunk and fetching only the intersecting chunk objects — "never
  the whole array (the ISRM linchpin)". `spec/conformance.md` §"Zarr decode notes".
* The cache key convention **already reserves byte ranges**: `#bytes=<a>-<b>` is
  appended to the URL before hashing, so a sub-slice is its own cache entry.
  `spec/cache-format.md` §1.
* `spec/cloud-future.md` §3 names **the NetCDF→Zarr conversion** as part of the
  `esio-cloud` epic — i.e. step 3 below is upstream's own plan, not a new idea.

**The gap** is one reader, and it is narrower than it looks:

* Julia `read_native(::NetCDFReader, path)` takes **no keyword arguments at
  all** — not even `variables`. `julia/src/readers.jl:122`.
* The Python track's `NetCDFReader.read_native` **does** take `variables` (and
  accepts `select` for interface parity while ignoring it).
  `earthsciio/readers.py:138`.
* Consequently `Provider._load` finds no `:variables` option on the Julia reader
  and takes the read-everything-then-`_select` path (`julia/src/provider.jl:190`),
  and a `select` handed to any whole-file reader is a hard error by design
  (`julia/src/provider.jl:186`; `spec/conformance.md` §3: "`select` never reaches
  a whole-file reader").

So: the vocabulary, the capability flags, the provider plumbing, the model-side
API and the cache-key convention are all in place. What is missing is a
**selection-capable whole-file reader** — plus this repo's own wiring.

---

## 3. The plan

Four steps. Each is independently useful and independently gate-able; only
step 4 depends on more than the step before it.

### Step 1 — `variables` decode option for the Julia netCDF reader

Give `read_native(::NetCDFReader, path; variables = nothing)` the projection the
Python track already has. `reader_option_keys` reads the method's own keyword
declaration, so declaring it is what makes `Provider._load` push `variables`
into the reader instead of decoding everything and calling `_select`.

* **Buys:** at 4x5, ~840 MB → ~150 MB of decode per tick (5.6x), most of it from
  A1 (6 providers × 47 variables) and A3dyn (4 providers × 5 variables). Pure
  win at every resolution, no new concepts, no spec change (`conformance.md` §3
  already describes exactly this arrangement: "A reader with no such option keeps
  the read-everything-then-`_select` path").
* **Cost:** ~20 lines and a test. Cross-track parity **improves**.
* **Gate:** the decoded `NativeDataset` for a requested subset must be
  field-for-field identical to today's decode-then-`_select`, including coords
  (which are always kept) and `attrs`.

### Step 2 — decode-time `select` for whole-file readers

Add a second capability tier so that a whole-file reader may honour an
orthogonal selection **at decode time**: it still fetches one blob, but it
materialises only the requested hyperslab. In NCDatasets this is
`v.var[i1:i2, j1:j2, :, :]`, which reads only the intersecting HDF5 chunks off
local disk.

Design points that need deciding **before** any code:

1. **Capability spelling.** `supports_selection` is currently documented as
   "store-backed reader that fetches only the selected chunks". Either widen its
   contract to "honours `select` without materialising the whole array" and let
   callers use `store_backed` to tell fetch-scoped from decode-scoped, or add a
   distinct `supports_selection_local`. The first is less API, the second cannot
   mislead a caller into thinking the download shrank. **Recommend the first plus
   an explicit note**, because `provider_supports_selection` on the AST side is
   already a single boolean and a second one would have to be threaded through.
2. **Axis order and base.** The Julia reader permutes NCDatasets' column-major
   arrays back to **file order**; a `select` must be in that same file order
   (`[time, lev, lat, lon]` for A3dyn), and the AST-side selection is 1-based
   while zarr's is 0-based. The existing zarr binding already does the 1→0
   translation; the netCDF one is 1-based throughout and must say so.
3. **Coordinates must be sliced with the data.** The zarr reader returns no
   coords, so this question has never come up. A netCDF hyperslab that returned
   full-length `lon`/`lat` coords next to a windowed variable would be a trap of
   exactly the kind this repo's degree-vs-index seam already documents.
4. **Time is the Provider's axis, not the reader's.** `records_per_sample = 2`
   means the Provider owns record selection (`conformance.md` §3: "Row selection
   is not a reader concern"). A `select` whose time axis is anything but `"all"`
   must be refused, and the two-record narrowing should be done by the Provider
   pushing a *time* selection of its own once it knows the bracket — that is the
   extra 4x noted above, and it is a separate, later bead.
5. **Cache key is untouched.** The same URL is fetched whole, so the blob and its
   `sha256(url)` key are unchanged; a windowed decode must never alter either.
   (This is the difference from step 3, which does change what is fetched, and
   which the `#bytes=` convention exists for.)
6. **Three tracks + a conformance case.** Python accepts `select` today and
   ignores it, so it currently satisfies the *interface* and violates the new
   *contract*: it must start honouring it, and Rust with it. This is the step
   that costs real upstream review, and it should be filed as one bead with the
   case (a small windowed read of an existing corpus netCDF, asserting three-way
   array equality against the full read sliced afterwards).

* **Buys:** ~27x at 4x5 (window is 3.6% of the native grid), ~43x at
  0.25x0.3125, on top of step 1 — the difference between 211 GB/tick and under
  1 GB/tick at 0.25°. Also a straight speed lever at 4x5, where refresh is 18%
  of the window budget.
* **Gate:** at 4x5, a windowed run must reproduce the whole-array run
  **bit-for-bit** — same gradient CSV, same accept/reject ladder. The arrays are
  indexed identically; anything else is an off-by-one.

### Step 3 — stop downloading the globe

Two routes, and they are not equivalent:

* **3a. NetCDF→Zarr transcode on ingest.** Fetch the daily file once, write a
  chunked zarr store into the cache (EarthSciIO already has zarr write), and
  point the provider at the store. Every subsequent read is the **landed**
  store-backed path with real chunk-scoped fetches, no new reader, no spec
  change, and `spec/cloud-future.md` §3 already names this conversion as part of
  `esio-cloud`. Price: one full download and one transcode per file (3.8 GB at
  0.25°), and cache growth.
* **3b. HTTP byte-range HDF5 reader.** A store-backed netCDF reader that parses
  the HDF5 chunk index and fetches only the intersecting byte ranges, each its
  own `#bytes=a-b` cache entry (the convention exists). Never downloads the
  globe at all. Price: a real HDF5 chunk-index implementation in three
  languages, and the transport must support ranged GETs (the `s3` charter lists
  byte-range requests as still-in-scope work).

**Recommend 3a first** — it reuses a landed, conformance-proven path and turns
"can we run at 0.25°?" into a question about disk rather than about a new
binary-format reader. 3b is the right long-run answer for a cloud-hosted run and
should stay on the charter, not on this critical path.

### Step 4 — wire the window in this repo

Only after step 2. The changes are small and all live in
`prototypes/reseact_3d_chem/`:

* `native_slice` grows a `window` field: the halo-inclusive native ranges
  `lon0 .. lon0+nlon+1` and `lat0+1 .. lat0+nlat`, derived from the same origin
  as everything else — a seventh consumer of it, and the note there already
  explains why nothing else may re-derive it.
* `reseact_forcing(...; window)` passes a per-axis selection to each provider
  (all-`Colon()` when `window === nothing`, so the default path is unchanged).
* **The metaparameters rebase**: with a pre-sliced array, `LON0` becomes 0 and
  `LAT0` becomes 0 — the `.esm` needs no edit at all, because `LON0+gi` is
  already the whole story. This is the one genuinely error-prone line, and it is
  exactly why the 4x5 bit-identity gate above is the acceptance test rather than
  a smoke check.
* Refuse a window that wraps the dateline (`lon0+nlon+1 > nlon_native`) rather
  than silently reading a truncated slab. The bounds check in `native_slice`
  already rejects it for the model's own reasons; the window path must not
  quietly relax it.

---

## 4. What each step is worth

Per cadence tick, decode + resident bytes (Float64), CONUS box:

| | 4x5 | 2x2.5 | 0.25x0.3125 |
|---|---|---|---|
| today (whole file, every var) | ~840 MB | ~3.3 GB | ~211 GB |
| + step 1 (`variables`) | ~150 MB | ~590 MB | ~37 GB |
| + step 2 (window) | ~5 MB | ~21 MB | ~870 MB |
| + time-axis pushdown (later bead) | ~1 MB | ~5 MB | ~220 MB |

Download per five-day run is unchanged by steps 1–2 (~170 MB at 4x5, ~2 GB at
2x2.5, ~100 GB at 0.25°) and is what step 3 addresses.

---

## 5. Traps worth writing down before starting

* **The halo is +1 on both lon sides and +1 on the north lat side.** The model's
  own bounds check is `lon0+nlon+1 <= nlon_native` and `lat0+nlat <= nlat_native`;
  a window that forgets the halo produces a model that runs and is wrong at the
  boundary only.
* **1-based vs 0-based, twice.** AST-side selections are 1-based, the zarr
  reader's `slice` is 0-based, and the `gated_select` `range` form is 0-based
  half-open. A netCDF `select` should follow the AST side (1-based, inclusive)
  and say so in its own docstring, because the reader is the layer where both
  conventions meet.
* **Coords, `attrs` and CF decode must survive the window.** Whatever the reader
  returns for a windowed read has to CF-decode identically to the same cells
  read whole — `scale_factor`/`add_offset` in float64, `_FillValue`→NaN, time
  raw.
* **Offline mode.** A windowed decode of a blob already in the cache must still
  work with `EARTHSCI_OFFLINE=1`; a *byte-ranged* fetch (3b) introduces new cache
  entries and therefore new offline misses, which is a further argument for 3a.
* **This model creates one provider per variable on the same file.** After step
  1 that is 4 decodes of one variable each rather than 4 decodes of five; a
  future consolidation (one provider per collection, fanned out to variables)
  would be a bigger win still, and belongs in this repo, not upstream.
