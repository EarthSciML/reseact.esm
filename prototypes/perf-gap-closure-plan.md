# Closing the performance gap to a hand-written kernel — plan

**Scope:** EarthSciAST tree-walk build + runtime, driven by the reseact_3d / reseact_3d_chem
prototypes. **Non-goals:** changing the .esm spec surface, weakening any bit-exactness
guarantee, or demoting the interpreter from reference semantics.

## Baseline and targets

| metric | today | hand-written | target |
|---|---|---|---|
| 72-level transport build (45,864 states) | 1548 s (after `08c4985b`) | ~ms (compile a loop nest) | **≤ 60 s cold, ≤ 5 s warm (cached)** |
| RHS wall time (4459 states, Stage C) | ~3.5 ms/call | ~0.1–0.3 ms | **≤ 2–3× hand-written** |
| 24 h gridded chem solve | intractable monolithic; split solver pending | minutes | **minutes** |

The two gaps have different physics:
- **Build (~1000× gap):** structural specialization over share-free inlined ASTs —
  cost ∝ rule instantiations × spine size × region classes, dominated by IdDict/String
  churn over spines with 10⁴–10⁵ nodes (PProf: ~82% in affine-stencil lowering).
  Root amplifier: no sharing — one species' RHS re-inlines to 57,688 `index` ops, and
  `_resolve_observed` (helpers.jl:151) copies the whole GEOS-FP forcing chain into every
  use site (measured: identical 9 instantiations, 277 s analytic vs 814 s GEOS-FP winds).
- **Runtime (~10–30× gap):** interpreted spine walk per cell per call (`_eval_node`,
  compile.jl:1220; `_run_acc_kernel!`, access_kernel.jl:643) — dispatch + pointer-chasing
  per node instead of a compiled loop nest; plus the stiff solver multiplying RHS calls
  through FD Jacobians.

## Invariants to preserve (operational definition of "the benefits")

1. **The .esm document stays the single source of truth** — declarative, rule-matched,
   bindings-portable. No spec/schema change required by any item below (everything is
   engine-internal; A1/A2 change in-memory representation only, serialization untouched).
2. **The interpreter remains the reference implementation.** Every accelerated tier ships
   with (a) a differential oracle against the interpreter and (b) an env-var kill switch —
   the pattern `ESS_STENCIL_DISABLE` / affine-vs-per-cell bit-exact oracle already
   establishes. Anything a tier cannot model falls back, never silently degrades.
3. **Bit-exactness is tiered explicitly.** The default path (interpreter, affine kernels,
   Julia codegen in B1) must be *bit-identical* — same op order, same associativity — so
   the CWC telescoping gates and conformance goldens keep passing to the bit. Backends
   that cannot promise that (XLA in B2) are opt-in and validated by tolerance oracle, never
   default.
4. **Live forcing semantics unchanged:** discrete-cadence buffers refreshed in place by
   callback remain visible to every backend (this is the one known Reactant blocker; see B2).
5. **Conformance surface:** all existing gates stay green byte-for-byte — the 111 ESD AST
   goldens, `tests/conformance/function_tables/`, the stencil_affine differential oracle,
   the tree_walk/oop/vectorized/template/pde suites (2758 tests), and the reseact_3d
   CWC + probe3 gates.

---

## Phase 0 — Harness first (small, do before anything else)

Every later item claims a ratio; make the ratios measurable and the regressions loud.

1. **Canonical bench fixtures** checked into EarthSciAST `scripts/`: the controlled 27-box
   proxy (exists, used in `08c4985b`), 7×7×7 transport, 7×7×72 transport, one chem probe.
   Each records: build wall time, `_BENCH_BODY_VARIANTS` / `_BENCH_COMPILE_CALLS`, warm
   RHS time + allocs (Float64 and Dual), and peak RSS.
2. **A perf CI job** (or at minimum a `make bench` target run before merging perf-relevant
   PRs) that diffs those counters against a committed baseline JSON and fails on >10%
   regression.
3. **Wire `Profile.Allocs` + PProf snippets** from the treewalk-perf-plan reproducers into
   `scripts/` so attribution is one command, not an archaeology session.

Exit: one command prints the table above for HEAD.

---

## Workstream A — Build time (no semantics change, pure engine work)

### A1. Hash-cons the expression AST at load/expand time  ★ biggest single lever
Intern every `OpExpr` into a structural-hash table as it is parsed / template-expanded, so
the in-memory AST is a DAG: textually identical subtrees become the *same object*. The
entire build is already identity-memoized end to end (`_sub_preserving`,
`foreach_subexpr_once`, `_BuildMemo.resolve/compile`, `_build_acc_cse`, the ESS-0hh
lowering memo) — today those memos mostly miss because inlining manufactures fresh copies.
With interning they hit, and the 57,688-op re-inlining blowup collapses to its unique-node
count without any `let` node or spec change.

- **Trap to design around:** two sites key behavior on object identity where identity means
  *use site*, not *structure* — `_TemplateCtx.sites :: IdDict{OpExpr,OpExpr}` and
  `vkey = (objectid(root), bodykey)` (stencil.jl:311). After interning, two textually
  identical roots at different sites would collapse. Fix by keying those on an explicit
  site id (equation index + path) carried alongside the node, not on `objectid`. Audit
  every `IdDict{OpExpr,…}` for the same assumption before flipping interning on.
- Oracle: build with/without interning (env flag), compare compiled kernels and `du`
  bit-for-bit on all fixtures.

### A2. Stop inlining observeds — substitute a shared reference, not a copy
Replace the naive RHS copy in `_resolve_observed` with a single shared node that reads a
named slot (the ess-obs-slots machinery in build.jl already evaluates named observeds once
per call into `_CSECache`). The GEOS-FP forcing chain is then *referenced* from every
stencil lane instead of *copied* into it. Evidence-based expectation: ~3× on
forcing-heavy builds (814 s → ~280 s class), and it compounds with A1.

### A3. Memoize compiled rule bodies across instantiations
`_TemplateCtx` is per-equation, so the same PPM template with the same bindings compiles
fresh for `D(mq,t)`, `D(dev,t)`, and — in Stage C1 — for each of 12 species (39
instantiations of ~3 distinct rules). Hoist the variant cache to the build level, keyed by
(template identity, bindings hash, region-class branch key, operand *structural* hash).
Expected: ~2× on reseact_3d (12 → ~6 unique), ~10× on the per-species chem assembly
(39 → ~4), turning the "~1 h full assembly" estimate into minutes.

### A4. Finish the mechanical churn kill (continuation of `08c4985b`)
Remaining known offenders: `_cell_key` string interpolation + `var_map::Dict{String,Int}`
lookups on the per-cell fallback and box-corner verification paths (replace with linear
index arithmetic over the state slab — the layout is regular), `_branch_key!`'s IOBuffer
string keys (replace with an isbits hash tuple like `_desc_key` got), and any residual
`IdDict{_Node,…}` in lowering (dense postorder numbering, as `_build_acc_cse` now does).
Re-profile after A1–A3 first — the hot mix will have shifted.

### A5. Persistent on-disk build cache  ★ biggest iteration-loop win
The prototypes rebuild the *same* document dozens of times while the runner/solver evolves.
Key: content hash of (canonicalized flattened doc + bindings + const_array shapes/values
hash + engine version + relevant env flags). Value: serialized compiled artifacts
(`_Node` spines, `_AccKernel` descriptor tables, lane recipes, interp specs — all plain
data, no closures). On hit, `build_evaluator` deserializes and rebinds live buffers.
Invalidation is pure content-hashing; a stale-cache bug is unrepresentable if the hash
covers every input. Warm rebuild target: seconds. (This also makes A1–A4 less urgent for
daily work, but they still gate the *cold* build and CI.)

### A6. Parallelize the build
Equations and template variants are independent; the per-equation compile (`_stencilize` →
`_compile` → affine lowering) can fan out across Julia threads once shared memos (interning
table, variant cache) are made thread-safe or sharded. Do last — it multiplies whatever
efficiency the serial build has, including waste, so land A1–A4 first.

**Workstream A exit gate:** 72-level transport cold build ≤ 60 s, warm ≤ 5 s; all
conformance goldens byte-identical; bit-exact oracle vs pre-A1 kernels on all fixtures.

---

## Workstream B — Runtime (compiled tiers behind the existing fallback architecture)

### B1. Julia codegen for access kernels  ★ the default fast path, bit-exact
The compile-once tier already reduces the PPM stack to ~40 spine bodies per model. Emit
each body as specialized Julia source — a straight-line expression mirroring the
interpreter's *exact* op order and associativity (the affine tier proves this contract is
achievable: its arithmetic is "byte-for-byte the per-cell path's") — wrapped in a per-box
`@inbounds` loop nest with direct `u[oln + Δ]` indexing from the descriptor table, then
compile via RuntimeGeneratedFunctions.jl (new light dep). ~40 small functions ⇒ Julia
compile latency is seconds, once, at build.

- Eltype-generic emission so ForwardDiff Duals work; the split solver's FD Jacobian only
  needs Float64 anyway.
- Fallback contract identical to the affine tier: anything the emitter can't model
  (contractions with dynamic valence, etc.) runs the interpreter kernel for that equation.
- Oracle: per-kernel `du` bit-compare against the interpreter on every fixture + the
  stencil_affine differential suite; kill switch `ESS_CODEGEN_DISABLE=1`.
- Expected: RHS 3.5 ms → 0.1–0.3 ms class (10–30×); SIMD on the inner lon loop is free
  once the loop nest exists.

### B2. Finish the Reactant/XLA path as the opt-in large-grid/GPU tier
`form = :oop` + `EarthSciASTReactantExt` already compiles the RHS by tracing the walk once
(program size grid-independent). The single blocker is root-caused and even measured in
the ext's own docstring: live forcing buffers are closure captures, so XLA bakes them in as
trace-time constants (`@test_broken` in test/reactant_oop_test.jl). Fix: promote
`param_arrays` / DiscreteMaterializer buffers from captures to explicit traced-function
arguments (`ConcreteRArray` inputs; in-place `copyto!` between calls is already verified
visible to the compiled program). Then the discrete-cadence refresh model needs no
rethinking — only the binding moves.
- XLA may reassociate floats ⇒ this tier is **opt-in**, validated by tolerance oracle
  (and the CWC gate run in tolerance mode), never the default. It buys GPU + fusion for
  big grids where bit-exactness is a conscious trade.

### B3. Cadence-tier time-only observeds (Fix 2 of treewalk-perf-plan.md)
Partition observed slots by dependence: state-dependent vs t-only (forcing gathers,
`w_time` blends, photolysis chains). Memoize t-only slots on `t`, invalidated when the
integrator moves/rejects; an FD Jacobian fill (fixed `t`, perturbed `u`) then reuses one
forcing evaluation across all N+1 columns instead of recomputing it 16–46×. The const
tier + `_cse_const_stale` machinery is the template; this adds one more tier between
const and continuous (the cadence audit of 2026-07-14 already maps this terrain).

### B4. Cross-node CSE for repeated fn/interp evaluations
The FastJX case: ~100 `interp.linear` nodes re-evaluated every call, many over the same
input (e.g. one `cos_zenith` feeding 18 bands). Extend ess-obs-slots so structurally
identical fn subtrees (cheap to detect once A1's interning lands — they're the same
object) share one slot per call. Largely subsumed by B1 within a kernel, still wanted
across kernels/observeds.

**Workstream B exit gate:** Stage-C RHS within 3× of a hand-written reference kernel
(write one 50-line PPM-lon reference loop for the bench fixture — measure, don't assume);
bit-exact oracle green for B1; 24 h gridded chem solve completes in minutes with B1+B3+C1.

---

## Workstream C — Solver-side (mostly done; finish and default it)

1. **Land the split/IMEX results** (SPLIT_SOLVER.md "Results" is pending): run the 3600 s
   comparison, pick Option A vs B on evidence, and make the split driver the documented
   default for gridded chemistry. Advection stays explicit — no transport Jacobian at all.
2. **Analytic chemistry Jacobian per cell class** via the Catalyst ext (one 13×13 symbolic
   block, shared across all cells) to replace the 13-color FD fill — removes NS+1 chem RHS
   evals per Jacobian and composes with B3.

---

## Sequencing

```
Phase 0 ──► A1 ──► A2 ──► A3 ──► A4 ──► A6
              │                    
              ├────────► A5 (cache; independent after A1's canonical hashing exists)
              │
              └► B1 ──► B4        B3 (independent; anytime)   C1 (independent; now)
                   └──► B2 (opt-in tier; after the buffer-argument refactor)   C2 (after C1)
```

Rationale: A1 first because interning is the substrate that makes A3's structural keys and
B4's detection trivial, and it shrinks every spine that A4/B1 walk. A5 and B3 and C1 are
independent and can be parallel-tracked. B1 before B2: the bit-exact tier is the default
and de-risks the emission contract that B2's tolerance tier then relaxes.

## Expected compound outcome (order-of-magnitude, to be replaced by Phase-0 measurements)

| stage | build (72-lev cold) | RHS |
|---|---|---|
| today | 1548 s | 3.5 ms |
| + A1/A2 (sharing) | ~300–500 s | ~2 ms (smaller spines) |
| + A3/A4 | ~100–200 s | — |
| + B1 (codegen) | +~10 s Julia compile | ~0.1–0.3 ms |
| + A5 (warm) | **~5 s** | — |
| + B3 (+C) | — | Jacobian-heavy solves ÷5–15 |

## Risks and their controls

- **Interning silently merges nodes whose identity encoded a use site** → pre-audit of all
  `IdDict{OpExpr,…}` / `objectid` keys (A1 bullet); bit-exact oracle on every fixture.
- **RuntimeGeneratedFunctions world-age / precompile friction** → kernels are built at
  `build_evaluator` time inside the session (no precompile requirement); fallback to the
  interpreter is always live.
- **Cache poisoning (A5)** → content-hash covers doc + bindings + arrays + engine version +
  flags; add a paranoid mode that rebuilds and bit-compares on cache hit in CI.
- **XLA numeric drift (B2)** → opt-in tier only, tolerance oracle, CWC in tolerance mode;
  never used for conformance runs.
- **Perf work regressing semantics under deadline pressure** → Phase-0 gates are merged
  first and are non-negotiable: no perf PR lands with a red oracle or golden diff.
