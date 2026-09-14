# Can the direct StableHLO emitter replace ReSEACT's traced `:oop` right-hand side?

Measurement, 2026-09-14, branch `direct-rhs`. Probe: `tools/diag/direct_rhs_census.jl`.
CONUS job file: `tools/diag/direct_rhs_census_conus.sbatch` (written, NOT submitted —
see "CONUS" below).

Nothing in the model, the emitter or the runners was changed to make anything
pass. One temporary instrumentation patch was applied to the EarthSciAST
worktree so the emitter would log and skip a refusal instead of stopping at the
first one; it was reverted and that worktree is clean.

## The answer

**No, not yet — and today there is no Reactant version on which the two lanes
can even be compared.** Three independent things are in the way.

1. **The direct emitter refuses both halves of the operator split.** Three
   distinct constructs are responsible, all of them things ReSEACT genuinely
   computes. None is an artefact of how the probe drives the emitter, and none
   is a per-cell fallback kernel.
2. **The lane it would replace does not compile on Reactant 0.2.285**, the
   version the emitter was built against — in two different ways depending on
   `ESS_OOP_SSA`. ReSEACT pins 0.2.280.
3. **The third program the brief asks about — EarthSciASTDiff's analytic band
   model (`jac = :sym`, `jacE.fJ!`) — cannot be built against this EarthSciAST
   branch at all.** It comes back EMPTY, for a reason that has nothing to do
   with the emitter and that also silently affects the production adjoint
   driver.

## Setup

Julia 1.12.6. Grid 6x6x8 (the drivers' demonstration preset), GEOS-FP 4x5 row:
**3,744 states, 162 runtime parameters, 15 discrete forcing providers**; the
transport and chemistry halves each bind 16 live forcing buffers, the band model
15. Base point: the harnesses' jittered state (`RESEACT_ADJ_UJITTER=1e-1`, seed
31337), with `Transport3D.m` seeded from the real GEOS-FP surface pressure —
never the default initial condition, which sits on the PPM limiters' switching
surfaces.

Both lanes are compiled with the SAME options, the adjoint driver's production
set: `CompileOptions(sync = true, xla_debug_options = (xla_cpu_prefer_vector_width = 128),
excluded_passes = ["dynamic_update_to_concat", "sub_const_prop"])`.

## 1. The gap list

Both halves refuse, and they refuse identically. With the census patch in place
the walk runs to the end and records every refusal; the granularity is the
emitter's own top-level unit (one observed fill, one fill kernel, one CSE
prelude entry, one state equation, one access kernel, one prefix scan), and the
construct named is the first offender inside that unit.

Measured at 6x6x8 on Reactant 0.2.285, `ESS_OOP_SSA=0`:

| units | refusal kind | class |
| ---: | --- | --- |
| 15 | the closed function `datetime.month` (a registry typed scalar core) | closed function |
| 9 | the operator `log10` | operator ladder |
| 1 | the closed function `datetime.hour` (a registry typed scalar core) | closed function |
| 1 | a read-before-write | cascade of the census guard |
| **26** | **total, per half — the same four kinds and the same counts in transport and in chemistry** | |

**Zero per-cell fallback access kernels.** Every access kernel in both halves
vectorized, so the structured-tier work the ruling calls for has nothing to
extend here. The instrumentation to report one is in place if a fallback ever
appears: the census patch makes the refusal carry
`[decline=<reason>]` straight off EarthSciAST's own classifier
(`_oop_decline_reason`, `src/tree_walk/oop.jl:2109` — `reduce_noncontig`,
`state_indirect`, `state_indirect_col`, `const_edge`, `const_cell_outs`,
`cached`), plus the cell-set shape, the descriptor kinds in the kernel's table
and the sub-kernel count.

### Gap A — the calendar family (`datetime.hour`, `datetime.month`, `datetime.is_leap_year`)

**16 of the 26 units in each half.** The emitter's own message:

> `E_DIRECT_EMIT_UNSUPPORTED: direct StableHLO emission cannot lower the closed
> function `datetime.hour` (a registry typed scalar core) (from the observed fill
> kernel writing flat slot 52165 … flat slot 52170 (6 lanes) (materialization
> level 4)): its body is an opaque Julia callee — the calendar/datetime family
> decomposes dates on the host — and a StableHLO program has nowhere to put one.`

Triggering constructs, recovered by walking the flattened system directly
(the emitter's IR carries no names for observed slots, so it can only say
"flat slot N"):

| model | variable | call |
| --- | --- | --- |
| `NEIRegrid` | `month_utc` | `datetime.month(t_utc)` |
| `NEIRegrid` | `is_leap_year_utc` | `datetime.is_leap_year(t_utc)` |
| `NEIRegrid` | `local_hour[gi]` | `datetime.hour(t_local_utc[gi]) + 1`, one lane per `lon` cell |
| `NEI2016Emis` | `month_utc`, `is_leap_year_utc`, `local_hour` | the same three |

All six appear in BOTH split halves — observed definitions are duplicated across
split parts — which is why the two halves have identical gap profiles. The single
`datetime.hour` unit is the `local_hour` fill kernel (6 lanes at NLON=6); the 15
`datetime.month` units are the observed fills and kernels that read `month_utc`
(with `is_leap_year_utc` inside the same units, masked by the first-offender rule).

**What the emitter would need.** `ext/reactant_direct/interp.jl` refuses every
`_FnTypedCoreSpec` by design: the body is a Julia callee (`Dates.unix2datetime`)
and StableHLO has nowhere to put one. Three ways out, and only two of them are
sound here:

* *Emitter-side (the general fix).* Lower the calendar family to StableHLO
  integer arithmetic. It is a small closed family, the arithmetic is exact in
  Float64 over the epoch range in use, and it is the only option that leaves
  every existing document working unchanged.
* *Model-side.* Replace the calendar decomposition with arithmetic on the
  elapsed time, exactly as `Transport3D`'s solar chain already does — the model's
  own notes record that `fastjx.esm` was passed over for
  `fastjx_interp_troposphere.esm` for precisely this reason, and that
  `days_in_month` is a table lookup "because EarthSciAST has no
  `datetime.daysinmonth`". `datetime.hour(t)` is `floor(mod(t/3600 + hour0, 24))`
  given the epoch; month and leap-year need a 12-entry table the model already
  knows how to write.
* *Forcing-buffer-side (what the refusal message suggests) — sound for two of
  the three, WRONG for the third.* `month_utc` and `is_leap_year_utc` change
  monthly and could be host-refreshed buffers. `local_hour` varies continuously
  with the solver time `t`; freezing it into a buffer would be the
  silent-staleness configuration unless the refresh cadence were at most an hour,
  which no runner has.

### Gap B — the operator `log10`

**9 of the 26 units in each half.** The emitter's message:

> `direct StableHLO emission cannot lower the operator `log10`: it is not in the
> direct-emission op ladder. Add it to `_de_op`
> (ext/reactant_direct/ops.jl) with the StableHLO op that matches the
> interpreter's `_oop_op` arm, or lower it away before the backend.`

Triggering construct: the **Troe falloff broadening factor of SuperFast reaction
R8**, three `log10` nodes in
`../EarthSciModels/components/gaschem/superfast.esm` of the shape
`0.6 ^ (1 / (1 + log10(k0[M]/k_inf)^2))`. They reach the rate of the
`SuperFast.NO2` equation and everything that shares it. `log10` appears nowhere
in `reseact.esm` itself — it arrives with the `reaction_systems` reference to
superfast.esm.

**What the emitter would need.** One arm in `ops.jl`. StableHLO has no `log10`
op, so it is `divide(stablehlo.log(x), const(log(10)))`; `log2` is the same shape
and is missing for the same reason. `_DE_UNARY` currently carries `log` but not
`log10`/`log2`. Note the tolerance consequence: the interpreter's `_oop_op` uses
Julia's `log10`, which is not bit-identical to `log(x)/log(10)`, so this pair
lands in the **transcendental** class (1e-12 relative), alongside `exp` and
`power`, rather than the algebraic class.

### Gap C — `a read-before-write` (1 unit per half)

> `direct StableHLO emission cannot lower a read-before-write: <slot> is read
> before anything in this emission wrote it. On host the flat buffer would supply
> a zero here; the emitter has no such buffer, so the ordering must come from the
> fill levels.`

This is a **cascade of the census guard, not an independent gap**: when the guard
skips a unit it fills that unit's own output slots with zeros, but a unit whose
producer was skipped can still reach a slot the skipped unit never listed. It is
counted separately so the 26 adds up, and it should disappear once gaps A and B
are closed. It is the one entry in the table that a gap-closing pass should
re-measure rather than plan for.

## 2. Agreement

**Not obtainable, in either direction, on any version tried.** Agreement is
defined as direct against traced at the same point; the direct lane has no
program for either half (it refuses), and on Reactant 0.2.285 the traced lane has
no program either (section 4). The probe therefore reports
`-- direct refused, nothing to compare` for both halves and writes no agreement
row, rather than comparing something against itself.

For the record, the tolerance classes this comparison would have been judged
against, had it run: 1e-13 relative for algebraic-only expressions, 1e-12 for
transcendental, 1e-11 for reductions, with an absolute floor of 1e-14. Both
halves would have landed in the **transcendental** class — the chemistry half
because the mechanism is `exp`/`power` throughout and (once gap B is closed)
`log10` lowers as `log(x)/log(10)` rather than as Julia's `log10`; the transport
half because the PPM stack's monotonicity guards go through `power` and the
solar chain through `sin`/`cos`. Neither is an algebraic-only program, so 1e-13
was never the right bar for either.

## 3. Cost

Two things can be measured despite the refusals, and one cannot.

**What can.** With the census guard substituting a zero constant for each refused
unit, the walk runs to completion and the rest of the model DOES emit and
compile. That program's values are wrong by construction and it is never
compared against anything — but its size and its compile and call costs are a
**lower bound** on what a gap-free direct lane would cost, missing the
arithmetic of the 26 skipped units and everything the zeros then const-folded
away.

Measured at 6x6x8, Reactant 0.2.285, `ESS_OOP_SSA=0`, 20 timed calls after
warm-up, optimized module (`optimize` on through the shared `CompileOptions`):

| program | `@compile` wall | per-call median | optimized module |
| --- | ---: | ---: | ---: |
| transport, direct, 26 units zeroed | 305.5 s | 1.5 ms | 14,832 lines / 14,828 ops |
| chemistry, direct, 26 units zeroed | 266.9 s | 1.5 ms | 14,689 lines / 14,685 ops |

The two stubbed programs coming out nearly the same size is itself a caution
about reading too much into them: what survives the zeroing is dominated by the
shared observed-fill and CSE prelude structure the two halves have in common,
not by each half's own arithmetic. Treat these as "the direct lane's fixed cost
on this model", not as a projection of the real programs.

**Build cost, for scale.** Building the two `:oop` halves took 244-395 s across
runs at this grid (same configuration; the spread is machine load). `ESS_OOP_SSA`
made no systematic difference to build time here.

**What cannot.** The traced rows of the cost table — the whole point of the
comparison — need a traced program, and section 4 is why there is none on
0.2.285.

## 4. The traced lane does not compile on Reactant 0.2.285

This is a finding in its own right, and it is why the comparison the brief asks
for has no common ground today. ReSEACT's default environment pins Reactant
**0.2.280**; the direct emitter was built against **0.2.285**, and the census
environment pins that. On 0.2.285 the traced `:oop` lane
(`EA.rhs_with_buffers(f)` under `Reactant.@compile`) fails on the transport half
in two different ways depending on the build-time `ESS_OOP_SSA` flag.

### 4a. `ESS_OOP_SSA=1` (ReSEACT's production default) — scalar indexing

```
ERROR: LoadError: Scalar indexing is disallowed.
Invocation of getindex(::TracedRArray, ::Vararg{Int, N}) resulted in scalar
indexing of a GPU array.
```

Call site, innermost ReSEACT/EarthSciAST frame outward:

```
[20] _oop_ssa_resolve   EarthSciAST/src/tree_walk/oop.jl:2540   (a `reduce` over a Vector{Any})
[23] _oop_ssa_ref       EarthSciAST/src/tree_walk/oop.jl:2522
[24] _oop_eval_acck     EarthSciAST/src/tree_walk/oop.jl:2577
```

and the scalar index itself is Reactant's own:
`overloaded_mapreduce` -> `unwrapped_broadcast` -> `unrolled_map`, which does
`push!(::Vector{Vector{Float64}}, ::TracedRArray)` and therefore
`Array(::TracedRArray)` (`Reactant/src/TracedRArray.jl:1350`, `:1332`, `:161`).

**Reading: a 0.2.280 -> 0.2.285 change in Reactant's `mapreduce` overlay, not a
ReSEACT-side assumption.** ReSEACT runs this exact code path in production on
0.2.280. What EarthSciAST does — `reduce` over a heterogeneous `Vector{Any}` of
traced values in the SSA class-to-class resolver — is unchanged; what changed is
that Reactant's overlay now tries to unwrap the elements into a concrete
`Vector{Vector{Float64}}` before reducing.

### 4b. `ESS_OOP_SSA=0` (the stock emitter) — the MLIR symbol uniquifier gives up

```
Could not find unique name for *_broadcast_scalar
```

raised by `Reactant.TracedUtils.__lookup_unique_name_in_module`
(`TracedUtils.jl:1055`), which tries at most 10,000 numeric suffixes before
erroring. It is reached from `finalize_mlir_fn` -> `make_mlir_fn(*)` ->
`elem_apply` -> the broadcast `_copyto!`, i.e. from the traced emitter's own
`EarthSciAST._oop_op` (`src/tree_walk/oop.jl:485`, via
`ext/EarthSciASTReactantExt.jl:654` and `:677`) on the `*` arm. The error prints
the entire module first, which is how a 50 MB log happens.

**Reading: NOT a version regression.** `__lookup_unique_name_in_module` is
byte-identical in 0.2.280 and 0.2.285, cap and all. This is the stock (non-SSA)
traced emitter minting more than 10,000 distinct private `*_broadcast_scalar`
helper functions for ReSEACT's transport half — a ReSEACT-scale property of the
model, not of the Reactant version. ReSEACT's production default
`ESS_OOP_SSA=1` is what keeps it under the cap, which makes the 4a failure the
one that actually blocks the port.

### The version matrix

| lane | Reactant 0.2.280 (ReSEACT's pin) | Reactant 0.2.285 (the emitter's) |
| --- | --- | --- |
| direct, `ESS_OOP_SSA=0` | not measured (see below) | **refuses**, 26 units per half |
| direct, `ESS_OOP_SSA=1` | not measured | **refuses** — measured to the first refusal only (`datetime.hour`, the same `local_hour` fill kernel); the full 26-unit census was run at `ESS_OOP_SSA=0` |
| traced, `ESS_OOP_SSA=0` | not measured; 4b is version-independent, so it should fail there too | **fails**, 4b — measured on the transport half; the chemistry half's traced compile had not finished when the process was stopped at 32 GB resident (the job's cap is 40 GB) |
| traced, `ESS_OOP_SSA=1` | not measured here; it is the configuration `tools/adjoint_gradient.jl` runs in production, and its recorded CONUS results are evidence that it works | **fails**, 4a |

**On the evidence in hand there is no version on which both lanes run**, and
no agreement number or cost ratio can be produced until 4a is resolved. The
0.2.280 column is the one that could still change that, and it is exactly the
column this session could not measure.

**Why the 0.2.280 cells are blank, and what it would take to fill them.** The
environment for it exists and is correct: a copy of `run-model-jl`'s manifest
with the EarthSciAST dev path rewritten to the oop-retire worktree (and its
version field to 0.2.0) and EarthSciASTDiff to the faq-patched scratch copy, at
`/scratch/$USER/oopretire-env-280`. What defeated four attempts was
**precompilation of Reactant 0.2.280 for that environment**: none of the shared
depot's eleven cached Reactant builds matched it, so Julia started a fresh one
every time, and every one of those workers sat in `cl_sync_io_wait` having
consumed about one second of CPU over ten or more minutes. A bare
`julia --project=run-model-jl -e 'using Reactant'` against the shared depot —
no rebuild needed, just loading — also timed out after 110 s, inside
`ijl_load_dynamic_library`, i.e. `dlopen` of Reactant's several-hundred-megabyte
native library. This is the shared filesystem, not the probe or the environment:
the 0.2.285 runs only got going because that version's cache was already
resident in the private depot from earlier in the session. The probe needs no
changes to produce these rows; it needs the Reactant 0.2.280 cache warm in a
private depot (build it once, alone, when the machine is quiet) and then one run
per `ESS_OOP_SSA` setting.

## 5. The `jac = :sym` band model cannot be built against this EarthSciAST

`EarthSciASTDiff.prepare_jacobian(splitparts[2]; wrt = :states, ...)` — the
program `tools/adjoint_gradient.jl` compiles as `jacE.fJ!` — returns

```
structure=empty  band states=3744  entries=0  scatter=0
```

i.e. **a band model with no Jacobian entries at all.** Nothing about the emitter
is involved; the third program of this census simply does not exist to be
measured.

**Root cause.** esm 1.1.0 renamed the `aggregate` node to `faq`, and EarthSciAST
normalizes the old spelling on load — every run of this probe logs
`[E_DEPRECATED_OP_ALIAS] "op": "aggregate" is the pre-1.1.0 spelling of "op":
"faq"; 130 nodes were normalized on load` for `reseact.esm`. But
`EarthSciASTDiff/pkg/EarthSciASTDiff.jl/src/system.jl:96` still reads

```julia
if l.op == "aggregate" && l.expr_body isa OpExpr && l.expr_body.op == "D"
```

so after normalization `lhs_state` recognizes no array state equation as a
differential equation, and `jacobian_bands` returns an empty `entries` vector.
Confirmed directly: `jacobian_bands(...; wrt = :states)` returns **0 entries on
split part 1, 0 on split part 2, and 0 on the whole flattened system**. Thirteen
sites in EarthSciASTDiff's source spell `"aggregate"`.

**Consequence beyond this census, worth its own look.** ReSEACT's production
`tools/adjoint_gradient.jl` with `RESEACT_ADJ_JAC=sym` (the default) builds this
same empty band model against a faq-normalizing EarthSciAST. Its guard —
`validate_plan`, "the gather plan does not reproduce the host Jacobian" —
compares over the entries that exist, so with zero entries it passes vacuously
and the Rosenbrock23 step proceeds with an all-zero block Jacobian.

**What was tried, and why the program still is not measured.** With those 13
sites renamed `"aggregate"` -> `"faq"` in a scratch copy of EarthSciASTDiff (the
main checkout untouched; the census environment's manifest repointed at the
copy), `prepare_jacobian` does real work — but at 6x6x8 it had consumed
**26 minutes of CPU and ~16 GB resident** without finishing, and was stopped to
protect the 40 GB job. The probe grew `RESEACT_CENSUS_JAC=0` for exactly this
reason, and the numbers in this report are with the band model skipped.

## 6. Differentiation

**Direct: no gradient to take.** Reverse mode differentiates a program, and the
direct lane has none for either half — the primal refuses. The probe says so
rather than differentiating the guard-stubbed program, which would produce a
confident-looking gradient of a model with 26 pieces replaced by zeros.

**Traced: not reachable on 0.2.285 either**, for the same reason as the primal
(section 4). The probe declines to attempt it when the traced primal did not
compile, again rather than reporting a gradient of nothing.

The calling forms the probe uses are the ones that work today in
`tools/adjoint_gradient.jl` and `tools/reactant_handoff/rx_traced_integrator.jl`,
with the payload an explicit argument rather than a closure — a closure over `p`
makes `dJ/dp` silently vanish instead of failing:

```julia
_wobj(f, u, p, t, b, w) = sum(w .* f(u, p, t, b))
grad_u(f, u, p, t, b, w) = EZ.gradient(EZ.Reverse, _wobj, EZ.Const(f), u,
                                        EZ.Const(p), EZ.Const(t), EZ.Const(b), EZ.Const(w))
grad_p(f, u, p, t, b, w) = EZ.gradient(EZ.Reverse, _wobj, EZ.Const(f), EZ.Const(u),
                                        p, EZ.Const(t), EZ.Const(b), EZ.Const(w))
```

with `w` the surface-ozone weight vector the adjoint driver scores
(`SuperFast.O3:surf`), and both compiled under the same `CompileOptions`. Forward
mode is deliberately absent: `Enzyme.jacobian(Forward, ...)` is on record as
failing under Reactant here, and checkpointing stays off.

What this means for the port: **the differentiability question is not yet
answerable.** It becomes answerable the moment gaps A and B are closed, and the
probe will answer it with no changes — `RESEACT_CENSUS_STAGES=emit,grad`.

## 7. CONUS

**Not submitted.** The brief's precondition was "only if the two RHS halves emit
at 6x6x8". Neither does — both refuse on the same three constructs — so a CONUS
job would have measured the same refusals at 13x7x72 for the price of a ~12
minute build and a ~10 minute compile per lane. The job file
`tools/diag/direct_rhs_census_conus.sbatch` is committed anyway: it is the
reproducer to run the moment gaps A and B are closed, it runs both `ESS_OOP_SSA`
settings in one allocation, and it needs no editing. No Slurm job ids were
created by this work.

## 8. How to reproduce

All of it runs from this worktree. The environments are scratch copies, so
nothing in any checkout is modified.

```bash
# The environment: ReSEACT's deps, EarthSciAST dev-linked to the oop-retire
# worktree, Reactant pinned to the emitter's 0.2.285.
ENV=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact-directrhs-env

# Always run Julia with a PRIVATE depot first: the shared depot's precompile
# lock will otherwise hang Pkg, and two of these runs started at once will
# block each other in `using Reactant`. Run them ONE AT A TIME.
export JULIA_DEPOT_PATH=/scratch/$USER/oopretire-depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia

# The census. Redirect to a file; never pipe a live run through tail/grep.
RESEACT_RXENV=$ENV ESS_OOP_SSA=0 RESEACT_CENSUS_JAC=0 \
RESEACT_CENSUS_JSON=/scratch/$USER/census-285-ssa0.json \
  timeout -k 15 14400 julia --project=$ENV --heap-size-hint=18G \
  tools/diag/direct_rhs_census.jl > /scratch/$USER/census.log 2>&1

# CONUS, once the gaps are closed:
sbatch tools/diag/direct_rhs_census_conus.sbatch
```

Knobs: `RESEACT_NLON/NLAT/NLEV` (default 6/6/8), `RESEACT_CENSUS_STAGES`
(`emit,agree,cost,grad`), `RESEACT_CENSUS_NCALL` (20), `RESEACT_CENSUS_EXCL`
(the excluded StableHLO passes, or `none`), `RESEACT_CENSUS_JAC` (1/0),
`RESEACT_ADJ_UJITTER` (1e-1), `ESS_OOP_SSA` (build-time, so one process per
setting).

**Enumerating every refusal needs a temporary patch to EarthSciAST.** Without it
the emitter stops at the first one, which is by design. The patch wraps each
top-level unit of `_de_emit!` in a guard that logs the `DirectEmitError`, fills
that unit's slots with zeros and carries on, and adds the classifier's decline
reason to the per-cell-fallback refusal. It is saved at
`/scratch/$USER/oopretire-logs/census-patch.diff` and applied and reverted with

```bash
cd /projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciAST-oopretire
git apply       /scratch/$USER/oopretire-logs/census-patch.diff   # instrument
git apply -R    /scratch/$USER/oopretire-logs/census-patch.diff   # revert
git diff --stat                                                   # must be empty
```

The probe detects the patch (`isdefined(ext, :_DE_CENSUS_LOG)`) and does nothing
extra without it, so the committed script runs against a clean worktree.

## 9. What the port needs, in order

1. **`log10` (and `log2`) in the direct op ladder.** One arm in
   `ext/reactant_direct/ops.jl`; `divide(log(x), const(log(10)))`. Cheapest fix
   on the list, removes 9 of 26 refusal units per half, and moves the model into
   the transcendental tolerance class where it already belonged.
2. **The calendar family.** Either lower `datetime.hour` / `datetime.month` /
   `datetime.is_leap_year` to StableHLO arithmetic in
   `ext/reactant_direct/interp.jl`, or rewrite `NEIRegrid`'s and
   `NEI2016Emis`'s six observeds the way `Transport3D`'s solar chain was already
   rewritten. Removes the other 16 units. Note that the emitter's suggested
   workaround — a forcing buffer — is only sound for `month_utc` and
   `is_leap_year_utc`, never for `local_hour`.
3. **Re-measure the read-before-write.** It is a guard cascade, and it should
   vanish with 1 and 2; if it does not, it is a real ordering gap in the fill
   levels.
4. **A Reactant version both lanes run on.** Until 4a is fixed upstream or
   worked around in `_oop_ssa_resolve`, there is no version on which the direct
   lane (needs 0.2.285) and ReSEACT's traced lane (runs on 0.2.280) can be
   compared in one process, so no agreement number and no cost ratio can be
   produced at all.
5. **`EarthSciASTDiff`'s `aggregate` -> `faq` rename** (13 sites), which is a
   prerequisite for the `jac = :sym` program existing — and, independently, for
   the production adjoint not integrating with an all-zero block Jacobian.
6. **Only then** is the CONUS run worth its allocation.

## 10. Where the runs are

Logs and machine-readable records from this session (outside the repository,
because `/tmp` and `/` are RAM-backed here):

| file | what |
| --- | --- |
| `/scratch/$USER/oopretire-logs/runA2-285-ssa0.log` | the 0.2.285 `ESS_OOP_SSA=0` census: both halves' gap lists and stubbed-direct costs |
| `/scratch/$USER/oopretire-logs/run1-emit-ssa1.log` | the 0.2.285 `ESS_OOP_SSA=1` run: first refusal, and traced failure 4a |
| `/scratch/$USER/oopretire-logs/run2-all-ssa0.log` | traced failure 4b with the full module dump and stack |
| `/scratch/$USER/oopretire-logs/census-patch.diff` | the temporary emitter instrumentation, applied and reverted |

No `RESEACT_CENSUS_JSON` file was produced: the probe writes it last, and every
run was stopped before that point — the 0.2.285 ones deliberately, once the
traced lane had failed and the memory was climbing past 32 GB, and the 0.2.280
ones because they never got past loading Reactant. Every number in this report
is from the run logs above.

No Slurm jobs were submitted, so there are no job ids to report.
