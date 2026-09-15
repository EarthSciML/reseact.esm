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
lane's own module, and the reverse-mode compile is still the wall.** That is the
honest state: `gather` made `ssp_step` comparable with the traced lane (207 s
against 170 s) and it did NOT make `ssp_vjp` comparable. 58,620 slices is 66% of
what remains, the reverse of a slice is a pad-and-add into a zero buffer, and
the traced lane's equivalent mass is 51,781 `broadcast_in_dim`, whose reverse is
a reduce. Getting further needs FEWER RUNS, not a cheaper form for the runs —
i.e. a materialization layout in which a stencil neighbour read is one
contiguous span — which is a question about `oop_merge.jl`'s block layout, not
about `_de_emit_runs`. It is the next thing to attack and it is not attacked
here.

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
