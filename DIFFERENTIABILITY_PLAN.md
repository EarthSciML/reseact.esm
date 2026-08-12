# Differentiable simulation: measured capabilities and the plan

The point of the Reactant/XLA work is not speed — measured, it is *slower* than the
native runner on CPU for this model. The point is **∂(simulation result)/∂(parameter
vector)**: the thing DifferentialEquations.jl gives you by letting `p` be a solve-time
argument.

This document records what was **measured** (not assumed), the one architectural
constraint that follows, and the phased plan. Every row of the capability matrix is a
run against a small reaction–diffusion model built through the same `:oop` emitter the
real model uses; the probes are `diffprobe{,2,3,4,5}.jl` and `whilegrad{,2}.jl`, plus
`whilegrad{3,4}.jl`, `loopshape.jl`, `reducegrad.jl`, `maskedloop.jl` and `maskedcost.jl`
in `EarthSciAST/bench/` for the 2026-08-12 loop-shape correction under §1.

Companion reading: HELPERS.md §4 (the same findings in context), and §2 for where each
helper should eventually live.

---

## 1. The capability matrix, measured

| | forward mode | reverse mode |
|---|---|---|
| the `:oop` RHS (straight-line traced) | ✅ exact | ✅ exact vs host ForwardDiff |
| coloured block Jacobian, one traced program | ✅ exact (worst error 0.0) | ✅ VJP exact |
| `@trace` loop, **static** trip count (`for _ in 1:20`) | ✅ exact | ✅ **exact (rel 0.0)** — corrected 2026-08-12 |
| `@trace` loop, trip count a runtime argument | ✅ exact | ❌ `stablehlo.dynamic_pad` untranslatable / no induction variable |
| `@trace while`, **data-dependent** condition | ✅ exact | ❌ *no known iteration count for cache removal* ([Enzyme-JAX #2565](https://github.com/EnzymeAD/Enzyme-JAX/issues/2565), open) |
| host-unrolled fixed step count (no while region) | ✅ | ✅ exact |

> **Correction, 2026-08-12.** Rows 3–5 replace a single row that read
> **`Reactant.@trace while` region — reverse ❌**, and §2 below drew from it the conclusion
> that reverse mode cannot cross a `stablehlo.while` at all. It can: with a static trip
> count the gradient is exact, and `stablehlo.while` is still present in the *differentiated*
> module, so it is genuinely being crossed rather than unrolled away. The discriminator is a
> **compile-time-constant trip count**, not the presence of the region — Enzyme-MLIR's
> `AutoDiffWhileRev` tapes the trajectory as a dense `[N, state…]` tensor, and a non-static
> `N` makes that `tensor<?x…>`, which XLA cannot translate. The bad row came from probe
> `whilegrad.jl` K1, labelled "FIXED trip count" but actually passing
> `RX.ConcreteRNumber(Float64(NSTEP))` as a runtime *argument* (line 55) — fixed in the Julia
> source, statically unknown in MLIR. Checkpointing does not change any of this; see
> HELPERS.md §4 item 3 for the retraction in full and for two `Binomial` hazards.

Also measured, and load-bearing:

* **`p` is a real XLA input, not a trace-time constant.** The same compiled program,
  handed new parameter values, produces new answers matching the host. So a gradient
  does **not** force a recompile per parameter value — decisive, given a ~550 s compile.
* Host ForwardDiff `∂/∂p` works through **both** emitters (`:oop` and `f!`), agreeing
  to rtol 1e-12.
* A `ComponentArray` traces (`to_rarray` → `ComponentVector{…, ConcretePJRTArray}`),
  and a **size-1 slice** read of a component works inside a trace where a scalar read
  raises "Scalar indexing is disallowed".

Two pieces of existing design are what make any of this possible, and must not be
"simplified" away:

* `_rhs_value_type` derives the RHS value type from `u`, `p` **and** `t` together, so
  `p` going `Dual` while `u` stays `Float64` works — which is exactly the shape of
  differentiating w.r.t. parameters.
* `const_tier.jl` deliberately does **not** constant-fold parameter-only subexpressions
  at build time, because freezing them would return a zero derivative for every
  parameter sensitivity: a wrong Jacobian that still looks plausible.

## 2. The constraint that sets the architecture

**Reverse mode needs a statically known trip count, and a macro step has two loops that
lack one** (the SSPRK43 and ROS23 adaptive loops in `rx_traced_integrator.jl`, whose
condition is set by the step-size controller). The `while` region is not itself the
blocker — the *data-dependent trip count* is.

*(Superseded 2026-08-12. This section previously read "Reverse mode cannot cross a
`stablehlo.while` … since even a fixed-trip while loop fails, the blocker is the while
region itself, not adaptivity." The fixed-trip measurement behind that was mislabelled;
see the correction under §1. The practical conclusion for the loops **as written today**
is unchanged — they are data-dependent, so reverse still cannot cross them — but the
remedy is no longer forced to be "remove the loop from the compiled program". Bounding
the loop works too: recast as a `@trace for` over a compile-time attempt cap with
termination as an `ifelse` on the carry, which `adaptive_solve`'s body already uses
throughout. That would put the controller back on the device, and with it would go both
the 3.1e-6 host-vs-device discrepancy in §5 and the whole `(t,dt)`-recording replay
apparatus. Measured on toys, not yet at ReSEACT state sizes.)*

This kills the obvious design — "call `Enzyme.gradient` on the compiled macro step" —
and it matters because reverse is the only affordable mode at this parameter count:

| route | cost for a 24 h CONUS run |
|---|---|
| forward sensitivity, 56 parameters | ~88 h (cost ∝ n_params) |
| forward sensitivity, ≤ 5 parameters | ~8 h — **usable today** |
| reverse adjoint | ~4–5 h, independent of n_params |

So the plan buys the cheap forward result immediately, and separately gets the gradient
path a statically known trip count to unlock reverse. Phases 3–4 below did that by
lifting the loop to the host, which works and is what shipped; per the correction above,
**bounding** the loop in place is the alternative that was wrongly ruled out, and it is
the better end state if it holds at ReSEACT state sizes.

## 3. Phases

Critical path is **1 → 3 → 4**. Phase 2 runs alongside and produces the ground truth
that validates Phase 3.

### Phase 0 — pin what already works (½ day)
There is currently **no parameter-gradient test in the repo**; every AD test
differentiates w.r.t. the state. Add one pinning: ForwardDiff `∂/∂p` through both
emitters agreeing; traced `∂/∂p` and `∂/∂u` vs host; `p`-is-a-real-XLA-input. Add
reverse-through-while as `@test_broken`, so the day Reactant/Enzyme fixes it the suite
says so out loud.

### Phase 1 — the parameter-vector ABI (1–2 days)
Needed by every route. `p` must stop being a `NamedTuple`-only interface.

* `_rhs_value_type(u, p::AbstractVector, t)`.
* Replace the four `getfield(p, n.sym)` sites (`oop.jl` 720/1385/2192, `compile.jl`
  1580) with a named read seam, mirroring the `_oop_read_state` **container seam** that
  already exists for precisely this reason ("XLA rejects scalar indexing of a traced
  array, and needs `u[i:i]`").
* `_NK_PARAM` carries an index; build assigns it from the already-sorted `param_names`
  — the stable order exists, it is just not exposed.
* Return `param_map::Dict{String,Int}` beside `var_map`. Accept `ComponentVector`
  publicly: `getdata` hands an optimizer the dense vector it wants, and the name→index
  axis lives in the type, so it costs nothing at runtime.
* Reactant method on the seam using the size-1 slice.

**Acceptance:** a `Vector`/`ComponentVector` `p` is accepted; traced `∂/∂p` works with
no `@allowscalar`; the Float64 path stays **bit-identical** (the repo already pins
`:oop` against `f!` bit-for-bit); no allocation regression on the scalar parameter read
— `build.jl` warns that a runtime-symbol `getfield` on a NamedTuple boxes the union, so
an index into a homogeneous vector should be strictly better, but measure it.

### Phase 2 — first real sensitivities, forward mode — **DONE** ✅
`tools/sensitivity_forward.jl`. Forward mode already crosses the while regions exactly,
so this needed nothing from Phase 1. The macro step is `@compile`d through
`Enzyme.autodiff(ForwardWithPrimal, …)` and the tangent is loop-carried across the
macro-step window alongside the state — **including the two step-size tangents**,
because the adaptive controller carries `dt` across macro steps and `dt` is a function
of `u` and `p`. The parameters ride in as a tuple argument merged back into `p` inside
the trace, so ONE compile serves every parameter (a one-hot seed picks the direction)
AND every finite-difference evaluation (perturbed values, zero seed) — which is only
affordable because `p` is a real XLA input.

**Acceptance: MET, at CONUS.** 13×7×72 (85,176 states), `T0 = 5400`, three 300 s macro
steps, objective = domain-mean **surface** O₃ at the end of the window (39.6531 ppb):

| parameter | ∂J/∂θ | units | best fd/AD rel | at h |
|---|---|---|---|---|
| `NEIRegrid.scale` | −7.387076375067e−02 | ppb per unit scale | **5.5e−9** | 1e−4 |
| `Transport3D.tau_pblmix` | −6.906232719590e−05 | ppb per second | **1.6e−7** | 9e−2 |
| `NEIRegrid.g0` | −7.532721546163e−03 | ppb per m s⁻² | **7.5e−9** | 9.8e−4 |

All three negative, and that is the physics: the window sits at night, when NOx titrates
O₃, so more emissions — or slower PBL mixing, which keeps fresh NO at the surface —
means *less* surface ozone. The accept/reject pattern was unchanged at every h tried, so
no controller branch flip is in play. In elasticity terms a +1% emissions perturbation
moves 15-minute domain-mean surface O₃ by **−0.00186%**.

Costs at CONUS: build 770 s, JVP `@compile` 770 s (vs 557 s for the primal alone — the
forward program is ~1.4× the compile, not the feared 2×), ~124 s per 3-macro-step window
pass. The whole run — 1 baseline + 3 AD sweeps + 24 fd evaluations — is ~2 h wall.

Four things worth carrying forward:

* **A structural identity pins the chain in a way fd cannot.** Every NEI emission rate
  in `reseact.esm` is literally `∝ scale · g0 / delp`, and `g0` appears nowhere else, so
  `scale·∂J/∂scale ≡ g0·∂J/∂g0` exactly. Two independently seeded forward sweeps satisfy
  it to **2.3e−13** relative (7.8e−15 at the small grid). A seeding or loop-carry error
  that corrupted both sweeps identically would still break that ratio.
* **Kink contamination is a small-domain effect.** The same driver at 6×6×8 (3,744
  states, 36 surface cells) floors at **~5e−7** fd/AD agreement, flat across four
  decades of h — neither the `h²` of truncation nor the `1/h` of cancellation, which is
  what §4's first caveat looks like quantitatively: a fixed handful of kinked states
  (`clamp_nonneg`'s `max(u,0)`, the `max(|u|,1e-9)` in the FD block-Jacobian step, the
  controller's `q` clamps). At CONUS the floor is ~100× lower, which is consistent with
  those kinks averaging out over ~100× more states — an inference from two grid points,
  not a measurement of the mechanism. Practical consequence: **check Phase 3's adjoint
  against the AD number, not against fd**, and do it at a realistic domain size.
* **`NEIRegrid.F_NO`…`F_FORM` are NOT split fractions and are NOT differentiable here.**
  They are the whole NEI2016 12US1 source-grid emission *fields*, wired in as CONST
  provider arrays and collapsed at build time by the conservative regrid. They never
  reach the runtime `p` (49 scalars, nothing else). Phase 5's acceptance list below has
  to move them out of "numeric", or the regrid has to become a runtime operation first.
  `NEIRegrid.g0` stands in as the third parameter.
* **`Transport3D.tau_pblmix` barely moves with domain**: −6.892e−05 at 6×6×8 against
  −6.906e−05 at CONUS, 0.2% apart, as a local PBL process should be. `NEIRegrid.scale`
  moves a lot (−1.115e−01 vs −7.387e−02) because the domain, the emissions map and the
  truncated 8-level column are all different — so quote the emissions sensitivity with
  its grid attached.

### Phase 3 — a discrete adjoint of the step (1–2 weeks; the real work)
Write the **VJP of the ROS23 / SSPRK43 stage algebra by hand**, so the reverse sweep is
a host loop over macro steps and each step's VJP is straight-line traced code — which
reverse mode handles exactly (measured).

The rejected alternative was host-unrolling the adaptive loop to a fixed step count.
Reverse does work through that (measured, exact), but it is unlikely to be numerically
efficient: a fixed step count either wastes work where the adaptive controller would
have taken long steps or loses accuracy where it would have taken short ones, and the
trace grows with the unroll factor — XLA's simplifier already warned of a circular
simplification loop at 100 unrolled steps on a *toy* model, which is the same family of
pathology as the `cse_slice` blowup that cost weeks here.

Fold in the free win: the traced block Jacobian is **finite difference** today. Coloured
forward JVPs give it exactly (measured, worst error 0.0) at similar cost, removing FD
contamination from every gradient and probably improving the forward solve too — the
native arm's switch to `block_ad_jac` cut rejected steps.

**Acceptance:** matches Phase 2's forward sensitivity to ~1e-8 on the parameters both
can do.

#### Phase 3 status — what landed, and the one wall

Landed in `tools/reactant_handoff/rx_traced_integrator.jl`, validated by
`tools/rx_adjoint_check.jl` (which builds ReSEACT through the same path
`run_reseact_reactant.jl` uses):

* `ros23_step_vjp` / `ssprk43_step_vjp` — one step's `(∂u_out/∂u_in)ᵀλ` and
  `(∂u_out/∂θ)ᵀλ` as ONE traced program, plus matching `*_jvp` for the
  dot-product identity. `θ` is the RHS's differentiable payload made an explicit
  argument (the production call site hides it in a closure, and a closure is
  opaque to Enzyme); a nested `NamedTuple` of traced scalars **and** arrays
  works, so all 49 runtime parameters and the forcing buffers come back at once.
* `ad_block_jac` — the exact block Jacobian by coloured forward-mode JVPs,
  replacing finite differences. `ros23_step(...; jac=:ad)`.

**The route is Enzyme reverse over a straight-line step body, not a hand-derived
stage adjoint.** The tableau part of the adjoint is mechanical, but `W` is built
from `J(u)`, so the exact step adjoint needs `d(Jv)/du` — a second derivative of
an arbitrary emitted RHS. Enzyme has that; hand-derivation could only introduce
error. The Phase 3 constraint ("no while region in the differentiated path") is
then met by *construction* — route through `ssprk43_step_unrolled` and
`ros23_step` with a host-unrolled Jacobian — rather than by rewriting algebra.
Verified by grepping the raw (pre-optimization) MLIR: every module in the
differentiated path has **`stablehlo.while` = 0**, against **2** in the
`adaptive_solve` control.

**THE WALL: reverse-over-forward SEGFAULTS on the real RHS.** With `jac=:ad` the
step contains a nested `enzyme.fwddiff`, and reverse over it dies inside
Enzyme-MLIR — `AutoDiffCallRev::createReverseModeAdjoint` → `func::CallOp::build`
→ `getAttr` on a **null** FuncOp, from `DifferentiatePass`. It is a crash, not a
diagnostic, and it is upstream, not a limit of this design: the identical nesting
is exact on a small model (dot-product test 1.8e-16) and survives every
helper-minting construct ReSEACT's RHS uses — `tools/rx_adjoint_toy.jl <variant>`
runs that isolation in about a minute. Plain reverse (`jac=:fd`) compiles
and runs on ReSEACT. So `ros23_step_vjp` defaults to `jac=:fd`, and until that is
fixed **the exact Jacobian is a forward-solve win only** — the adjoint still
carries the FD contamination of §4 item 4. Worth filing upstream next to the
reverse-over-`while` report.

> **Re-established on a much smaller reproducer, 2026-08-12 — and note the evidence
> behind the paragraph above was bad even though its conclusion was right.**
> `tools/rx_adjoint_check.jl:294` `ros_vjp` omitted the `jac` kwarg while
> `ros23_step_vjp` defaults `jac::Symbol=:fd`, so the row printed as
> `"ROS23 (chemistry, jac=:ad)"` ran the **FD** VJP — byte-identical modules after
> renaming. That row never exercised `:ad`. Re-run properly, the segfault reproduces,
> and in far less code than a Rosenbrock step: reverse over the coloured forward-mode
> Jacobian **alone** is sufficient —
> ```julia
> Jb = ad_block_jac(uu -> gC(uu, th, t), u, Val(13), NC)   # 13 coloured JVPs
> sum(sum(Jb[r,s]) for r in 1:13, s in 1:13)               # reverse-differentiated
> ```
> Five on-model probes, all `EXIT=139`, all on the **default pass pipeline** with no
> `excluded_passes` — so the separate `concat_broadcast_slice` miscompile (below) did not
> contaminate any of them — and all with an identical frame sequence:
>
> | stage | program | colours | build | verdict |
> |---|---|---|---|---|
> | `jacrev` NCOL=1 | the Jacobian alone | **1** | 595.3 s | SIGSEGV |
> | `jacrev` NCOL=13 | the Jacobian alone | 13 | 601.9 s | SIGSEGV |
> | `ros_advjp` | full `jac=:ad` ROS23 step VJP (the original question) | 13 | 644.4 s | SIGSEGV |
>
> `ros_advjp` agreeing is what makes the Jacobian-only form a reduction of the *real* bug
> rather than a different one. And **NCOL=1 crashing reduces the upstream reproducer to
> "one `enzyme.fwddiff` inside one reverse pass"** — at K=1, `ad_block_jac` is a single
> `Enzyme.autodiff(Forward, Const(f), Duplicated, Duplicated(u, seed))`. That turns an
> absence into a finding: the toy survives **13** nested `enzyme.fwddiff` ops, the model
> dies with **1**, so the nesting count is not the trigger on either side — the difference
> is the RHS, not the derivative structure. Stack matches the predicted mechanism frame for
> frame:
> `DifferentiatePass::runOnOperation → lowerEnzymeCalls → MEnzymeLogic::CreateReverseDiff
> → differentiate → visitChild → ReverseAutoDiffOpInterface::createReverseModeAdjoint
> → AutoDiffCallRev::createReverseModeAdjoint → func::CallOp::create → func::CallOp::build
> → mlir::Operation::getAttr` ← fault. `func::CallOp::build` calls `callee.getNameAttr()`
> on `callReverseHandler`'s **unchecked** `revFn`, so `CreateReverseDiff` returned failure
> for some callee, silently, with no diagnostic first. *Which* callee is not
> determinable from outside the process — the jll has no debug symbols and nothing
> prints before the fault. That is precisely the argument for the one-line null check
> upstream: with an `emitError()` there, the callee would name itself.
>
> Separately, a **second and independent** upstream bug found on the way, reproducible
> with no Enzyme and no autodiff at all — the `concat_broadcast_slice` HLO pattern
> miscompiles: `rowstack(x) = vcat([reshape(x[(k*5+1):(k*5+5)],1,5) for k in 0:12]...)`
> fails the verifier by default and passes with
> `excluded_passes=["concat_broadcast_slice"]`. Root cause: `mergeConcatSlicedElems`
> (Enzyme-JAX `src/enzyme_ad/jax/Utils.cpp`) takes a `concatDim` parameter and never
> uses it, so it merges adjacent slice operands without checking the slice axis is the
> concat axis. This is what breaks reverse-mode compiles at NS ≥ 5.
>
> Also retracted: an earlier `EXIT=137` on `ros_ad` was read as evidence about the AD
> JVP. It was a memcg OOM-kill from running five ReSEACT builds in one 40 GiB cgroup,
> compounded by reading node-wide `/sys/fs/cgroup/memory.stat` instead of the step
> cgroup. Not a signal.

Also measured, and it changes an expectation: at 6×6×8 over one 300 s macro step
the exact Jacobian is compile-neutral (89.3 s vs 87.3 s) and **step-count
neutral** (120 accepts / 4 rejects vs 116 / 3; a second run gave 116 / 3 for
both, so the counts are not even stable to that resolution — which is now known to
be the XLA:CPU threading race of FINDING 2, spuriously rejecting steps, not a
resolution limit and not ulp amplification). The native arm's `block_ad_jac` cut
rejected steps; this did not, at this size. `RESEACT_RXJAC=ad` exists so it can
be measured at CONUS, but the default stays `fd` until it is.

#### Phase 3 — what is NOT verified, and the next question

* **A systematic 9.2e-6 disagreement between forward-mode AD and finite
  differences of the SSPRK43 step in the STATE direction**, concentrated in
  `SuperFast.CO` (91%) and `SuperFast.O3` (9%). It is not a kink: the forward,
  backward and central quotients agree with each other to every digit printed
  and the residual is flat from eps 1e-2 to 1e-6, so AD and the true derivative
  differ by a fixed vector. It is also **not** an adjoint error — the
  dot-product identity in that same direction is 5.7e-16, so the VJP is the
  exact transpose of whatever the JVP computes. That makes it a question about
  the derivative of the emitted `:oop` RHS, not about the stage algebra: the
  obvious suspect is a closed discrete function contributing no partials by
  contract (§4), and the test that would settle it is Phase 0's missing one —
  traced `∂/∂u` of the RHS against host ForwardDiff, per state group.
* **Whether the exact Jacobian removes the parameter-direction FD disagreement**
  is unmeasured. With `jac=:fd` the FD check in the θ directions has no clean
  eps window at all (best 2e1 relative in the parameter direction, 9e-4 in the
  forcing-buffer direction) — consistent with the FD difference quotient's own
  noise floor sitting above the signal, which is exactly what the exact Jacobian
  should fix. The forward-only probe for it (`RESEACT_ADJ_STAGES=ros_adfwd`,
  which needs no reverse pass and so dodges the segfault) never finished: one
  attempt was OOM-killed by the shared cgroup, and a second was abandoned after
  **85 minutes inside XLA** with the trace long done — against **141 s** for the
  same JVP of the `jac=:fd` step, and 89 s for the `jac=:ad` step's own primal.
  So nesting AD around `ad_block_jac` has a compile-cost problem *independent of*
  the segfault, and a second attempt should probably start by removing the
  nesting — Enzyme's batched forward mode over the `Val(NS)` colours, or an
  `ad_block_jac` written directly against `Reactant.Ops` — rather than by waiting
  for the upstream crash to be fixed.

  **The mechanism is now measured, not argued** (2026-08-12, `addump` op census of the
  genuine `jac=:ad` module):

  | module | `enzyme.fwddiff` | `func.func` | size | trace |
  |---|---:|---:|---:|---:|
  | `jac=:ad` | **13** | 1359 | 120.1 MB | 148.5 s |
  | `jac=:fd` | 0 | 1346 | 34.8 MB | 24.2 s |
  | chemistry RHS alone | 0 | 85 | 8.1 MB | 4.3 s |

  3.5× the size and 6.1× the trace time, for +13 `func.func` against a standalone RHS of
  8.1 MB. So `ad_block_jac`'s NS `enzyme.fwddiff` ops do **not** share one differentiated
  callee — each colour carries roughly its own copy of the RHS, where the FD Jacobian is
  NS *calls* into one copy. That is the cost mechanism. Boundary kept explicit: this is
  the trace/size half only; the pass-pipeline and XLA time on top of it remains
  unmeasured, and the "85 minutes, did not finish" above was not reproduced.

  The census also confirms the AD label **on the model** rather than only on a toy — 13
  `enzyme.fwddiff`, one per species colour, against 0 for FD — and closes the harness bug
  numerically: the old mislabelled `_ad` dump is 34,764,311 bytes against this `fd`
  module's 34,764,314, a three-byte delta that is exactly the length of the function name.

  Two routes to remove the nesting are both blocked upstream: `BatchDuplicated` leaves an
  `enzyme.extract` that XLA export rejects, and `Ops.batch` around a `fwddiff` emits a
  rank-mismatched transpose. "Build the columns directly against `Reactant.Ops`" has no
  target, because the only Ops-level derivative primitive *is* `enzyme.fwddiff`.

### Phase 4 — the time-loop adjoint driver (~1 week)
Forward pass checkpoints `(u_k, t_k, forcing epoch)`; backward sweep accumulates
`λ_k = (∂step/∂u)ᵀ λ_{k+1}` and `g += (∂step/∂p)ᵀ λ_{k+1}`.

Checkpointing is cheap enough to be uninteresting: 85,176 states × 8 B = 681 KB/step →
**196 MB for 24 h, 1.4 GB for a week**. Store every step; no Griewank scheme needed.
The forcing refresh is a clean boundary — meteorology is `Float64` data with zero
derivative (which is what you want: you are not differentiating w.r.t. GEOS-FP), so the
backward sweep replays it from the checkpointed epoch.

**Acceptance:** ∂(24 h CONUS mean O₃)/∂p for the full numeric vector, finite-difference
checked on 2–3 components.

#### Phase 4 status — a working whole-window gradient, and two nondeterminism findings

Landed as `tools/adjoint_gradient.jl`. It produces **dJ/dθ for all 49 runtime scalar
parameters in ONE backward sweep**, and the cost does not depend on how many parameters
you ask for — which was the entire point of the phase.

Everything below is at **6×6×8** (3,744 states), `T0 = 5400`, three 300 s macro steps,
objective = domain-mean surface O₃ at the end of the window, at a **jittered base point**
(`RESEACT_ADJ_UJITTER=1e-1`, seed 31337 — the same stream `rx_adjoint_check.jl` uses, so
"the jittered base point" is literally the same point in all three drivers), with
`RESEACT_ADJ_CLAMP=0`. **It has not been run at CONUS.**

**The architecture.** Reverse cannot cross a `stablehlo.while`, so the whole adaptive
controller — stage attempts, PI update, accept/reject, `clamp_nonneg` — is lifted onto
the host, and the only compiled objects are single steps at a fixed `dt`. A macro step is
2.6 s forward; the backward sweep is 13.2 s per macro step, i.e. **5.1× the forward
pass** (4.6× VJPs at 0.14 s each over 251 calls, 0.6× the replay). Compiles: `ssp_step`
80 s, `ros_step` 92 s, `ssp_vjp` 139 s, `ros_vjp` 164 s.

**Acceptance, measured.** Against central finite differences of the *frozen-dt* composed
map — the map the discrete adjoint is by construction the derivative of — through the
same compiled single-step programs, on the same 12-transport + 239-chemistry step
sequence, whose replay at θ₀ reproduces the forward pass's objective **bit-identically**:

| parameter | adjoint ∂J/∂θ | best fd rel | at h |
|---|---|---|---|
| `NEIRegrid.scale` | −8.008172497475e−02 | **4.3e−10** | 1e−3 |
| `Transport3D.tau_pblmix` | −4.112488570218e−04 | **2.8e−9** | 9e−3 |
| `NEIRegrid.g0` | −8.166063332005e−03 | **5.5e−11** | 9.8e−3 |

The structural identity `scale·∂J/∂scale ≡ g0·∂J/∂g0` holds on the adjoint gradient to
**7.9e−13** (and 3.6e−12 on a second run). That is looser than Phase 2's forward sweeps
(7.8e−15 at this grid) and it should be: the adjoint accumulates 251 separately-rounded
VJP contributions into each component, where a forward sweep carries one chained tangent.

**Against Phase 2 directly the agreement is 3.1e−6, not 1e−8, and that is expected.**
`sensitivity_forward.jl` at the same base point gives −8.008197329598e−02 /
−4.112291264030e−04 / −8.166088653717e−03. Three confounders, each larger than 1e−8,
separate the two and none of them is an error in either:

* it **differentiates the controller's `dt`**; a discrete adjoint holds `dt` fixed;
* it runs the **clamp**, which the adjoint cannot (below);
* its trajectory is the **device while-loop's**, not the host loop's. Measured over one
  macro step (`ctl` stage): identical accept/reject counts (5/0 transport, 117/3
  chemistry) but states 8.8e−16 relative apart for transport and **2.8e−10** for
  chemistry. The chemistry half runs the *same Julia source* in both arms, so that is
  XLA's: the same stage algebra compiled standalone and compiled inside a `while` body
  are not the same floating-point program. It compounds to 3.8e−6 in J over three macro
  steps, the same order as the gradient gap. **This is a floor on how well any
  host-lifted adjoint can match a device-loop forward sensitivity**, and it is why the
  acceptance check above uses a reference computed with the adjoint's own programs.

**FINDING 1: re-running the adaptive controller does not reproduce the forward pass.**
A replay of macro step 2 took 95 accepts / 2 rejects where the forward pass took 92 / 1
— same checkpoint, same θ, same forcing, same process. The compiled ROS23 step is not
bit-deterministic call to call — **but not for the reason recorded here until
2026-08-12.** It is not ulp amplification: across >400,000 calls there was never a
ULP-level difference. It is the XLA:CPU threading race of FINDING 2, which NaNs `EEst`,
and `host_adaptive!` maps `isnan(EEst)` to a rejection — so the extra rejects are
SPURIOUS. Phase 3 saw the same fault *between* runs. A checkpoint-and-replay
adjoint that re-derives its own step sequence is therefore silently taking the adjoint
of a different trajectory. The fix is cheap and is now the design: the forward pass
records the accepted `(t, dt)` of every inner step — **16 B each, against 681 kB for the
state** — and the replay replays that sequence with the controller off. The replayed
states then match every checkpoint to **0.000e+00**.

**FINDING 2: the compiled programs intermittently return non-finite values, and it is
not reproducible.** Across six sweeps of ~251 VJP calls, two acquired non-finite λ
partway back (once with the clamp on, once with it off), always as 312 entries at once =
24 cells × all 13 state groups. The `PROBE` stage re-runs the exact failing VJP — same
`u`, λ, `t`, `dt`, θ — and gets a **finite** answer, for four different seeds including
all-ones and all-zeros, at a step whose primal is finite, with no exact zeros in `u`, and
at 0.1× and 0.01× the step size. It is **not confined to reverse mode**: in the same run
that produced the acceptance table above, 2 of the 30 pure-primal frozen replays in the
`fdtape` sweep returned a NaN objective, and re-running the identical replay produced a
finite one. So this is the compiled ReSEACT step program on this box, not the adjoint.
The driver re-issues the identical call on detection and **reports the retry count** (0
of 251 in the accepted run above); that is a workaround for a nondeterministic fault, not
a fix, and it is the first thing to pick up. `clamp_nonneg` was the initial suspect and
is **not** the cause (it moves the objective by only 6.7e−11 relative, but the sweep is
still run with `RESEACT_ADJ_CLAMP=0` because the clamped sweep failed twice).

> **DIAGNOSED, 2026-08-12: it is a data race in XLA:CPU's intra-op parallel execution.**
> Reproducible on demand in ~5 minutes — see `tools/diag/README-nondet.md`.
>
> | XLA threads | fault rate, chemistry RHS, 40,000 calls/cell |
> |---|---|
> | 1 (`taskset -c 8`) | **0 of 40,000** — and 0 of 400,000 across ten compile variants |
> | 4 (`taskset -c 8-11`) | 310 and 437 of 40,000 ≈ **1 %** |
> | 20 (unpinned) | 19, 28 of 40,000 ≈ 5–7 × 10⁻⁴ |
>
> The emitted module is a pure StableHLO dataflow graph — zero `while`, `rng`,
> `custom_call`, `sort` — so a call-to-call difference on identical inputs is not a
> numerics question: XLA is executing a pure function wrongly, and it needs concurrency
> to do it. Narrowed to the **fusion emitters at 256-bit vector width**: at 4 threads,
> `xla_cpu_use_fusion_emitters=false`, `xla_cpu_experimental_ynn_fusion_type=[]` and
> `xla_cpu_prefer_vector_width=128` each give 0 of 200,000 against an expectation of
> ~800, while `multi_thread_eigen=false`, `use_xnnpack=false` and `max_isa=AVX2/SSE4_2`
> all still fault at full rate.
>
> **The shape never varied:** NaN in exactly the six dry-deposition species
> (`O3, NO2, NO, HNO3, H2O2, CH2O` — precisely `superfast_deposition_sink.esm`'s list),
> at one grid cell, essentially always lane 1 = `(1,1,1)`, the first surface cell, with
> every other entry bit-identical. Chemistry RHS only; the transport RHS is 0 of 70,000.
>
> **FINDING 1 and FINDING 2 are the same fault.** Of 32 faulting `ros23_step` calls, all
> 32 returned `EEst = NaN` and most returned a bit-identical *state* — because
> `f2 = f(unew, t+dt)` feeds only `k3`, i.e. only the error estimate. `host_adaptive!`
> maps `isnan(EEst)` to a rejection, so the commonest consequence is a **silent spurious
> rejected step**, which is exactly the 95/2-vs-92/1 replay signature. A check that looks
> only at the state undercounts the fault by ~4×.
>
> **Workaround**, per-compile, no rebuild:
> `Reactant.CompileOptions(; sync=true, xla_debug_options=(; xla_cpu_prefer_vector_width=128))`.
> Confirmed on the full `ros23_step`: 0 of 20,000 against a baseline of 4 and 4, at no
> wall-time cost. **Not numerically free** — it moves the answer by up to 1.6e-6 relative
> on one step, so treat it as a new floor rather than a null change. Pinning XLA:CPU to
> one thread also works and changes no numbers at all, at the cost of the thread pool.
>
> **Ruled out with evidence:** `clamp_nonneg`; the integrator algebra, FD Jacobian, block
> solve and reverse mode (the RHS *alone* faults harder than the step); the inputs;
> argument donation; the host reading a buffer early; stale/uninitialised allocator memory
> (poisoning Julia's heap *and* PJRT's own allocator moved nothing — this was the leading
> hypothesis and it failed); machine load.
>
> **Still unexplained:** which fused kernel; why lane 1 essentially always; why chemistry
> and never transport; and why the rate is *higher* at 4 threads than at 20. **Method
> caution:** `parallel_codegen_split_count=1` measured 330 then 2 faults per 40,000 in
> consecutive passes of the *same* executable — a fixed program's rate swings two orders
> of magnitude, so single-pass comparisons here are worthless. Closest upstream matches:
> jax-ml/jax#39741, openxla/xla#32974; XLA publishes no CPU determinism guarantee
> (`openxla.org/xla/determinism` is GPU-only).

**NOT VERIFIED, and worth stating plainly:**

* **CONUS.** Everything above is 6×6×8. A CONUS build is ~740 s and the VJP compiles
  would be several times the 139/164 s measured here.
* **A window that crosses a GEOS-FP refresh.** The three-macro-step window at `T0=5400`
  contains **zero** forcing boundaries, so the epoch-replay path — checkpoint the epoch,
  re-`refresh_forcing` on the way back — is implemented and exercised, but the *changing*
  of the epoch mid-window has never actually happened in a validated run.
* **The forward-mode reference does not work on this RHS.** `Enzyme.autodiff(Forward,
  …, Duplicated, …)` comes back with a single slot, and under an exactly zero seed that
  slot has norm ≈ ‖u‖ — it is the primal, not the tangent, so a chained forward reference
  built on it returns the state instead of its derivative (‖du‖ came back as 2.3e4 for a
  parameter that does not appear in the transport RHS at all, and the per-step
  dot-product identity read rel = 1.2–1.6). The same call shape returns the correct
  tangent on a toy carrying the same constructs. The driver detects this by the
  zero-seed test and skips the stage rather than reporting the state as a derivative.

  **The inference this originally carried — that `rx_traced_integrator.jl`'s
  `ros23_step_jvp` / `ssprk43_step_jvp` are therefore suspect, and Phase 3's results
  with them — was CHECKED AND IS WRONG.** Two measurements clear them:
  * `autodiff(Forward, f, Duplicated, Duplicated(x,dx))[1]` is the DERIVATIVE, host and
    traced, on a function where primal and tangent are unmistakable; the same call under
    an exactly zero seed returns exactly zero.
  * On the REAL model at a jittered base point, CHECK 2 — which calls
    `ssprk43_step_jvp` directly — satisfies `⟨λ,Jv⟩ == ⟨Jᵀλ,v⟩` to **6.7e-16 (state),
    1.0e-15 (params), 9.7e-16 (forcing), 3.4e-16 (all)**. If the JVP returned the primal
    those two numbers could not agree to sixteen digits in four independent directions.

  So the bad slot is real in the *chained forward reference built here*, and its cause is
  in that construction, not in the merged JVPs. Worth resolving, because a working
  forward reference is what would let criterion 1 be answered directly instead of through
  FD on a frozen tape — but nothing already validated needs revisiting.
* **The 1e−8 target against Phase 2 itself** — for the reasons above, that comparison
  cannot reach 1e−8 while the host loop and the device loop are different floating-point
  programs. Closing it needs either reverse-over-`while` upstream, or an
  `adaptive_solve` whose body is compiled identically inside and outside the region.

### Phase 5 — structural/numeric partition + the SciML surface (~1 week)
**Structural = its value changes the shape of the problem.** You cannot differentiate
w.r.t. something that changes the dimension of the answer — the sharpest case is
`value_invention`, where a scalar parameter like a grid spacing is read by `_vi_param`
to invent an index-set extent, i.e. it decides `length(u)`.

Today `simulate(prep, tspan; parameters=…)` **throws** on any override, because
parameters bake at `prepare()` time (they feed setup geometry, value-invention extents,
binning coordinates, and `ic()` folds). That is the honest behaviour — it refuses rather
than silently baking — but it is too broad. Narrow it:

* Make the partition automatic. `_resolve_param_scope` already hands `param_scope` to
  the four build-time consumers (`_fold_ic_equations`, `_fold_field_ics!`,
  `_materialize_geometry_setup`, `_vi_param`); have them **record which names they
  read**. Touched at build ⇒ structural. Note it is per-*name*, not per-use:
  `lon0_deg` is structural for NEIRegrid's ring builder even though Transport3D's solar
  chain also reads it at runtime.
* `remake(prob; p=…)` then keeps SciML semantics for the numeric half — cheap and
  AD-transparent, which is *why* SciML's `remake` swaps `p` rather than rebuilding `f`.
  Overloading it to re-run `prepare` would make gradients impossible by construction.
  A structural change is an explicit re-`prepare`, not something hidden inside `remake`.

**There are THREE categories, not two.** An earlier draft of this document had only
structural and numeric, and put `F_NO`…`F_FORM` in "numeric" on the strength of their
being declared `"type": "parameter"` in the `.esm`. Phase 2 found that wrong by trying
to differentiate one. The missing category:

* **structural** — read at build; its value changes the SHAPE of the problem (index-set
  extents, geometry weights, which cells overlap). Not differentiable; changing one is a
  rebuild.
* **numeric** — a scalar in the runtime `p`. Differentiable. This is what a parameter
  vector exposes.
* **const-folded data** — declared a parameter, but supplied by a CONST provider and
  collapsed into `const_arrays` at build (`EarthSciIO.const_provider`, wired in
  `split_common.jl`). It never reaches `p` at all. Differentiating w.r.t. one returns an
  unconditional zero — and, the reason this category has to be named rather than lumped
  in with structural, **a finite-difference check would "confirm" that zero**, because
  perturbing the declared default changes nothing either. A wrong gradient and a wrong
  check agreeing silently is the worst failure mode available here.

Declared type is therefore NOT the discriminator; where the value is CONSUMED is. The
automatic partition must classify by consumption, and must distinguish folded
arithmetic from folded data — they want different error messages.

### Phase 5 — **DONE** ✅, and it corrected this list twice more

`parameter_classes(prep | insp) -> Dict{String,Symbol}`. The partition is recorded, not
declared: a dynamically scoped sink is installed around the build-time consumers and
read at the **three places a name is actually resolved to a build-time value**
(`_compile(::VarExpr)`, `_geo_compile(::VarExpr)`, `_vi_param`). Recording at the
resolution site rather than at the parameter dict is load-bearing — every consumer
materializes the WHOLE parameter scope into a NamedTuple before compiling, so a
read-recording dict would have reported every parameter structural. Only the compiler
knows which names an expression mentions.

**FOUR classes, not three.** `:forcing` was added for live GEOS-FP buffers that a
discrete provider rewrites in place. Calling one "const-folded" would be false in
exactly the way this section warns against, and the two want different remedies —
a const-folded field needs the regrid made runtime, a forcing buffer just needs
`prep.param_buffers[…]` written.

**Measured on `reseact.esm`** (6×6×8, |p| = 49): **35 `:numeric` + 14 `:structural` =
49 = |p| exactly**, plus 9 `:const_folded` and 15 `:forcing`, with no `p` slot
unclassified. A solve-time numeric override produces a `du` **bit-identical** (max |Δ|
exactly 0.0) to a second full build with the value baked in, while differing from
baseline in 159/3744 slots.

**The list below was wrong twice, and the second reason is the instructive one.**
* Written from the `.esm`, it put `F_NO`…`F_FORM` under "numeric" because they are
  declared `"type": "parameter"`. Phase 2 corrected that: they are const-folded data.
* Phase 5 corrected it again: **`NEIRegrid.F_NO`…`F_FORM` and `NEIRegrid.lev` are not
  variables of the built model at all.** `variable_map` couplings with
  `transform: "param_to_var"` replace them at flatten time —
  `NEI2016Emis.NEI2016.{NO,NO2,CO,ISOP,FORM} → NEIRegrid.F_*`, and
  `Transport3D.levc → NEIRegrid.lev`. So the emission fields survive under their
  LOADER names (which do classify `:const_folded` — the physical claim was right, only
  the spelling was pre-coupling), and `lev` becomes a runtime observed, so it is not
  structural and not a parameter.

The general lesson, having now been wrong three times in three different ways: **an
acceptance list written by reading `.esm` names is not evidence.** Declared type does
not determine class, and the name you read may not survive coupling. Classify by
building the model and asking it.

**Acceptance on `reseact.esm`, as measured:**
* **numeric**: `NEIRegrid.scale`, `NEIRegrid.g0`, `Transport3D.tau_pblmix`, `Rd_air`,
  `g_acc` (and `Transport3D.lon0_deg`/`lat0_deg` — the per-name split the three-category
  note predicts, since the NEIRegrid copies are structural and the Transport3D ones are
  read at runtime by the solar chain).
* **structural**: the `NEIRegrid.*` geometry — `lon0_deg`, `lat0_deg`, `dlon_deg`,
  `dlat_deg`, `src_x0/y0/dx/dy`, `lcc_lat_1/2/0`, `lcc_lon_0`, `lcc_R`, `atol`.
* **const-folded data**: `NEI2016Emis.NEI2016.{NO,NO2,CO,ISOP,FORM}` — the whole NEI2016
  12US1 source-grid emission fields, collapsed by the conservative regrid at build.
  Making these differentiable is separate, larger work: the regrid would have to become
  a runtime operation. Until then per-species emissions sensitivity is reachable only
  through the scalars multiplying the folded field (`scale`, `g0`).
* **forcing**: the live GEOS-FP buffers, e.g. `GEOSFP.GEOSFP_I3.PS`.

## 4. Cross-cutting caveats

* **THE TRANSPORT STEP IS ONLY PIECEWISE DIFFERENTIABLE, AND THE DEFAULT IC SITS
  EXACTLY ON A SWITCH.** The most important entry here, because it makes the obvious
  validation procedure return the wrong verdict. ReSEACT's transport is
  monotonicity-limited PPM, and every limiter switch in EarthSciDiscretizations is a
  PRODUCT OF NEIGHBOUR DIFFERENCES compared with zero — CW84 eq.(1.8)
  `ifelse((ap-a0)*(a0-am) > 0, …, 0)` in `ppm_slope_mono.esm` / `ppm_lev_slope_mono.esm`,
  and eq.(1.10) in `ppmflux_limit_left.esm` / `ppm_limit_right.esm`.

  Every SuperFast field in the default IC is EXACTLY spatially uniform — visible in the
  first digest row of any run, where `O3_min == O3_mean == O3_max == 40.00000`. On a
  uniform field every guard product is exactly 0, so `u0` sits ON the switching surface
  of essentially every interior cell.

  **And the usual kink test cannot see it.** The guard is *quadratic* in the
  perturbation: at `u0 ± e·v` it equals `e²·(dv_{i+1}-dv_i)(dv_i-dv_{i-1})` — the SAME
  SIGN on both sides. Both one-sided quotients flip to the same other branch, so
  forward, backward and central all agree with each other and the residual stays flat as
  `eps` shrinks. `fwd ≠ bwd` is sufficient for a kink but **not necessary**, and this is
  the case that proves it. Measured: at `u0` the AD-vs-FD disagreement is 100%; displace
  the base point by ±1e-14 *in either direction* and AD matches FD to 2.2e-13
  (`tools/diag/ppm_limiter_kink_repro.jl` — host-only, ~40 lines of `.esm`, runs in
  seconds).

  Two consequences, both load-bearing. AD returns the exact derivative of the ACTIVE
  BRANCH, so gradients are correct almost everywhere — this is the familiar 4D-Var
  situation with limited advection, not a defect. But **never finite-difference-validate
  at the default IC**: use a spun-up or jittered state (`RESEACT_ADJ_UJITTER=1e-1`,
  `eps ≤ 1e-6`), and prefer checking an adjoint against forward-mode AD over checking it
  against FD at all.
* **`clamp_nonneg` zeroes gradients where it bites.** It is a nonlinear edit of the
  state; a clamped component contributes nothing. Offer a gradient mode without it.
* **Discrete closed functions contribute no derivative by contract** —
  `interp.searchsorted` and the calendar `datetime.*` accept a `Dual` and return zero
  partials. Correct, but a parameter reaching the objective *only* through one of them
  is silently zero-gradient rather than an error. Add a diagnostic.
* **Enzyme on the CPU needs `Enzyme.API.strictAliasing!(false)`**, because the walk
  loads fields from `_VecNode`, heterogeneous by design (`payload::Any`). Not on the
  critical path — the traced route does not need it — but the durable fix (a
  payload-free, concretely-typed lowered IR) is the same lowering a device backend
  wants anyway.
* **File reverse-over-while upstream.** If Reactant/Enzyme gains it, Phase 3 collapses
  to nothing. Worth reporting regardless of whether we wait for it.

---

## 5. Two properties of the traced pipeline, established while reviewing Phase 3

**The traced pipeline is BIT-REPRODUCIBLE run to run.** Two independent CONUS
13×7×72 runs of the same code, in different processes, agree to `0.000e+00` on every
digest column across all aligned records. This was not previously known — an earlier
guess in this session attributed a difference between two runs to "XLA
nondeterminism", and that guess was wrong (the two runs being compared had different
Jacobians). The practical consequence is large: **any difference between two traced
runs of the same configuration is signal, not noise**, so a 4-row sanity window is a
usable regression test for the whole traced path. Use it as one.

**Phase 3 and the EarthSciAST parameter work are both numerically INERT — and an
earlier version of this section said otherwise, wrongly.** Three 3-macro-step CONUS
sanity runs, same slice/window/config:

| | reseact code | EarthSciAST |
|---|---|---|
| **A** | pre-Phase-3 driver | `2abfadaf` |
| **B** | Phase 3, INTERMEDIATE commit | `2abfadaf` |
| **C** | Phase 3 **as merged** | `18fd10f0` (Phases 1 + 5) |

**A vs C: 28/28 values bit-identical, worst 0.000e+00, `nC=159/5` both.** That single
comparison carries a lot: it crosses a different EarthSciAST (with the
parameter-vector ABI and the parameter-class recording, which touches
`_compile(::VarExpr)` on every model's build path), a different runner, and a
different Julia environment — and lands on the same bits. Both repos' changes are
inert for reseact.

B differed from both by 1.4e-7 on `o3_min`, with `nC=162/6`. I diagnosed that at the
time as "Phase 3 perturbs the chemistry path at roundoff because loading Enzyme
changes the pass pipeline". **That was wrong**: the merged Phase 3 code also loads
Enzyme, and reproduces A exactly. The intermediate and merged integrators differ only
in comments, so B's difference is attributable to neither the EarthSciAST version nor
the reseact code, and its actual cause is **unestablished** (machine load and XLA
autotuning are plausible and unproven).

The process lesson is the durable part: **I diagnosed a one-run difference without
repeating the run.** A lone diff against a recorded baseline is a reason to re-run
first, not to explain. The explanation I reached was self-consistent, mechanistically
plausible, and false — and it would have stood unchallenged if an unrelated
cross-repo integration check had not happened to falsify it.

What survives: the pipeline reproduces bit-exactly across code and environment changes
in every controlled comparison here, so a diff is worth investigating — but one
uncontrolled run varied, so confirm before diagnosing. And the adaptive controller does
convert ulp-level differences into different accept/reject counts, so never read a
changed step count as changed solver quality without checking the state difference
behind it.
