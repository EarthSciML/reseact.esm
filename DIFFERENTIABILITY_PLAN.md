# Differentiable simulation: measured capabilities and the plan

The point of the Reactant/XLA work is not speed — measured, it is *slower* than the
native runner on CPU for this model. The point is **∂(simulation result)/∂(parameter
vector)**: the thing DifferentialEquations.jl gives you by letting `p` be a solve-time
argument.

This document records what was **measured** (not assumed), the one architectural
constraint that follows, and the phased plan. Every row of the capability matrix is a
run against a small reaction–diffusion model built through the same `:oop` emitter the
real model uses; the probes are `diffprobe{,2,3,4,5}.jl` and `whilegrad{,2}.jl`.

Companion reading: HELPERS.md §4 (the same findings in context), and §2 for where each
helper should eventually live.

---

## 1. The capability matrix, measured

| | forward mode | reverse mode |
|---|---|---|
| the `:oop` RHS (straight-line traced) | ✅ exact | ✅ exact vs host ForwardDiff |
| coloured block Jacobian, one traced program | ✅ exact (worst error 0.0) | ✅ VJP exact |
| **`Reactant.@trace while` region** | ✅ exact | ❌ *MLIR pass pipeline "all" failed* |
| host-unrolled fixed step count (no while region) | ✅ | ✅ exact |

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

**Reverse mode cannot cross a `stablehlo.while`, and a macro step *is* two of them**
(the SSPRK43 and ROS23 adaptive loops in `rx_traced_integrator.jl`). Since even a
*fixed*-trip while loop fails, the blocker is the while region itself, not adaptivity.

This kills the obvious design — "call `Enzyme.gradient` on the compiled macro step" —
and it matters because reverse is the only affordable mode at this parameter count:

| route | cost for a 24 h CONUS run |
|---|---|
| forward sensitivity, 56 parameters | ~88 h (cost ∝ n_params) |
| forward sensitivity, ≤ 5 parameters | ~8 h — **usable today** |
| reverse adjoint | ~4–5 h, independent of n_params |

So the plan buys the cheap forward result immediately, and separately removes the while
regions from the gradient path to unlock reverse.

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

Also measured, and it changes an expectation: at 6×6×8 over one 300 s macro step
the exact Jacobian is compile-neutral (89.3 s vs 87.3 s) and **step-count
neutral** (120 accepts / 4 rejects vs 116 / 3; a second run gave 116 / 3 for
both, so the counts are not even stable to that resolution — the controller
amplifies ulp-level nondeterminism). The native arm's `block_ad_jac` cut
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

**Acceptance on `reseact.esm`** (the runtime `p` is 49 scalars, measured):
* **numeric**: `NEIRegrid.scale`, `NEIRegrid.g0`, `Transport3D.tau_pblmix`, `Rd_air`,
  `g_acc` — the first three confirmed by an actual measured sensitivity in Phase 2.
* **structural**: `lon0_deg`, `lat0_deg`, `dlon_deg`, `dlat_deg`, `src_x0/y0/dx/dy`,
  `lcc_lat_1/2/0`, `lcc_lon_0`, `lcc_R`, `atol`, `lev`.
* **const-folded data**: `NEIRegrid.F_NO`, `F_NO2`, `F_CO`, `F_ISOP`, `F_FORM` — the
  whole NEI2016 12US1 source-grid emission fields, collapsed by the conservative regrid
  at build. Making these differentiable is separate, larger work: the regrid would have
  to become a runtime operation. Until then, per-species emissions sensitivity is only
  reachable through the scalars multiplying the folded field (`scale`, `g0`).

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

**Phase 3 perturbs the production chemistry path at roundoff, and the adaptive
controller amplifies it.** Merging Phase 3 changed nothing semantically on the
default path (`jac=:fd`, `unrolled=true` dispatches to exactly the previous
`fd_block_jac_unrolled` call), but the 3-macro-step CONUS sanity is no longer
bit-identical to pre-Phase-3 runs:

| | t=1.50 | t=1.58 | t=1.67 | t=1.75 |
|---|---:|---:|---:|---:|
| `o3_min` rel | 0 | 4.6e−10 | 1.4e−7 | 1.1e−7 |
| `o3_mean` rel | 0 | 1.3e−13 | 2.1e−11 | 2.1e−11 |
| `m_min` rel | 0 | 0 | 0 | 3.4e−14 |

and the chemistry step counts moved from `nC=159/5` to `162/6` (transport `nT=8/0`
unchanged). The shape is diagnostic: it enters at roundoff after the first chemistry
step, and AIR MASS — advanced by the transport step, which Phase 3 does not touch —
stays bit-identical for three macro steps. So this is roundoff in the recompiled
chemistry step (loading Enzyme into the session changes the pass pipeline), amplified
by a controller that is chaotic at that level, and not a semantic change. It is four
orders below the traced-vs-native agreement (1.6e−3) and inside the 5e−2 validation
tolerance.

Worth stating plainly because both facts cut the same way: the pipeline is
reproducible enough that small diffs are meaningful, AND the adaptive controller
converts ulp-level differences into different step counts. Do not read a changed
accept/reject count as a change in solver quality without checking the magnitude of
the underlying state difference first.
