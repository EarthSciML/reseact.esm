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

### Phase 2 — first real sensitivities, forward mode (2–3 days, parallel)
Forward mode already crosses the while regions exactly, so this needs nothing from
Phase 1. Target a handful of parameters where cost ∝ n_params is affordable:
`NEIRegrid.scale` (the uniform emissions knob), `Transport3D.tau_pblmix`, one split
fraction. Produces a defensible ∂(CONUS O₃)/∂(emissions scale) early, and the reference
Phase 3 is checked against.

**Acceptance:** vs central finite differences on a 3-macro-step window, ~1e-6 relative.

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

**Acceptance on `reseact.esm`:** `scale`, `F_NO`…`F_FORM`, `tau_pblmix`, `Rd_air`,
`g_acc` land **numeric**; `lon0_deg`, `lat0_deg`, `dlon_deg`, `dlat_deg`, `src_x0/y0/dx/dy`,
`lcc_lat_1/2/0`, `lcc_lon_0`, `lcc_R`, `atol`, `lev` land **structural**.

## 4. Cross-cutting caveats

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
