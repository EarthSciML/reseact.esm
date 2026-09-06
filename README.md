# reseact.esm — ReSEACT: a differentiable regional chemical-transport model

ReSEACT is a 3-D chemical-transport model written as an **esm document** (`reseact.esm`,
esm 1.0.0): GEOS-FP meteorology, monotone flux-form PPM transport, the SuperFast
gas-phase mechanism with Fast-JX photolysis, PBL mixing, dry and wet deposition, and
NEI2016 emissions, on a native slice of the GEOS-FP 4°×5° grid (CONUS by default:
13×7×72 cells, 85,176 states). The model is compiled by EarthSciAST either to native
Julia or, through Reactant/XLA, to a traced program that Enzyme differentiates.

The headline result (2026-09-05): the gradient of mean surface O₃ over 48 h of CONUS
with respect to all 160 runtime scalars of the model, in 47 min of loop time on one
40-core CPU node, exact to 1.3e-11 against the single-process reference. Five days is
the runner's default and projects to about 2 h of loop.

## Layout

| Path | What it is |
|---|---|
| `reseact.esm` | The model. Imports rules and components BY REFERENCE from sibling repos (see below). |
| `run_reseact.jl` | Native runner: SciML operator splitting on the CPU. The reference. |
| `run_reseact_reactant.jl` | Traced runner: the same split through Reactant/XLA. |
| `run_reseact_adjoint.jl` | **The gradient runner.** Defaults to the 5-day CONUS adjoint. Its header is the authoritative record of what has been run, what it cost, and what is still unproven. |
| `tools/adjoint_gradient.jl` | The adjoint driver the runner wraps: host-lifted adaptive loops, per-macro-step checkpoints, replay, Enzyme VJPs, chemistry sharding. |
| `tools/shard_chem.jl`, `tools/shard_worker.jl`, `tools/capacity_chem.jl` | Process-level chemistry sharding (`RESEACT_ADJ_SHARDS`). |
| `tools/reactant_handoff/` | Traced integrator (`rx_traced_integrator.jl`), operator-split loop, Reactant patches. |
| `prototypes/reseact_3d_chem/` | Split/build machinery shared by the runners (`split_common.jl`), block Jacobian, hybrid-coordinate coefficients. |
| `prototypes/reseact_3d_chem/geosfp_grids.jl` | The GEOS-FP **resolution table** — URL tokens, cell spacings, native extents, per-resolution CONUS box. Dependency-free, so the launcher and the drivers read the same rows. |
| `tools/diag/` | ~140 measurement probes and their sbatch files. Each probe's header records its measured result; they are the provenance for every number in the docs. |
| `run-model-jl/` | The Julia environment (Project + Manifest). |
| `DIFFERENTIABILITY_PLAN.md` | The adjoint plan, blockers, and §6: the wall-time campaign toward the 30-minute target, with every measured lever and negative result. |
| `HELPERS.md` | What each runner depends on and where the helpers should eventually live upstream. |
| `INDEX_RANGE_READS.md` | Plan for reading only the model's window out of the GEOS-FP files — what the pushdown seam already provides, the one missing reader capability, and what each step buys. |
| `tools/diag/README-nondet.md` | The XLA:CPU nondeterminism (race) and its workaround, on by default. |

## Prerequisites

* **Julia 1.12** (developed on 1.12.6). Linux x86-64; everything runs on CPU
  (XLA:CPU). The GPU backend is deliberately not used.
* **Sibling repositories**, checked out next to this one so that `../EarthSciDiscretizations/...`
  and `../EarthSciModels/...` resolve from `reseact.esm`, and so the Julia environment can
  `develop` the packages. State used for the 2026-09-05 results (all on GitHub under
  `EarthSciML/`):

  | Repo | State the 2026-09-05 results ran on | Role |
  |---|---|---|
  | `EarthSciAST` (`pkg/EarthSciAST.jl`) | `17cac5d8d` = `main` at `a1dc9bb30` + 2 commits of branch `inline-test-array-observeds` (an inline-test bindings fix; no emitter change). `origin/main` `05c9064a6` adds only the off-by-default `ESS_OOP_SHIFT_SLICE`. Use `main` once that branch merges. | compiler: `:oop` emitter, SSA emitter (`ESS_OOP_SSA=1`), Reactant extension |
  | `EarthSciASTDiff` (`pkg/EarthSciASTDiff.jl`) | `main` `7f83c11` — REQUIRED: the driver calls `prepare_jacobian(; source_layout)`, which this commit adds | symbolic block Jacobian (`jac=:sym`) |
  | `EarthSciASTSplitter.jl` | `main` `35902c6` | transport / chemistry split |
  | `EarthSciIO` (`julia/`) | `main` `d109951` (v0.1.3) | GEOS-FP / NEI providers, on-disk cache |
  | `EarthSciDiscretizations` | `main` (`grids/latlon3d/`, `regridding/`, `reprojection/` identical on `main` and the `a9a7c3d` branch the runs used) | PPM stencil rules referenced by the model |
  | `EarthSciModels` | `main` ≥ `8a25818` (runs used `bf9a7e3`, which contains it) | GEOS-FP, NEI2016, SuperFast, Fast-JX, deposition components |

  The runner's provenance block prints exactly these for any run, so compare against
  it rather than this table when in doubt. `tools/diag/provenance/` keeps the patch
  that was uncommitted in EarthSciASTDiff at the time of the 48 h runs (now `7f83c11`, compat widened in `0bd517f`).

  The Manifest pins Reactant v0.2.280 and Enzyme v0.13.199. It still RECORDS
  EarthSciAST as v0.1.1 while the checkout's Project.toml says 0.2.0; path-developed
  packages load from the path regardless, and the `Pkg.develop` step below re-resolves
  the record. (EarthSciASTDiff's compat was widened to `0.1, 0.2` for the same reason.) Other Reactant versions
  have not been validated; 0.2.274–0.2.284 have all shown the checkpointing bug noted
  in the plan, and the race workaround is required on every one of them.

## Setting up on another machine

1. Clone this repo and the six siblings into one directory.
2. Re-point the environment's development paths. `run-model-jl/Manifest.toml` records
   them as absolute paths (`path = "/projects/.../EarthSciAST/pkg/EarthSciAST.jl"` etc.).
   From the repo root:
   ```bash
   julia --project=run-model-jl -e '
     using Pkg
     Pkg.develop(path="../EarthSciAST/pkg/EarthSciAST.jl")
     Pkg.develop(path="../EarthSciASTDiff/pkg/EarthSciASTDiff.jl")
     Pkg.develop(path="../EarthSciASTSplitter.jl")
     Pkg.develop(path="../EarthSciIO/julia")
     Pkg.instantiate(); Pkg.precompile()'
   ```
3. Data. EarthSciIO caches GEOS-FP and NEI slices under `$EARTHSCIDATADIR`
   (default `/scratch.local/$USER/earthsci-cache`; never a home directory with an inode
   quota). Set it, and make the NEI2016 mirror resolvable: the component
   `EarthSciModels/components/earthsci_data/nei2016_monthly.esm` points at
   `../../data/...`, so create `EarthSciModels/data -> $EARTHSCIDATADIR` (gitignored).
   `EARTHSCI_OFFLINE=1` forbids downloads. The 48 h CONUS gradient needs three days of
   GEOS-FP files (the forcing window plus the day its final bracket lands in) and the
   NEI2016 monthly files for the month.
4. `export RESEACT_RXENV=$PWD/run-model-jl` (the probes and sbatch files read it).

## Running

Everything of production size is a **batch job**: it wants a whole 40-core node with
160 GB. Never run a CONUS build in a shared interactive cgroup (page-fault thrash for
hours; see the plan's operational note).

```bash
mkdir -p logs
sbatch tools/diag/adjoint_conus_5d.sbatch          # 5-day CONUS gradient (the default)
sbatch --export=ALL,RESEACT_ADJ_NMACRO=576 tools/diag/adjoint_conus_5d.sbatch   # 48 h
```

### Resolution

`RESEACT_RES` selects the GEOS-FP grid; the row carries the URL, the cell
spacings, the native extent and the **CONUS index box**, so the same geography
(lon −125..−65, lat 26..50) is simulated at whichever resolution is asked for:

| `RESEACT_RES` | native grid | CONUS columns | A3dyn per day |
|---|---|---|---|
| `4x5` (default) | 72 × 46 × 72 | 13 × 7 = 91 | 28 MB |
| `2x2.5` | 144 × 91 × 72 | 25 × 13 = 325 | 111 MB |
| `0.25x0.3125` | 1152 × 721 × 72 | 193 × 97 = 18,721 | 3.8 GB |
| `0.25x0.3125_CH` | 225 × 161 × 72 | (China domain) | — |

The vertical is 72 hybrid levels in all of them, so `NLEV` and the `dA`/`dB`/
`Ap`/`Bp` tables never change with the horizontal.

**Measured scaling** (2026-09-06, one whole node per arm, 8 chemistry shards):

| arm | forward | backward | loop | all in |
|---|---|---|---|---|
| 4x5, 48 h | 1.160 s/window | 3.657 s/window | 46 min | 1 h 11 m |
| 4x5, 5 day | 1.147 s/window | 3.647 s/window | 1 h 55 m | 2 h 21 m |
| 2x2.5, 48 h | 4.088 s/window | 12.419 s/window | 2 h 38 m | 3 h 21 m |
| 2x2.5, 5 day | 4.185 s/window | 12.509 s/window | 6 h 41 m | 7 h 25 m |

Cost is **linear in cells, slightly sublinear**: 2x2.5 has 3.57× the columns and
costs 3.43× the 48 h loop and 3.48× the five-day loop. The window is linear too
— each five-day arm is 2.5× its own 48 h, with no degradation over 120 h. Halving the cell width did *not* force more substeps — the
adaptive controller took 28,384 inner steps against 27,973, because at 300 s
macro steps the substep count is set by accuracy, not CFL. Setup scales
sublinearly too (build 226 → 330 s). Full breakdown, including which per-window
line grew by how much, in `tools/diag/adjoint_res_scaling.sbatch`'s header. The nested North-America,
Europe and Asia domains are **not** offered: both mirrors carry only their
`soil` files, no meteorology, at every year sampled — high resolution over CONUS
means slicing the global 0.25° files, which the whole-file reader cannot yet do
economically (see `INDEX_RANGE_READS.md`).

```bash
sbatch --job-name=adjres-2x25-48h --time=12:00:00 \
       --export=ALL,RESEACT_RES=2x2.5,RESEACT_ADJ_NMACRO=576 \
       tools/diag/adjoint_res_scaling.sbatch
```

Small demonstration (minutes, any machine, all four validation stages):

```bash
RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8 RESEACT_ADJ_NMACRO=3 \
RESEACT_ADJ_CLAMP=0 RESEACT_ADJ_UJITTER=1e-1 RESEACT_ADJ_STAGES=fwd,adj,ref,fdtape \
julia --project=run-model-jl run_reseact_adjoint.jl
```

Every default in the runner is a `get!`, so the environment wins. The knobs that
matter: `RESEACT_RES` (GEOS-FP resolution), `RESEACT_NLON/NLAT/NLEV` and
`RESEACT_LON0/LAT0` (grid box, defaulting to the resolution's CONUS footprint),
`RESEACT_ADJ_NMACRO` (300 s macro steps),
`RESEACT_ADJ_SHARDS` (chemistry worker processes, default 8, 0 = single process),
`RESEACT_ADJ_JAC` (`sym`), `RESEACT_ADJ_STAGES` (`fwd,adj[,ref,fdtape]`),
`RESEACT_ADJ_OBJ` (objective, default `SuperFast.O3:surf`), `RESEACT_LABEL`,
`RESEACT_ADJ_CSV` (gradient output). The runner prints a **provenance block** at
start — the git HEAD and dirty state of this repo and of every developed sibling — so
a logged result can be tied to commits.

The forward-only runners: `julia --project=run-model-jl -t 8 run_reseact.jl` (native) and
`run_reseact_reactant.jl` (traced); see `HELPERS.md` for their cost and agreement.

## What to expect (CONUS 13×7×72, 40-core node, 2026-09-05)

| | |
|---|---|
| setup (build ~4 min, Jacobian prep ~1 min, four compiles ~10 min, 8 shards ~10 min) | ~20 min |
| 48 h loop: forward 1.16 s/window, backward 3.66 s/window | 46 min (1 h 11 m all in) |
| **5-day loop: forward 1.15 s/window, backward 3.65 s/window** | **1 h 55 m (2 h 21 m all in)** |
| memory | ~3.5 GB per worker, driver peak ~45 GB |
| gradient checks | replay lands 0.0 at every one of 1,440 checkpoints, 0 retries over 70,401 VJP calls, J to 13 digits vs the 48 h runs |

The five-day window **has now been run end to end** (slurm 10386110, 2026-09-06):
1,440 macro steps, 70,401 accepted inner steps, J = 27.0756235696808 ppb. Days
3–5 are no longer unpaid-for. The same run at 2x2.5 is in
`tools/diag/adjoint_res_scaling.sbatch`, whose header carries the resolution
scaling.

The per-window budget, the remaining levers toward the 30-minute target, and the
measured negatives (including why the transport VJP costs 22x its step and what would
fix it) are in `DIFFERENTIABILITY_PLAN.md` §6.

## Rules learned the hard way

* No caches that grow with grid × run length (forcing-epoch cache, kept inner tapes):
  the model has to run far larger than CONUS.
* CPU only. The Quadro nodes' FP64 rate rules the GPU backend out.
* Measure in place; two "in-driver is 10x slower" inferences were wrong.
* The XLA:CPU race workaround (`sync=true`, vector width 128) stays on: it costs ~6%
  and buys 525x on O₃ agreement.
* `/tmp` is RAM and counts against the cgroup; the scratchpad is node-local and
  invisible to batch jobs. Write results under `logs/` (gitignored).
