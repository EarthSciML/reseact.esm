# The intermittent non-finite fault: how to reproduce it in five minutes

`DIFFERENTIABILITY_PLAN.md`, Phase 4, FINDING 2 has the diagnosis and the numbers.
This is the operating manual for the harness.

**Short version.** The compiled ReSEACT chemistry RHS is a pure StableHLO dataflow
graph, and XLA:CPU executes it wrongly on about 1 call in 2,000 when it has more than
one intra-op thread. The corruption is always the same: NaN in the six dry-deposition
species at one grid cell (lane 1). With ONE XLA thread it never happens.

## Reproduce

```bash
./tools/diag/run_server.sh            # 20 XLA threads: ~5e-4 per call
./tools/diag/run_server_ncpu.sh       # taskset -c 8-11, 4 threads: ~1e-2 per call, the
                                      #   fastest way to see it
./tools/diag/run_server_1thread.sh    # taskset -c 8,  1 thread: zero, always
```

Each builds ReSEACT once (~580 s at 6x6x8) and then watches its job directory
(`tools/diag/jobs`, `jobs4t`, `jobs1t` respectively): drop a `.jl` file in and it is
`include`d in that process with the driver's `CSSP`/`CROS`/`THT`/`THC`/`UBASE`/`var_map`
in scope. Write `STOP` to quit. Sources of the jobs that produced the plan's numbers are
in `tools/diag/jobs_src/`; the transcripts are in `tools/diag/logs/*.log.gz`.

`NANHUNT_CPU=8-9 ./tools/diag/run_server_ncpu.sh` picks a different CPU set. Only
CPUs in the process's affinity mask work (`grep Cpus_allowed_list /proc/self/status`).

## Which job answers what

| job | question |
|---|---|
| `j01_determinism` | is the compiled step a function of its inputs? (also double-reads every result buffer, which separates a bad computation from a bad host read) |
| `j03_capture` | full dump of every fault: cells, groups, bad vs re-issued values |
| `j04_localize` | is it the integrator or the RHS? (a probe copy of `ros23_step` that returns its intermediates) |
| `j05_rhs_detail` | which states, at which points, and dump the HLO |
| `j06b_xlaflags` | the rate under each XLA:CPU backend option |
| `j07_poison` / `j09_xlapoison` | does disturbing recently-freed memory change the rate? (Julia's heap; PJRT's own allocator) |
| `j08_confirm` | does the fix hold on the whole ROS23 step, and what does it cost numerically? |
| `j11_eest` | does a fault poison `EEst` even when the state is clean? (yes, and that is FINDING 1) |

## Workaround

Per-compile, no rebuild:

```julia
co = Reactant.CompileOptions(; sync = true,
                             xla_debug_options = (; xla_cpu_prefer_vector_width = 128))
C = Reactant.@compile compile_options=co ros_step(U_R, THC, T_R, DTC_R)
```

`xla_cpu_use_fusion_emitters=false` USED TO work equally well, and this file used to
recommend it as an equal alternative. It is not one any more: XLA removed the flag, and
on Reactant 0.2.280 `DebugOptions` has no such field, so passing it raises

    ArgumentError: Failed to assign fields (:xla_cpu_use_fusion_emitters,) to object
    with fields (...)

not at `CompileOptions` construction -- which validates nothing and stores the
NamedTuple as given -- but later, inside `Reactant.XLA.get_debug_options`, i.e. at
`@compile` time, after the trace. `tools/diag/xla_debug_field_check.jl` prints the
accepted `xla_cpu_*` set for whatever Reactant the environment resolves; run it before
quoting any flag from this file. `xla_cpu_prefer_vector_width` is still a field.

Neither costs wall time at the per-call level, and the vector-width one changes the
answer by up to 1.6e-6 relative on one step, so treat it as a new numerical floor rather
than a free fix -- and see WHERE THIS STANDS NOW below before switching it on at all.
Pinning XLA:CPU to one thread also works and changes no numbers at all, at the cost of
the thread pool.

## WHERE THIS STANDS NOW (2026-08-24, Reactant 0.2.280 / Reactant_jll 0.0.405+0)

**The fault is still live at CONUS. Do not turn the workaround off.**

It got much rarer, and that is exactly what made it worth re-checking -- and what
makes the small grid a trap. At 6x6x8 it now looks gone: 178,400 real-model calls
and 60,000 synthetic calls with `RESEACT_ADJ_XLAFIX=0`, at both 4-CPU and 20-CPU
affinities, came back with zero non-finite and zero bit-differing results
(`mwe_xla_thread_confirm.sbatch`, `mwe_xla_thread_nondet.jl`). The documented
~1e-2 per-call rate would have predicted several hundred faults in that alone.

At 13x7x72 (6,552 cells, 85,176 states) it is NOT gone. `conus_race_soak.sbatch`
with the workaround off, at `--cpus-per-task=4` (slurm 10127446):

| stage | calls | non-finite | bit-differing |
|---|---|---|---|
| E1 determinism, both programs, both base points | 800 | 0 | 0 |
| `ssp_step` soak (transport) | 40,000 | 0 | 0 |
| `ros_step` soak (chemistry) | 40,000 | **1** | **1** |
| E4 walk along a real trajectory | 4,000 | 0 | 0 |

Call 7,909 of the chemistry soak returned 13 non-finite entries -- ALL THIRTEEN
state groups at ONE cell, (i,j,k) = (2,1,1) -- and four immediate re-issues of the
byte-identical call all came back finite and all differed from the bad result in
exactly those 13 entries. That is the signature this file has always described,
one cell corrupted, reproducible only in the statistical sense.

So the per-call rate is now ~2.5e-5 (1 in 40,000) rather than ~1e-2, a drop of
about 400x, but it is a rate, not a zero. And it scales the wrong way for us: the
program that faults is the CHEMISTRY one, the transport program did not fault in
40,000 calls, and it did not fault at all at 6x6x8 across 178,400 calls. Whatever
is left is reached by the bigger graph.

**What that costs a production run.** A 48 h CONUS adjoint issues roughly 28,000
accepted chemistry steps forward, replays every one of them in the backward sweep,
and issues a VJP for each -- order 1e5 chemistry calls. At 2.5e-5 that is a
handful of expected faults, and ONE is enough to abort: the fixed-sequence replay
check in `tools/adjoint_gradient.jl` (`tapes_for`) errors out when a replayed
macro step lands NaN away from its checkpoint, which is how the very first CONUS
backward sweep died in 2026-08. It happened again on 2026-08-24 (slurm 10127204,
`RESEACT_ADJ_XLAFIX=0`, 12 macro steps): "fixed-sequence replay of macro step 11
lands NaN relative away from the checkpointed end state".

**And the workaround still works.** The same soak with `RESEACT_ADJ_XLAFIX=1`
(slurm 10129402, also `--cpus-per-task=4`) ran the chemistry program clean:
40,000 `ros_step` calls, 0 non-finite, 0 bit-differing, 0.2553 s/call -- against
0.2153 s/call with it off, an 18.6% per-call cost (different nodes, ccc0233 vs
ccc0236, so treat the size as indicative; the within-allocation comparison below
says 6.8% on the forward pass).

### What turning it off would buy, measured at CONUS

Very little, and less than the driver's comment claims. Numbers below are the two
legs of slurm 10127204 -- ONE allocation, 16 cpus, both legs the same process
shape minutes apart, NMACRO=12, jac=sym, clamp=0, ujitter=0.1.

READ THEM WITH THIS CAVEAT: EarthSciAST changed underneath that job between the
legs (src hash 8a342184 -> a874ed8f, and again to 3d94e127 during leg 0), and the
traced graph changed with it -- leg 0 traced 42,253 nodes against leg 1's 41,309,
with `floor` and `minimum` ops leg 1 never saw. Leg 0 is therefore tracing a
slightly LARGER program, which biases against the workaround-off leg on both time
and memory; it came out cheaper anyway, so the direction of each conclusion is
safe even if the last digit is not. `xlafix_cost_conus.sbatch` now runs both legs
out of a private rsync snapshot in `~/.cache` precisely so this cannot recur.

* **Compile wall time: a wash.** Four compiles, 1,051.3 s with the workaround
  (129.1 + 145.0 + 608.8 + 168.4) against 1,077.8 s without (136.7 + 152.7 +
  617.9 + 170.5). The workaround is not why CONUS compiles are slow; `ssp_vjp`
  alone is ~610 s either way.
* **Peak memory: ~1.13x, NOT the ~2x on record.** 17.83 GB with it on against
  15.82 GB with it off (`/usr/bin/time -v`, per leg, so per process). Job-level
  cgroup `memory.peak` over both legs was 19.34 GB against an 80 GiB
  `memory.max`. The "roughly doubles compile memory" line in
  `tools/diag/adjoint_conus.sbatch` and in the driver comment predates the
  symbolic Jacobian and does not survive re-measurement. sacct MaxRSS is only a
  job-level ceiling; the limit that actually kills a job is `memory.max` in
  `/sys/fs/cgroup/system.slice/slurmstepd.scope/<token>/`, NOT the node-wide
  `/sys/fs/cgroup/memory.stat`.
* **Run time: the workaround is SLOWER.** Same allocation, same 359-step
  accept/reject ladder: forward pass 76.63 s with it on against 71.78 s with it
  off, 6.8% slower. That matches the 6.4% already recorded for the traced runner
  (slurm 10017939) and is the opposite of "costs nothing in wall time". The
  chemistry soaks agree in direction (0.2153 s/call off, ~0.25 s/call on) though
  those two ran on different nodes.

So the workaround costs single-digit percent of run time and ~2 GB, and buys
freedom from a fault that aborts the whole backward sweep. That is not a close
call.

The J shift is the one figure that stands roughly as recorded: (J_on - J_off) /
J_off = +8.301e-05 at 12 macro steps, against the +4.0e-5 recorded at 3 macro
steps -- same sign, same order. What HAS changed is the mechanism. In 2026-08 the
gap was attributed to the racy leg integrating a different trajectory through
spurious NaN-driven rejections; here the two legs produced BYTE-IDENTICAL
accept/reject ladders and identical accepted-step counts (359 = 42 transport +
317 chemistry), so at 12 macro steps the residual gap is the vector-width change
itself, not a trajectory split. Treat 1e-4 relative as the floor a
workaround-toggle imposes on J.
