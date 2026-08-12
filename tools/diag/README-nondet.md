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

`xla_cpu_use_fusion_emitters=false` works equally well. Both cost nothing in wall time
and change the answer by up to 1.6e-6 relative on one step, so treat them as a new
numerical floor rather than a free fix. Pinning XLA:CPU to one thread also works and
changes no numbers at all, at the cost of the thread pool.
