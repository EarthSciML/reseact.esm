#!/bin/bash
# ===========================================================================
# traced_24h_racefix.sbatch -- re-measure the traced runner's 24 h CONUS solve
# with the XLA:CPU race workaround ON.
# ===========================================================================
# WHY. Every week-scale projection in DIFFERENTIABILITY_PLAN.md and in the
# adjoint work rests on one number: the traced runner's 5,923 s for 288 macro
# steps of 24 h CONUS (HELPERS.md line 18), which extrapolates to an 11.5 h week
# forward and, at the measured 2.87x backward/forward ratio, a 33 h week adjoint.
#
# That number was measured WITH the XLA:CPU race active. On the adjoint driver at
# the same grid the race cost 3.14x of forward wall clock -- not through slower
# arithmetic (per-attempt cost was unchanged, 0.436 -> 0.449 s) but through
# SPURIOUS WORK: a faulting call NaNs `EEst`, the PI controller reads that as a
# rejection, and each spurious rejection also shrinks dt, so the solve grinds out
# many more, smaller accepted steps. Rejections 24.5% -> 2.9%, accepted 428 -> 170.
#
# If any meaningful part of that carries over to a real (un-jittered) trajectory,
# the 11.5 h week forward is wrong and every cost number downstream of it moves.
# This job settles it.
#
# WHY A FULL 24 h AND NOT A SHORT WINDOW. The runner's own header is explicit:
# step cost is strongly diurnal, ~9.6 s/step pre-dawn against ~35 s/step at
# midday, because photochemistry is what makes the chemistry half stiff. A short
# window sits in the dark, carries the first-step warm-up, and "gets the right
# answer for the wrong reasons". 24 h is also exactly what the baseline measured,
# which is the only reason this run is comparable to it.
#
# WHAT TO COMPARE. Baseline, race active: build 718 s, @compile 557 s, solve
# 5,923 s / 288 macro steps. Read the same three numbers out of this log. The
# solve is the one that matters; expect the compile to be SLOWER, since the
# workaround roughly doubled compile memory on the adjoint driver (Julia RSS
# 5.5 -> 13.3 GB on the same trace) and it is paid once either way.
#
# Also worth reading: whether the 24 h output still agrees with the native arm to
# the documented 6.3e-5 on O3_mean. If the race was biasing the trajectory (it
# shifted the adjoint objective 4.0e-5 relative), that agreement should IMPROVE.
#
# SIZING. Sized from measurement, not from fear: the adjoint job at this grid --
# which compiled FOUR programs including two reverse-mode VJPs -- peaked at
# MaxRSS 14.9 GB. This compiles one. 64G is already generous. An earlier
# 40-cpu/180G request for the adjoint sat PENDING an hour on (Resources) while
# nodes had 39/40 CPUs idle but memory reserved; right-sizing started it at once.
#
# -t 8 matches every other measurement in this repo, including the 5,923 s
# baseline, so the timings stay comparable. The race is thread-count dependent
# (~1% of calls at 4 threads, 0 of 400,000 at one thread), so changing the thread
# count would confound exactly the effect under test.
#
# SUBMIT WITH:  mkdir -p logs && sbatch tools/diag/traced_24h_racefix.sbatch
# ===========================================================================
#SBATCH --job-name=reseact-rx24
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# NB: slurm will NOT create this directory -- submit with `mkdir -p logs` first
# or the job is rejected at submission. logs/ is gitignored.
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

JULIA="$(command -v julia)"
echo "host=$(hostname)  julia=$JULIA  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "started $(date -Is)"

export RESEACT_SOLVE_SECS=86400      # 24 h -- the baseline's window, exactly
export RESEACT_RXFIX=1               # the workaround under test
export RESEACT_LABEL=reseact-rx24-racefix
export RESEACT_CSV=logs/rx24-racefix.csv

"$JULIA" --project=run-model-jl -t 8 --heap-size-hint=48G run_reseact_reactant.jl

echo "finished $(date -Is)"
