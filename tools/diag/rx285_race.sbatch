#!/bin/bash
# ===========================================================================
# rx285_race.sbatch -- is the XLA:CPU intra-op wrong-answer race still there on
#                      Reactant 0.2.285, and is xla_cpu_prefer_vector_width=128
#                      still the fix?
# ===========================================================================
# The affinity mask is what sizes XLA:CPU's intra-op pool
# (tsl::port::MaxParallelism -> sched_getaffinity), so --cpus-per-task=4 IS the
# "taskset -c 8-11" configuration of README-nondet.md, where the measured fault
# rate was ~1e-2 per call -- 8000 calls showed ~80 hits with the workaround off
# and 0 of 20,000 with it on.  Four arms: both versions x workaround off/on.
#
#   sbatch tools/diag/rx285_race.sbatch
#SBATCH --job-name=rx285race
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
CODE=/projects/illinois/eng/cee/ctessum/ctessum/code
cd $CODE/reseact.esm
export RESEACT_MODEL=$CODE/reseact.esm/reseact.esm
export EARTHSCIMLBASE_SRC=$CODE/EarthSciMLBase/src
export RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8
export RESEACT_ADJ_STAGES=none RESEACT_ADJ_CLAMP=0 RESEACT_ADJ_JAC=sym
export RESEACT_HAMMER_STAGES=${RESEACT_HAMMER_STAGES:-soak}
export RESEACT_HAMMER_PROGS=${RESEACT_HAMMER_PROGS:-ros,ssp}
export RESEACT_HAMMER_SOAK=${RESEACT_HAMMER_SOAK:-8000}
export RESEACT_HAMMER_VJP=0
export RESEACT_HAMMER_WALK=${RESEACT_HAMMER_WALK:-2000}
export RESEACT_HAMMER_K=${RESEACT_HAMMER_K:-200}
echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK ==="
grep Cpus_allowed_list /proc/self/status
echo "started $(date -Is)"
for ARM in rx280 rx285; do
  case $ARM in
    rx280) E=$PWD/run-model-jl ;;
    rx285) E=$PWD/envs/rx285 ;;
  esac
  for FIX in 0 1; do
    echo "=== ARM $ARM xlafix=$FIX  $(date -Is)"
    RESEACT_RXENV="$E" RESEACT_ADJ_XLAFIX=$FIX \
      RESEACT_LABEL=rx285race-$ARM-fix$FIX-$SLURM_JOB_ID \
      julia --project="$E" -t "$SLURM_CPUS_PER_TASK" --heap-size-hint=36G \
      tools/diag/nondet_hammer.jl 2>&1 | tail -40
  done
done
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
