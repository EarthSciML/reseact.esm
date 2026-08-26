#!/bin/bash
# ===========================================================================
# mwe_xla_thread_confirm.sbatch -- TARGET C step 1: does the XLA:CPU
# multi-thread wrong-answer fault still happen AT ALL on Reactant 0.2.280?
# ===========================================================================
# Not an MWE -- this is the re-confirmation against the REAL model, which is
# the cheap and decisive question. It rebuilds ReSEACT at 6x6x8 (~580 s) with
# the blocker-4 workaround OFF, then soaks the two compiled step programs.
#
# The affinity mask is what sets XLA:CPU's intra-op pool size
# (tsl::port::MaxParallelism -> sched_getaffinity), so --cpus-per-task=4 IS
# the "taskset -c 8-11" configuration from README-nondet.md, at which the
# measured fault rate was ~1e-2 per call. 8000 calls should show ~80 hits.
#
#   sbatch tools/diag/mwe_xla_thread_confirm.sbatch
# ===========================================================================
#SBATCH --job-name=mweC4t
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=$PWD/run-model-jl
export RESEACT_NLON=6 RESEACT_NLAT=6 RESEACT_NLEV=8
export RESEACT_ADJ_STAGES=none
export RESEACT_ADJ_CLAMP=0
export RESEACT_ADJ_JAC=sym
export RESEACT_ADJ_XLAFIX=0          # <-- the workaround OFF; this is the point
export RESEACT_LABEL=mweC-4t-$SLURM_JOB_ID
export RESEACT_HAMMER_STAGES=${RESEACT_HAMMER_STAGES:-soak}
export RESEACT_HAMMER_PROGS=${RESEACT_HAMMER_PROGS:-ros,ssp}
export RESEACT_HAMMER_SOAK=${RESEACT_HAMMER_SOAK:-8000}
export RESEACT_HAMMER_VJP=0

echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK ==="
grep Cpus_allowed_list /proc/self/status
echo "started $(date -Is)"
export RESEACT_HAMMER_WALK=${RESEACT_HAMMER_WALK:-2000}
export RESEACT_HAMMER_K=${RESEACT_HAMMER_K:-200}
julia --project="$RESEACT_RXENV" -t "$SLURM_CPUS_PER_TASK" --heap-size-hint=36G tools/diag/nondet_hammer.jl
echo "finished $(date -Is) rc=$?"
