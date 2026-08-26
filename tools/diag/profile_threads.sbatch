#!/bin/bash
# ===========================================================================
# profile_threads.sbatch -- does XLA:CPU use the cores it is given?
# ===========================================================================
# XLA sizes its CPU intra-op pool from tsl::port::MaxParallelism(), which reads
# the process CPU affinity mask, so `taskset` is the reliable knob (see
# tools/diag/run_server_1thread.sh). This runs the SAME per-step profile under a
# 1-core and a 16-core mask. If the per-step time barely moves, the step is not
# being parallelised at all and the cores are free speedup sitting unused.
#
#   sbatch --export=ALL,PT_CPUS=0-0 tools/diag/profile_threads.sbatch
# ===========================================================================
#SBATCH --job-name=pthreads
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
PT_CPUS=${PT_CPUS:-0-15}
export RESEACT_RXENV=$PWD/run-model-jl
export RESEACT_BACKEND=cpu
export RESEACT_NLON=${RESEACT_NLON:-13} RESEACT_NLAT=${RESEACT_NLAT:-7} RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_ADJ_JAC=sym RESEACT_ADJ_UJITTER=0
export RESEACT_PROF_REPS=${RESEACT_PROF_REPS:-15}
export RESEACT_PROF_VJP=${RESEACT_PROF_VJP:-0}
echo "=== node $(hostname) taskset=$PT_CPUS grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} ==="
echo "started $(date -Is)"
# NB the Julia thread count is left at 8 in both arms so the ONLY thing that
# moves is the mask XLA reads.
taskset -c "$PT_CPUS" julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/profile_adjoint.jl
echo "finished $(date -Is)"
