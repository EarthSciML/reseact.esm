#!/bin/bash
#SBATCH --job-name=directcensus
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ctessum/oopretire-logs/%x-%j.out
# tools/diag/direct_rhs_census.jl at CONUS 13x7x72: emission, agreement and
# cost for the direct StableHLO emitter against the traced `:oop` lane, for
# both halves of the operator split and the analytic band model.
#
# The GRADIENT stage is deliberately absent. Reverse mode over these programs
# at CONUS is a separate measurement with its own memory profile, and the
# question this job answers -- does the direct lane still emit, agree and pay
# off at production size -- does not need it.
#
# TWO PASSES, because ESS_OOP_SSA is read at BUILD time: one process per
# setting, each measuring the direct lane against the traced lane of its own
# build. A single process cannot hold both.
#
# NOTE ON --export: `sbatch --export=ALL,K=V` splits on COMMAS; export in the
# submitting shell and use the default --export=ALL.
set -euo pipefail
export JULIA_DEPOT_PATH="/scratch/ctessum/oopretire-depot:/projects/illinois/eng/cee/ctessum/ctessum/.julia"
export RESEACT_RXENV="/projects/illinois/eng/cee/ctessum/ctessum/code/reseact-directrhs-env"
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
export RESEACT_NLON=${RESEACT_NLON:-13} RESEACT_NLAT=${RESEACT_NLAT:-7} RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_T0=${RESEACT_T0:-5400}
export RESEACT_ADJ_UJITTER=${RESEACT_ADJ_UJITTER:-1e-1}
export RESEACT_CENSUS_STAGES=emit,agree,cost
export RESEACT_CENSUS_NCALL=${RESEACT_CENSUS_NCALL:-20}
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact-directrhs

NT=${SLURM_CPUS_PER_TASK:-20}
HH=$(( ${SLURM_MEM_PER_NODE:-122880} / 1024 * 6 / 10 ))G
OUT=/scratch/ctessum/oopretire-logs
echo "=== node $(hostname) grid=${RESEACT_NLON}x${RESEACT_NLAT}x${RESEACT_NLEV} cpus=$NT heap=$HH ==="
echo "started $(date -Is)"
rc=0
for SSA in 1 0; do
    echo "---------------- ESS_OOP_SSA=$SSA ----------------"
    # tee, never a bare grep: a filtered pipe eats stack traces.
    ESS_OOP_SSA=$SSA \
    RESEACT_CENSUS_JSON="$OUT/conus-census-ssa$SSA-${SLURM_JOB_ID}.json" \
    julia --project="$RESEACT_RXENV" -t "$NT" --heap-size-hint="$HH" \
        tools/diag/direct_rhs_census.jl 2>&1 \
        | tee "$OUT/conus-census-ssa$SSA-${SLURM_JOB_ID}.log" || rc=$?
done
echo "finished $(date -Is) rc=$rc"
exit "$rc"
