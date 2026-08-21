#!/bin/bash
# ===========================================================================
# traced_48h_gate.sbatch -- can the model integrate 48 h of CONUS at all?
# ===========================================================================
# WHY THIS IS A SEPARATE, CHEAPER JOB. The 48 h ADJOINT costs its forward pass
# plus ~2x that again for the backward sweep. If the forward trajectory does not
# survive 48 h, that is ~16 h of node time spent to discover a fact this job
# establishes in ~5. It is a gate, not a measurement, and it runs concurrently
# with the correctness work rather than after it.
#
# WHY IT IS AN OPEN QUESTION. The longest CONUS simulation on record is 24 h
# (slurm 10017939, solve 6,301.7 s over 288 macro steps, race workaround on).
# Longer attempts have died twice, both times in the chemistry half:
#   * at 36.2 h -- surface O3 dry-deposits away (correct physics, there being no
#     emissions at the time), and the FD chemistry Jacobian then fails on species
#     sitting at 1e-30.
#   * at 11.33 h -- with NEI2016 emissions added. The exact AD Jacobian was
#     tested as the cause and REFUTED. The root cause found later was that the
#     model had NO vertical mixing at all: PBLH was consumed by nothing, so the
#     surface layer was a sealed box and went stiff before dawn. GEOS-Chem full
#     PBL mixing was added in response.
# 24 h has completed since that fix. 25-48 h has never been attempted, so the
# 36.2 h death in particular is UNRETESTED -- and note that its stated mechanism
# (an FD Jacobian differencing a species at 1e-30) is exactly what jac=:sym
# removes, though NOT in this runner: run_reseact_reactant.jl's Rosenbrock23
# lives inside a traced `@trace while` and still uses the FD block Jacobian.
# So a death here around 36 h would be informative rather than surprising, and
# would point straight at porting :sym into `adaptive_solve`.
#
# WHAT TO READ. Whether it reaches 172,800 s at all; the solve seconds against
# 6,301.7 s for half the window (a >2x ratio means the second day is stiffer,
# which is the diurnal cost profile, not a fault); and the CSV for anything
# going negative or flat-lining.
#
# SIZING. The 24 h run peaked well under 64G and this changes only the window
# length -- the state is the same 6,552 cells either way, and the tape is not
# kept. 10 h of walltime is 2x the projection, which is the right margin for a
# run whose whole purpose is to find out where it stops.
#
#   mkdir -p logs && sbatch tools/diag/traced_48h_gate.sbatch
# ===========================================================================
#SBATCH --job-name=reseact-rx48
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

JULIA="$(command -v julia)"
echo "host=$(hostname)  julia=$JULIA  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
echo "started $(date -Is)"

# Same env and same model as the adjoint runs, so this actually gates THEM.
# run-model-jl (what the 24 h baseline used, pre-migration) can no longer resolve
# the corpus: the sibling corpora went esm 1.0.0 on 2026-08-20.
export RESEACT_RXENV=${RESEACT_RXENV:-/u/ctessum/reseact-esast-pin/env-sym}
export RESEACT_MODEL=${RESEACT_MODEL:-/u/ctessum/reseact-esast-pin/corpus-pin/reseact.esm/reseact.esm}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
[ -f "$RESEACT_MODEL" ] || { echo "FATAL: model $RESEACT_MODEL missing"; exit 1; }

export RESEACT_SOLVE_SECS=${RESEACT_SOLVE_SECS:-172800}   # 48 h
export RESEACT_RXFIX=${RESEACT_RXFIX:-1}
export RESEACT_LABEL=reseact-rx48-gate
export RESEACT_CSV=logs/rx48-gate-${SLURM_JOB_ID}.csv

# -t 8 matches the 24 h baseline exactly, so the two solve times are comparable.
"$JULIA" --project="$RESEACT_RXENV" -t 8 --heap-size-hint=48G run_reseact_reactant.jl

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
