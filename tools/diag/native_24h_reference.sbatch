#!/bin/bash
# ===========================================================================
# native_24h_reference.sbatch -- the native 24 h CONUS reference CSV, so the
# traced arm's agreement can be re-checked with the XLA:CPU race fixed.
# ===========================================================================
# WHY. run_reseact_reactant.jl's header records that the traced arm agrees with
# the native arm over 24 h (289 aligned records) to max relative difference
# 6.3e-5 on O3_mean, 1.6e-3 on O3_min, 1.1e-3 on OH_max. That was measured with
# the XLA:CPU race ACTIVE.
#
# The race is now known to shift the answer by integrating a DIFFERENT trajectory
# -- it moved the adjoint objective 4.0e-5 relative -- and that biasing, not
# speed, is the entire reason RESEACT_RXFIX defaults to on (it costs ~6% of wall
# clock; slurm 10017939). That justification is so far untested on this runner.
#
# It makes a falsifiable prediction: with the race fixed, agreement with the
# native arm should IMPROVE, and by roughly the size of the bias -- the 6.3e-5 on
# O3_mean is suspiciously close to the 4.0e-5 the race was worth on the adjoint.
# If agreement does NOT improve, the correctness argument for defaulting the
# workaround on is weak here and the 6% is being paid for nothing.
#
# The traced side of the comparison already exists: logs/rx24-racefix.csv (slurm
# 10017939). Only the native reference was never kept. This job regenerates it
# over the identical window.
#
# COMPARE WITH:
#   python3 - <<'PY'
#   import csv
#   def load(p):
#       return {round(float(r["hours"]),4): r for r in csv.DictReader(open(p))}
#   a, b = load("logs/native24.csv"), load("logs/rx24-racefix.csv")
#   ks = sorted(set(a) & set(b))
#   for col in ("o3_mean","o3_min","o3_max","oh_max","no2_mean","m_min"):
#       w = max((abs(float(b[k][col])-float(a[k][col]))/max(abs(float(a[k][col])),1e-30), k)
#               for k in ks)
#       print(f"{col:10s} max rel {w[0]:.3e} at h={w[1]}")
#   print(f"{len(ks)} aligned records")
#   PY
#
# SIZING. The native runner is pure Julia -- no XLA compile, no Reactant tracing
# -- so it is far lighter than the traced one: build 683 s, solve 2,143 s for this
# window. 32G and 2 h are already generous. -t 8 matches every other measurement
# in this repo, including the 2,143 s figure.
#
# SUBMIT WITH:  mkdir -p logs && sbatch tools/diag/native_24h_reference.sbatch
# ===========================================================================
#SBATCH --job-name=reseact-nat24
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# NB: slurm will NOT create this directory -- submit with `mkdir -p logs` first
# or the job is rejected at submission. logs/ is gitignored.
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

JULIA="$(command -v julia)"
echo "host=$(hostname)  julia=$JULIA  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "started $(date -Is)"

export RESEACT_SOLVE_SECS=86400      # 24 h -- identical to logs/rx24-racefix.csv
export RESEACT_LABEL=reseact-nat24
export RESEACT_CSV=logs/native24.csv

"$JULIA" --project=run-model-jl -t 8 --heap-size-hint=24G run_reseact.jl

echo "finished $(date -Is)"
