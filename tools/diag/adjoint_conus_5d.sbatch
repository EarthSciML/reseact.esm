#!/bin/bash
# ===========================================================================
# adjoint_conus_5d.sbatch -- d(mean surface O3)/d(theta) over FIVE DAYS of CONUS,
# every runtime scalar in the model in ONE backward sweep.
# ===========================================================================
# THIS IS THE DEFAULT CONFIGURATION OF run_reseact_adjoint.jl, which is why this
# file sets almost nothing. The runner's preset block is 13x7x72, 1,440 macro
# steps of 300 s = 120 h, clamp_nonneg ON, jac=:sym, the XLA:CPU race workaround
# on, un-jittered, stages fwd,adj -- and every one of those is a `get!`, so
# anything exported here would silently WIN over it. Keep that list in the runner
# where it is documented, not duplicated here where it can drift.
# (tools/diag/adjoint_conus_48h.sbatch is the opposite convention -- it drives
# tools/adjoint_gradient.jl directly and therefore has to set all of them.)
#
# WHAT IT SHOULD COST. Scaled from the 48 h CONUS gradient, which has run twice:
# slurm 10055533 (esm 1.0.0) 8 h 02 m, MaxRSS 38.4 GB, forward 7,460.23 s over
# 576 macro steps / 27,973 accepted inner steps, backward 18,761.89 s at
# 0.4585 s/VJP, ratio 2.51; and slurm 10044327 (esm 0.8.0) 8 h 18 m, MaxRSS
# 55.2 GB, with a BYTE-IDENTICAL accept/reject ladder. Five days is 2.5x that
# window, so ~20 h and ~40 GB, plus ~980 MB of macro-step checkpoints (681 kB
# each, 1,440 of them) held in memory. The linear scaling is pessimistic if
# anything: 24 h -> 48 h cost 1.64x, because the spin-up day is the stiff one.
# --time and --mem below carry roughly 2x margin on those figures, which is what
# the 48 h run's 86%-of-request MaxRSS says the margin has to be.
# Both 48 h runs used the STOCK emitter and pipeline; the driver has since
# defaulted to ESS_OOP_SSA=1 plus the excluded-passes pair (4.27x on the CONUS
# chemistry step, slurm 10154418), so the figures above are an upper bound
# until a window-length run is timed under the new defaults.
#
# WHAT IS AND IS NOT PROVEN. 48 h is, twice, with every reference-free check
# passing: the gather plan reproduces the host Jacobian at 0.000e+00, the
# fixed-sequence replay lands 0.000e+00 from all 576 checkpoints, zero
# flaky-reverse retries in 27,973 VJP calls, and scale*dJ/dscale == g0*dJ/dg0 to
# 4.6e-15. 120 h has NOT been run. The two things that used to kill long windows
# are both accounted for -- the 36.2 h forward death was missing PBL mixing, not
# window length, and the clamp/FD-Jacobian interaction that made the sweep
# non-finite is closed by jac=:sym -- so days 3-5 are unpaid-for rather than
# known-bad. Watch the first log line of each macro step: the replay check is
# per-checkpoint, so a faithfulness failure surfaces immediately, not at the end.
#
#   mkdir -p logs && sbatch tools/diag/adjoint_conus_5d.sbatch
#   squeue -u $USER ; tail -f logs/adj5d-<jobid>.out
#
# A shorter window without editing anything (the environment wins over the
# runner's defaults):
#   sbatch --export=ALL,RESEACT_ADJ_NMACRO=576 tools/diag/adjoint_conus_5d.sbatch
# ===========================================================================
#SBATCH --job-name=adj5d
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# The peak is the ssp_vjp COMPILE, not the window -- 1,440 macro steps add only
# ~980 MB of checkpoints on top of it. 80G was measured against the 48 h run
# (38.4 GB under esm 1.0.0, 55.2 GB under 0.8.0). Do not trim it: 96G sat PENDING
# on (Resources), and these nodes are often held by whole-node --mem=0 jobs, so
# the margin is cheaper than a requeue.
#SBATCH --mem=80G
# ~20 h projected; 48 h of walltime is 2.4x that. The partition allows 3 days.
#SBATCH --time=48:00:00
# NB: slurm will NOT create this directory -- submit with `mkdir -p logs` first
# or the job is rejected at submission. logs/ is gitignored.
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm

# run-model-jl, no overrides: since the esm 1.0.0 migration it develops all four
# packages at their release versions (EarthSciAST 0.1.1, EarthSciASTDiff 0.1.0,
# EarthSciASTSplitter 0.1.0, EarthSciIO 0.1.2), so jac=:sym needs no pinned env
# and the model is this repo's own working copy.
export RESEACT_RXENV=${RESEACT_RXENV:-/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }

export RESEACT_LABEL=${RESEACT_LABEL:-conus-adj5d-$SLURM_JOB_ID}
export RESEACT_ADJ_CSV=${RESEACT_ADJ_CSV:-logs/adj5d-${SLURM_JOB_ID}.csv}

echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M ==="
free -g | head -2
echo "started $(date -Is)"

# -t 8 matches every other measurement in this repo, so the timings stay
# comparable; the extra cores are headroom for XLA's own thread pool.
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G run_reseact_adjoint.jl

echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
