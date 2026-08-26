#!/bin/bash
# ===========================================================================
# conus_race_soak.sbatch -- does the XLA:CPU multi-thread wrong-answer fault
# still fire AT PRODUCTION SCALE, with the workaround OFF?
# ===========================================================================
# THE GAP THIS CLOSES. mwe_xla_thread_confirm.sbatch already ran 178,400
# real-model calls and 60,000 synthetic calls with RESEACT_ADJ_XLAFIX=0 on
# Reactant 0.2.280 and saw zero non-finite and zero bit-differing results --
# but every one of those calls was at 6x6x8 (288 cells, 3,744 states). The
# production grid is 13x7x72 (6,552 cells, 85,176 states), a 22.75x bigger
# program, and flipping a production default should be decided there.
#
# WHY A SOAK RATHER THAN A LONG TRAJECTORY. The fault is a per-CALL property of
# the compiled program (README-nondet.md: "executes it wrongly on about 1 call
# in 2,000"), so what buys statistical power is calls per CPU-hour, and a soak
# maximizes exactly that: no controller, no tape, no checkpoints, no VJP, just
# the compiled step issued back to back. A 48 h CONUS adjoint costs 8 h and
# issues ~28,000 accepted chemistry steps; this job issues more calls than that
# in under two hours, and at the affinity where the fault was WORST.
#
# THE AFFINITY IS THE EXPERIMENT. XLA:CPU sizes its intra-op pool from
# sched_getaffinity (tsl::port::MaxParallelism), so --cpus-per-task=4 IS the
# "taskset -c 8-11" configuration of README-nondet.md, at which the measured
# rate was ~1e-2 per call -- 20x worse than the ~5e-4 at 20 threads. One thread
# never faulted, so 4 is the cheapest way to be wrong quickly.
#
# WHAT THE HISTORICAL RATE PREDICTS FOR THIS JOB. 40,000 ros_step calls at
# 1e-2/call is ~400 faults, and 40,000 ssp_step calls on top of that; the walk
# adds 4,000 more along a real trajectory. Anything above single digits refutes
# "the race is fixed"; a clean sweep of ~84,000 CONUS calls puts the 95% upper
# bound on the rate at 3.6e-5 per call, i.e. ~280x below the documented 4-thread
# rate and ~14x below the documented 20-thread rate.
#
#   mkdir -p logs && sbatch tools/diag/conus_race_soak.sbatch
# ===========================================================================
#SBATCH --job-name=conusrace
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=${RESEACT_RXENV:-$PWD/run-model-jl}

export RESEACT_NLON=${RESEACT_NLON:-13}
export RESEACT_NLAT=${RESEACT_NLAT:-7}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
export RESEACT_ADJ_STAGES=none          # build + the two primal compiles, nothing else
export RESEACT_ADJ_CLAMP=0
export RESEACT_ADJ_JAC=sym
# Default OFF -- that is the question this job was built to answer. Export
# RESEACT_ADJ_XLAFIX=1 in the submitting shell (bare --export=ALL, never
# --export=ALL,K=V, which splits on commas) to run the CONTROL leg: the same
# soak with the workaround on, which is what turns "the fault still fires" into
# "the fault still fires and the workaround still suppresses it".
export RESEACT_ADJ_XLAFIX=${RESEACT_ADJ_XLAFIX:-0}
export RESEACT_LABEL=conusrace${RESEACT_ADJ_XLAFIX}-$SLURM_JOB_ID
export RESEACT_HAMMER_STAGES=${RESEACT_HAMMER_STAGES:-e1,soak,walk}
export RESEACT_HAMMER_PROGS=${RESEACT_HAMMER_PROGS:-ssp,ros}
export RESEACT_HAMMER_K=${RESEACT_HAMMER_K:-200}
export RESEACT_HAMMER_SOAK=${RESEACT_HAMMER_SOAK:-40000}
export RESEACT_HAMMER_WALK=${RESEACT_HAMMER_WALK:-4000}
export RESEACT_HAMMER_VJP=0

echo "=== node $(hostname) cpus=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE}M ==="
grep Cpus_allowed_list /proc/self/status
echo "=== driver md5: $(md5sum tools/adjoint_gradient.jl) ==="
echo "=== hammer md5: $(md5sum tools/diag/nondet_hammer.jl) ==="
echo "started $(date -Is)"

# This is a shared checkout with several agents in it, and a half-landed
# EarthSciAST include reorder has already cost this job one attempt: it loaded
# an uncached EarthSciAST at 14:06, EarthSciASTDiff was then rebuilt against the
# changed source mid-run, and the ssp_step trace died on "Scalar indexing is
# disallowed" 16 minutes in (slurm 10127142). Load it FIRST, in a throwaway
# process, and wait for the window to close rather than burning the build.
EASTSRC=/projects/illinois/eng/cee/ctessum/ctessum/code/EarthSciAST/pkg/EarthSciAST.jl/src
easthash() { find "$EASTSRC" -name '*.jl' -type f | sort | xargs md5sum | md5sum | cut -d' ' -f1; }
for attempt in $(seq 1 30); do
    if julia --project="$RESEACT_RXENV" -e 'using EarthSciAST, EarthSciASTDiff' >/dev/null 2>&1; then
        echo "=== EarthSciAST+Diff load (attempt $attempt), src hash $(easthash) ==="; break
    fi
    echo "=== EarthSciAST does not precompile (attempt $attempt); retrying in 60 s ==="
    sleep 60
done

julia --project="$RESEACT_RXENV" -t "$SLURM_CPUS_PER_TASK" --heap-size-hint=40G \
      tools/diag/nondet_hammer.jl
rc=$?
echo "finished $(date -Is) rc=$rc  EarthSciAST src hash now $(easthash)"
sacct -j "$SLURM_JOB_ID" --format=JobID%16,JobName%14,Elapsed,MaxRSS,AllocCPUS,State || true
