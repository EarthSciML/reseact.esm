#!/bin/bash
# ===========================================================================
# p13_ssa_ab.sbatch -- the ess-oop-ssa READ-REDIRECT ARMS at a real resolution:
# timing A/B in one process, plus the whole-extended-buffer COPY census.
# ===========================================================================
# tools/diag/p13_ssa_ab.jl includes the driver TWICE (the arms are BUILD-time
# flags, so they cannot ride a trace-time `Ref` the way ess-oop-levelbase did),
# compiles each program in both arms, times them interleaved, gates the values
# bit for bit, and dumps the optimized HLO per arm so p11_ue_traffic.py can
# count the whole-buffer copies that are 45% of the transport VJP's write
# traffic.
#
# P13_ORDER swaps which arm compiles first: on this filesystem "which arm went
# first" moved `compile step` from 219.8 to 151.2 s for identical code, so both
# orders are run and only an effect that survives the swap is reported.
#
#   mkdir -p logs
#   sbatch --job-name=p13-4x5-offon --export=ALL,RESEACT_RES=4x5,P13_ORDER=off,on \
#          tools/diag/p13_ssa_ab.sbatch
#
# BASELINE to beat (slurm 10430018, 4x5 = 13x7x72, extended length 247 423):
#   whole-buffer copies  rhs 3,  rhs_vjp 99,  ssp_step 8,  ssp_vjp 394
#   real element writes  rhs 2.49 M, rhs_vjp 56.33 M, ssp_step 10.43 M,
#                        ssp_vjp 217.36 M (1739 MB, ~5.6 GB/s, write-bound)
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
CODE=/projects/illinois/eng/cee/ctessum/ctessum/code
export RESEACT_REPO=${RESEACT_REPO:-$CODE/reseact.esm}
cd "$RESEACT_REPO"
export RESEACT_MODEL=${RESEACT_MODEL:-$CODE/reseact.esm/reseact.esm}
export EARTHSCIMLBASE_SRC=${EARTHSCIMLBASE_SRC:-$CODE/EarthSciMLBase/src}
# Must name an environment whose EarthSciAST carries the redirect ARMS
# (`_oop_ssa_sub_enabled` &c.); the probe prints `oop_ssa_stats` per arm, and if
# the two arms print the same numbers the environment is the wrong one.
export RESEACT_RXENV=${RESEACT_RXENV:-$CODE/.ssaenv/run-model-jl}
[ -d "$RESEACT_RXENV" ] || { echo "FATAL: env $RESEACT_RXENV missing"; exit 1; }
export RESEACT_RES=${RESEACT_RES:-4x5}
export RESEACT_NLEV=${RESEACT_NLEV:-72}
# P13_SMALL=1 keeps whatever RESEACT_NLON/NLAT/NLEV the caller exported (the
# 6x6x8 validation box); otherwise the resolution row's CONUS box is imposed,
# because the probe `get!`s NLON/NLAT to 6 before including the driver and would
# otherwise never reach CONUS.
if [ "${P13_SMALL:-0}" != "1" ]; then
  case "$RESEACT_RES" in
    4x5)   export RESEACT_NLON=13 RESEACT_NLAT=7  ;;
    2x2.5) export RESEACT_NLON=25 RESEACT_NLAT=13 ;;
    *) echo "FATAL: no CONUS box wired for RESEACT_RES=$RESEACT_RES"; exit 1 ;;
  esac
fi
export P13_PROGRAMS=${P13_PROGRAMS:-rhs,rhs_vjp}
export P13_ORDER=${P13_ORDER:-off,on}
export P13_REPS=${P13_REPS:-40}
echo "=== node $(hostname) res=$RESEACT_RES nlev=$RESEACT_NLEV cpus=$SLURM_CPUS_PER_TASK progs=$P13_PROGRAMS order=$P13_ORDER ==="
echo "started $(date -Is)"
julia --project="$RESEACT_RXENV" -t 8 --heap-size-hint=64G tools/diag/p13_ssa_ab.jl
G=$(ls -d logs/p13-ssa-*-"$SLURM_JOB_ID" 2>/dev/null | head -1)
echo "=== dump dir $G"
for prog in ${P13_PROGRAMS//,/ }; do
  for arm in off on; do
    python3 tools/diag/p11_ue_traffic.py "$G/$prog-$arm"
  done
done
echo "finished $(date -Is)"
sacct -j "$SLURM_JOB_ID" --format=JobID,MaxRSS,Elapsed,State || true
