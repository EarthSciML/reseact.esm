#!/bin/bash
# TARGET C, synthetic half: run tools/diag/mwe_xla_thread_nondet.jl at ONE,
# FOUR and TWENTY XLA:CPU intra-op threads in the same allocation. The thread
# count is set by the affinity mask (XLA reads sched_getaffinity), so each leg
# is a separate `taskset` process; the 1-cpu leg writes the golden vector that
# the other two are checked against bit-for-bit.
#SBATCH --job-name=mweCsyn
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd /projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
export RESEACT_RXENV=$PWD/run-model-jl
export MWE_LANES=${MWE_LANES:-3744} MWE_DEPTH=${MWE_DEPTH:-60} MWE_CALLS=${MWE_CALLS:-20000}
rm -f tools/diag/logs/mwe_xla_thread_nondet.golden
# The allocation's own cpu list; take the first N of it for each leg.
CPUS=$(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')
echo "=== node $(hostname) alloc cpus=$CPUS ==="
mapfile -t LIST < <(python3 -c "
import sys
out=[]
for part in '$CPUS'.split(','):
    if '-' in part:
        a,b=part.split('-'); out += list(range(int(a),int(b)+1))
    else: out.append(int(part))
print(' '.join(map(str,out)))
")
read -ra C <<< "${LIST[0]}"
one="${C[0]}"
four=$(IFS=,; echo "${C[*]:0:4}")
all=$(IFS=,; echo "${C[*]}")
echo "legs: 1cpu=$one  4cpu=$four  ${#C[@]}cpu=$all"

echo; echo "########## LEG 1: ONE thread (writes the golden) ##########"
MWE_GOLDEN=write taskset -c "$one" julia --project="$RESEACT_RXENV" -t 1 tools/diag/mwe_xla_thread_nondet.jl
echo; echo "########## LEG 2: FOUR threads ##########"
MWE_GOLDEN=check taskset -c "$four" julia --project="$RESEACT_RXENV" -t 1 tools/diag/mwe_xla_thread_nondet.jl
echo; echo "########## LEG 3: ${#C[@]} threads ##########"
MWE_GOLDEN=check taskset -c "$all" julia --project="$RESEACT_RXENV" -t 1 tools/diag/mwe_xla_thread_nondet.jl
echo "finished $(date -Is)"
