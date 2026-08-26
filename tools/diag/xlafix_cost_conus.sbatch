#!/bin/bash
# ===========================================================================
# xlafix_cost_conus.sbatch -- what does the XLA:CPU race workaround COST at
# CONUS, and what does turning it off do to the answer?
# ===========================================================================
# A MATCHED PAIR IN ONE JOB: leg A with RESEACT_ADJ_XLAFIX=1, leg B with 0,
# same node, same process shape, same driver bytes, minutes apart. It measures
# the three things tools/adjoint_gradient.jl's blocker-4 comment claims and
# nobody has re-measured since 2026-08-19:
#
#   * COMPILE WALL TIME -- the four @compile lines. The retraction in that
#     comment ("at no wall-time cost" -- WRONG) is about the FORWARD PASS; the
#     compile has never been timed both ways at CONUS.
#   * PEAK MEMORY -- claimed "roughly doubles compile memory ... the ros_vjp
#     compile alone pushed cgroup anon past 30 GiB once the workaround was
#     switched on". Measured three ways: /usr/bin/time -v per LEG (the honest
#     per-process high-water mark, and it does include XLA's allocations --
#     they are in the same address space, they are just not in Julia's own
#     accounting); a 10 s trace of the JOB cgroup's memory.current, with leg
#     markers, so the peak can be attributed to a phase; and sacct MaxRSS,
#     which for this job is the max over BOTH legs and so is only a ceiling.
#     The limit that actually kills a job is memory.max in the job cgroup at
#     /sys/fs/cgroup/system.slice/slurmstepd.scope/<token>/ -- NOT the
#     node-wide /sys/fs/cgroup/memory.stat, which is a different number and
#     has misled a sizing decision here before.
#   * THE ANSWER -- J and the gradient. The comment records the workaround
#     moving J by 4.0e-5 relative, and attributes that NOT to rounding but to
#     the racy leg integrating a different trajectory via spurious NaN-driven
#     rejections. If the race is gone, the accept/reject ladders should now
#     agree step for step and the residual gap should collapse to a
#     reassociation floor.
#
# WHY BOTH LEGS RUN FROM A PRIVATE SNAPSHOT. This is a shared checkout with
# several agents in it, and the churn is fast enough to invalidate a matched
# pair outright. On 2026-08-24: two separately queued jobs ran against DIFFERENT
# bytes of tools/adjoint_gradient.jl 90 seconds apart (94362255 vs fa11308e);
# both then died on a half-landed EarthSciAST include reorder; and the first
# in-job attempt at this pair (slurm 10127204) had EarthSciAST change UNDERNEATH
# IT between the two legs (src hash 8a342184 -> a874ed8f), which changed the
# traced graph itself -- leg 0 traced 42,253 nodes against leg 1's 41,309, with
# `floor` and `minimum` ops that leg 1 never saw. Comparing J or compile time
# across that is meaningless.
#
# So this job first rsyncs a PRIVATE SNAPSHOT of everything the run depends on
# into ~/.cache: the four dev-pathed packages (EarthSciAST, EarthSciASTDiff,
# EarthSciASTSplitter, EarthSciIO -- about 8 MB together), the reseact.esm
# working tree, and run-model-jl's Project/Manifest with its four absolute
# `path =` entries rewritten to the snapshot. Both legs then run entirely out of
# the snapshot, so nothing anyone else does to the checkout can reach them. The
# hashes are still printed before and after each leg, now as a check that the
# snapshot itself is stable rather than as a hope about the checkout.
#
# NOT /tmp: it is node-local, and slurm jobs pointing there die in seconds.
#
# NMACRO=12 (one simulated hour) rather than 3: three macro steps is ~170
# accepted inner steps, too few to separate a trajectory split from a rounding
# difference; and rather than 576, which is an 8 h run and buys nothing this
# question needs. UJITTER stays at the driver default 0.1 -- the recorded
# 4.0e-5 was measured at the jittered base point, where the controller sits at
# its limits and a divergence shows up fastest.
#
#   mkdir -p logs && sbatch tools/diag/xlafix_cost_conus.sbatch
# ===========================================================================
#SBATCH --job-name=xfixcost
#SBATCH --partition=ctessum
#SBATCH --account=ctessum
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 4 cpus / 40G, NOT the 16 / 80G every previous CONUS adjoint used. Measured
# peak for ONE leg here is 17.8 GB (slurm 10127204), so 48G is ~2.7x headroom,
# and the small shape backfills in minutes where the 16/80G ask sat PENDING
# (Priority) behind other agents' jobs for over an hour. The compile is mostly
# single-threaded LLVM/MLIR work and this job compares the two legs against EACH
# OTHER inside one allocation, so the shape does not have to match the historical
# 16-cpu records -- but do not compare these compile seconds to those.
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
LIVE=/projects/illinois/eng/cee/ctessum/ctessum/code/reseact.esm
CODE=/projects/illinois/eng/cee/ctessum/ctessum/code
cd "$LIVE"

SNAP_ROOT=$HOME/.cache/reseact-xlafix/$SLURM_JOB_ID
mkdir -p "$SNAP_ROOT/pkgs" "$SNAP_ROOT/env"
rsync -a --exclude .git "$CODE/EarthSciAST/pkg/EarthSciAST.jl"             "$SNAP_ROOT/pkgs/"
rsync -a --exclude .git "$CODE/EarthSciASTDiff/pkg/EarthSciASTDiff.jl"     "$SNAP_ROOT/pkgs/"
rsync -a --exclude .git "$CODE/EarthSciASTSplitter.jl"                     "$SNAP_ROOT/pkgs/"
# trailing slash on the SOURCE: without it rsync nested this one as
# pkgs/EarthSciIO-julia/julia and the env reported EarthSciIO "not installed".
rsync -a --exclude .git "$CODE/EarthSciIO/julia/"                          "$SNAP_ROOT/pkgs/EarthSciIO-julia/"
rsync -a --exclude .git --exclude logs --exclude run-model-jl "$LIVE/"     "$SNAP_ROOT/reseact.esm/"
# prototypes/reseact_3d_chem/blockdiag_local.jl `include`s map_algorithm.jl and
# blockdiagonal.jl straight out of the EarthSciMLBase CHECKOUT -- not a Pkg
# dependency, a sibling-relative path -- so the snapshot needs that too, and the
# EARTHSCIMLBASE_SRC override is how the file lets you say where.
rsync -a --exclude .git "$CODE/EarthSciMLBase/src/"                        "$SNAP_ROOT/EarthSciMLBase/src/"
cp "$LIVE/run-model-jl/Project.toml" "$SNAP_ROOT/env/"
sed -e "s#^path = \"$CODE/EarthSciAST/pkg/EarthSciAST.jl\"#path = \"$SNAP_ROOT/pkgs/EarthSciAST.jl\"#" \
    -e "s#^path = \"$CODE/EarthSciASTDiff/pkg/EarthSciASTDiff.jl\"#path = \"$SNAP_ROOT/pkgs/EarthSciASTDiff.jl\"#" \
    -e "s#^path = \"$CODE/EarthSciASTSplitter.jl\"#path = \"$SNAP_ROOT/pkgs/EarthSciASTSplitter.jl\"#" \
    -e "s#^path = \"$CODE/EarthSciIO/julia\"#path = \"$SNAP_ROOT/pkgs/EarthSciIO-julia\"#" \
    "$LIVE/run-model-jl/Manifest.toml" > "$SNAP_ROOT/env/Manifest.toml"
[ "$(grep -c "^path = \"$SNAP_ROOT/" "$SNAP_ROOT/env/Manifest.toml")" -eq 4 ] || {
    echo "FATAL: expected 4 rewritten dev paths, got:"; grep '^path = ' "$SNAP_ROOT/env/Manifest.toml"; exit 1; }

REPO=$SNAP_ROOT/reseact.esm
export RESEACT_RXENV=$SNAP_ROOT/env
export RESEACT_MODEL=$REPO/reseact.esm
export EARTHSCIMLBASE_SRC=$SNAP_ROOT/EarthSciMLBase/src
EASTSRC=$SNAP_ROOT/pkgs/EarthSciAST.jl/src

export RESEACT_NLON=13 RESEACT_NLAT=7 RESEACT_NLEV=72
export RESEACT_ADJ_NMACRO=${RESEACT_ADJ_NMACRO:-12}
export RESEACT_ADJ_STAGES=fwd,adj
export RESEACT_ADJ_JAC=sym
export RESEACT_ADJ_CLAMP=0
export RESEACT_ADJ_OBJ=SuperFast.O3:surf

CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
JOBCG="/sys/fs/cgroup$(echo "$CG" | sed -E 's#(/system\.slice/slurmstepd\.scope/[^/]+).*#\1#')"
MEMTRACE="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.memtrace"
SNAP="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-driver.jl"

# The driver resolves the chemistry prototype, the reactant handoff and the
# model off `@__DIR__`; the copy we run lives in logs/, so pin REPO to the
# SNAPSHOT tree instead. logs/ is gitignored -- no cruft in the checkout.
sed -E 's#^const REPO = normpath\(joinpath\(@__DIR__, "\.\."\)\)#const REPO = "'"$REPO"'"#' \
    "$REPO/tools/adjoint_gradient.jl" > "$SNAP"
grep -q "^const REPO = \"$REPO\"" "$SNAP" || { echo "FATAL: REPO pin failed"; exit 1; }

easthash() { find "$EASTSRC" -name '*.jl' -type f | sort | xargs md5sum | md5sum | cut -d' ' -f1; }

HEAP_HINT="$(( SLURM_MEM_PER_NODE * 7 / 10 ))M"
echo "=== node $(hostname)  cpus=$SLURM_CPUS_PER_TASK  mem=${SLURM_MEM_PER_NODE}M  heap-size-hint=$HEAP_HINT ==="
echo "=== nmacro=$RESEACT_ADJ_NMACRO jac=$RESEACT_ADJ_JAC clamp=$RESEACT_ADJ_CLAMP ==="
echo "=== job cgroup: $JOBCG   memory.max: $(cat "$JOBCG/memory.max" 2>/dev/null || echo n/a) bytes ==="
echo "=== snapshot root: $SNAP_ROOT ==="
echo "=== driver snapshot md5: $(md5sum "$SNAP" | cut -d' ' -f1)  (live tools/adjoint_gradient.jl: $(md5sum "$LIVE/tools/adjoint_gradient.jl" | cut -d' ' -f1)) ==="
free -g | head -2
echo "started $(date -Is)"

# The snapshot can still have caught the checkout mid-edit (a half-landed
# include reorder killed two earlier attempts with UndefVarError: _Node). If it
# did, RE-TAKE it -- there is no point running an hour against a broken tree,
# and unlike the live checkout the snapshot only has to be good once.
for attempt in $(seq 1 30); do
    if julia --project="$RESEACT_RXENV" \
             -e 'using EarthSciAST, EarthSciASTDiff, EarthSciASTSplitter, EarthSciIO, Reactant' \
             >/dev/null 2>&1; then
        echo "=== snapshot loads (attempt $attempt), EarthSciAST src hash $(easthash) ==="; break
    fi
    echo "=== snapshot does not precompile (attempt $attempt); re-taking it in 60 s ==="
    sleep 60
    rsync -a --delete --exclude .git "$CODE/EarthSciAST/pkg/EarthSciAST.jl" "$SNAP_ROOT/pkgs/"
    rsync -a --delete --exclude .git "$CODE/EarthSciASTDiff/pkg/EarthSciASTDiff.jl" "$SNAP_ROOT/pkgs/"
done

( while :; do
    printf '%s %s\n' "$(date +%s)" "$(cat "$JOBCG/memory.current" 2>/dev/null || echo 0)"
    sleep 10
  done ) > "$MEMTRACE" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run_leg () {
    local fix=$1
    export RESEACT_ADJ_XLAFIX=$fix
    export RESEACT_LABEL=xfix${fix}-${SLURM_JOB_ID}
    export RESEACT_ADJ_CSV=logs/xfix${fix}-${SLURM_JOB_ID}.csv
    echo ""
    echo "##################################################################"
    echo "### LEG XLAFIX=$fix   start $(date -Is)  epoch $(date +%s)"
    echo "### EarthSciAST src hash before: $(easthash)"
    echo "##################################################################"
    # threads and heap hint track the ALLOCATION, not the 8/64G every historical
    # CONUS run used: -t 8 on a 4-cpu cgroup oversubscribes, and a 64G hint on a
    # 40G cgroup tells Julia not to collect until well past the OOM killer.
    /usr/bin/time -v julia --project="$RESEACT_RXENV" \
        -t "$SLURM_CPUS_PER_TASK" --heap-size-hint="${HEAP_HINT}" "$SNAP" 2>&1
    local rc=$?      # capture FIRST: the $(date ...) substitutions below would clobber it
    echo "### LEG XLAFIX=$fix   end   $(date -Is)  epoch $(date +%s)  rc=$rc"
    echo "### EarthSciAST src hash after:  $(easthash)"
}

run_leg 1
run_leg 0

kill $SAMPLER 2>/dev/null
echo ""
echo "finished $(date -Is)"
echo "=== cgroup memory.peak (BOTH legs): $(cat "$JOBCG/memory.peak" 2>/dev/null || echo n/a) bytes ==="
awk '{if ($2+0 > m) m = $2+0} END {printf "=== sampled memory.current max: %d bytes (%.2f GiB) over %d samples ===\n", m, m/1073741824, NR}' "$MEMTRACE"
echo "=== per-leg peaks: see the \"Maximum resident set size\" line in each leg's time -v block ==="
sacct -j "$SLURM_JOB_ID" --format=JobID%16,JobName%14,Elapsed,MaxRSS,AllocCPUS,State || true
echo "=== snapshot kept at $SNAP_ROOT (delete when the numbers are written up) ==="
