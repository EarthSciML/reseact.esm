#!/usr/bin/env bash
# Binary-search the enzyme-hlo rewrite pattern responsible for the malformed
# `stablehlo.concatenate` (bug #2 in UPSTREAM_reverse_over_forward.md).
# Invariant: excluding the WHOLE candidate list makes NS=13 compile; excluding
# none makes it fail. Halve until one name is left.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CAND=$(grep -oE '[a-z_0-9]*concat[a-z_0-9]*|[a-z_0-9]*slice[a-z_0-9]*' \
       /tmp/claude-1111265/passes.txt | sort -u | tr '\n' ' ')
set -- $CAND
list=("$@")
i=0
while [ ${#list[@]} -gt 1 ]; do
  half=$(( ${#list[@]} / 2 ))
  first=("${list[@]:0:$half}")
  excl=$(printf '%s,' "${first[@]}"); excl=${excl%,}
  i=$((i+1))
  out=$(ROF_TIMEOUT=1800 "$REPO/tools/diag/rof_sweep.sh" "bisect$i" \
        ROF_NS=13 ROF_NC=5 ROF_THETA=vec ROF_JAC=ad ROF_MODE=vjp ROF_EXCL="$excl")
  echo "step $i: ${#first[@]} of ${#list[@]} excluded -> $out"
  if [[ "$out" == *"-> OK"* ]]; then
    list=("${first[@]}")            # culprit is in the excluded half
  else
    list=("${list[@]:$half}")       # culprit is in the other half
  fi
done
echo "CULPRIT PATTERN: ${list[0]}"
