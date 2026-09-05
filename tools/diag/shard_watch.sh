#!/bin/bash
# shard_watch.sh JOBID... -- emit milestone/failure lines from each job's
# shard_run.sh log and its Slurm state until every job has left the queue.
cd "$(dirname "$0")/../.."
declare -A seen; declare -A state
PAT='^BUILD|SHARDS|shard sizes|shard builds|SHARDS READY|parameters:|forward pass|accept/reject|J = |backward sweep|of which|fan-out|sum of shard|worker RSS|STRUCTURAL|wrote|ERROR|Error|error:|Killed|OOM|finished rc|On worker'
while true; do
  left=0
  for J in "$@"; do
    s=$(squeue -j "$J" -h -o %T 2>/dev/null); s=${s:-GONE}
    if [ "${state[$J]:-}" != "$s" ]; then echo "job $J state: $s $(date +%H:%M)"; state[$J]=$s; fi
    L=$(ls logs/*-"$J".log 2>/dev/null | head -1)
    if [ -n "$L" ]; then
      n=$(wc -l < "$L"); p=${seen[$J]:-0}
      if [ "$n" -gt "$p" ]; then
        tail -n +"$((p+1))" "$L" | grep -E "$PAT" | cut -c1-220 | sed "s/^/[$J] /"
        seen[$J]=$n
      fi
    fi
    [ "$s" = "GONE" ] || left=$((left+1))
  done
  if [ "$left" -eq 0 ]; then
    sacct -j "$(IFS=,; echo "$*")" -X --format=JobID,Elapsed,MaxRSS,State | tail -n "$#"
    echo "WATCH_END"; break
  fi
  sleep 60
done
