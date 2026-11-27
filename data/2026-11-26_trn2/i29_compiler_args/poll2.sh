#!/usr/bin/env bash
# poll2.sh <out dir> <container name>: records every neuronx-cc compile command line of that container
# (matched by the container id in /proc/<pid>/cgroup) until <out dir>/poll.stop exists
OUT="$1"; NAME="$2"; mkdir -p "$OUT/seen"
CID=""
while [ ! -e "$OUT/poll.stop" ]; do
  [ -n "$CID" ] || CID=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null)
  if [ -n "$CID" ]; then
    for p in $(pgrep -f "neuronx-cc compile" 2>/dev/null); do
      [ -e "$OUT/seen/$p" ] && continue
      touch "$OUT/seen/$p"
      grep -q "$CID" /proc/$p/cgroup 2>/dev/null || continue
      cl=$(sudo cat /proc/$p/cmdline 2>/dev/null | tr '\0' ' ')
      [ -n "$cl" ] || continue
      cwd=$(sudo readlink /proc/$p/cwd 2>/dev/null)
      echo "$(date -u +%FT%TZ) pid=$p cwd=$cwd CMD: $cl" >> "$OUT/cc_cmdlines.txt"
    done
  fi
  sleep 0.2
done
rm -rf "$OUT/seen"
echo POLL_STOPPED >> "$OUT/cc_cmdlines.txt"
