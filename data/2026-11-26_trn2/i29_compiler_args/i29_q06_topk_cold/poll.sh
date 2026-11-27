#!/usr/bin/env bash
# records every neuronx-cc compile command line (from /proc/<pid>/cmdline) with its cgroup, until ~/w/i29/poll.stop exists
W=$HOME/w/i29
CID=$(cat $W/cid.txt)
while [ ! -e "$W/poll.stop" ]; do
  for p in $(pgrep -f "neuronx-cc compile" 2>/dev/null); do
    [ -e "$W/seen/$p" ] && continue
    mkdir -p "$W/seen"; touch "$W/seen/$p"
    cl=$(sudo cat /proc/$p/cmdline 2>/dev/null | tr '\0' ' ')
    [ -n "$cl" ] || continue
    cg=$(cat /proc/$p/cgroup 2>/dev/null | tr '\n' ' ')
    case "$cg" in *"$CID"*) own=OWN ;; *) own=OTHER ;; esac
    [ "$own" = OWN ] || continue
    cwd=$(sudo readlink /proc/$p/cwd 2>/dev/null)
    echo "$(date -u +%FT%TZ) pid=$p cwd=$cwd CMD: $cl" >> "$W/cc_cmdlines.txt"
  done
  sleep 0.2
done
echo POLL_STOPPED >> "$W/cc_cmdlines.txt"
