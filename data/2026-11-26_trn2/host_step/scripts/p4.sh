#!/usr/bin/env bash
# Phase 4: Qwen3-8B all-on, --context-length 2560 (p4v8) against the 8B all-on control (p4c8), C V C V.
# The variant compiles its new decode graphs; it starts on cores 40-43 as soon as phase 3 frees them.
source "$HOME/w/host/scripts/hx_lib.sh"
until grep -q '"stopped","name":"host_p3v6"' "$R/events.jsonl"; do sleep 10; done
cv=$(mkcache q8_all p4v8); cc=$(mkcache q8_all p4c8)
start host_p4v8 8 40-43 30541 "$cv" "--context-length 2560" ""
until grep -q P3_DONE "$H/p3.log"; do sleep 10; done
start host_p4c8 8 32-35 30540 "$cc" "" ""
r0=$(wait_ready host_p4c8 30540); rv=$(wait_ready host_p4v8 30541); say "ready ctrl8=$r0 v8=$rv"
if [ "$r0" != FAIL ] && [ "$rv" != FAIL ]; then
  bench p4c8_t0_p1 30540 8; bench p4v8_t0_p1 30541 8 && { bench p4c8_t0_p2 30540 8; bench p4v8_t0_p2 30541 8; }
fi
stopsrv host_p4c8; stopsrv host_p4v8
say P4_DONE
