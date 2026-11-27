#!/usr/bin/env bash
# Phase 1b: all-on with the engine's host trace (SGLANG_NEURON_HOST_TRACE, TP rank 0), benched after phase 1 ends
source "$HOME/w/host/scripts/hx_lib.sh"
ct=$(mkcache q06_all p1t); mkdir -p "$R/htrace"; sudo chmod 777 "$R/htrace"
start host_p1t 06 40-43 30502 "$ct" "" "SGLANG_NEURON_HOST_TRACE=/htrace/p1t SGLANG_NEURON_HOST_TRACE_RANKS=0" "$R/htrace:/htrace"
rt=$(wait_ready host_p1t 30502); say "ready trace=$rt"
until grep -q P1_DONE "$H/p1.log"; do sleep 10; done
if [ "$rt" != FAIL ]; then bench p1t_t0_p1 30502 06; bench p1t_t0_p2 30502 06; fi
stopsrv host_p1t
say P1B_DONE
