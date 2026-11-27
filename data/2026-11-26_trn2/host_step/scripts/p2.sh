#!/usr/bin/env bash
# Phase 2: launch-flag levers, one change per start, each benched back to back with the control (C V C V).
# Control host_p2c (all-on) stays up on cores 32-35. Variants on 36-39 and 40-43, two at a time.
# Qwen3-8B all-on with host trace on 44-47, started now, benched at the end.
source "$HOME/w/host/scripts/hx_lib.sh"
cc=$(mkcache q06_all p2c); c1=$(mkcache q06_all p2v1); c2=$(mkcache q06_all p2v2); c3=$(mkcache q06_all p2v3); c4=$(mkcache q06_all p2v4)
c8=$(mkcache q8_all p8t); mkdir -p "$R/htrace"; sudo chmod 777 "$R/htrace"
start host_p2c 06 32-35 30510 "$cc" "" ""
start host_p2v1 06 36-39 30511 "$c1" "--scheduler-recv-interval 8" ""
start host_p2v2 06 40-43 30512 "$c2" "--stream-interval 4" ""
start host_p8t 8 44-47 30520 "$c8" "" "SGLANG_NEURON_HOST_TRACE=/htrace/p8t SGLANG_NEURON_HOST_TRACE_RANKS=0" "$R/htrace:/htrace"
r0=$(wait_ready host_p2c 30510); r1=$(wait_ready host_p2v1 30511); r2=$(wait_ready host_p2v2 30512)
say "ready ctrl=$r0 v1=$r1 v2=$r2"
[ "$r0" = FAIL ] && { say "control not ready"; stopsrv host_p2c; stopsrv host_p2v1; stopsrv host_p2v2; stopsrv host_p8t; exit 1; }
pair() { # pair <vtag> <vport>
  bench "p2c_for_$1_p1" 30510 06; bench "$1_p1" "$2" 06 || { say "variant $1 bench failed, stopped"; return 1; }
  bench "p2c_for_$1_p2" 30510 06; bench "$1_p2" "$2" 06; }
[ "$r1" != FAIL ] && pair p2v1 30511
stopsrv host_p2v1
start host_p2v3 06 36-39 30513 "$c3" "--tokenizer-worker-num 2" ""
[ "$r2" != FAIL ] && pair p2v2 30512
stopsrv host_p2v2
start host_p2v4 06 40-43 30514 "$c4" "--detokenizer-worker-num 2" ""
r3=$(wait_ready host_p2v3 30513); say "ready v3=$r3"
[ "$r3" != FAIL ] && pair p2v3 30513
stopsrv host_p2v3
r4=$(wait_ready host_p2v4 30514); say "ready v4=$r4"
[ "$r4" != FAIL ] && pair p2v4 30514
stopsrv host_p2v4
stopsrv host_p2c
r8=$(wait_ready host_p8t 30520); say "ready 8b=$r8"
[ "$r8" != FAIL ] && { bench p8t_t0_p1 30520 8; bench p8t_t0_p2 30520 8; }
stopsrv host_p8t
say P2_DONE
