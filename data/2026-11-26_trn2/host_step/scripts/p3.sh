#!/usr/bin/env bash
# Phase 3 (after phase 2): one change per start, C V C V against a fresh control host_p3c (all-on, cores 32-35).
#   p3v5  --disable-overlap-schedule (synchronous loop: host work no longer hidden behind the device step)
#   p3v6  SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=3 (env; default 2 per env_registry / 81_opt_sweep V17 notes)
#   p3v7  --context-length 2560 (decode contexts 768/1280/2560 instead of 1024/2048/4096: an llmbench row of
#         1025..1280 tokens then runs a 1280-context decode graph instead of the 2048 one; new graphs compile)
#   p3t   --disable-overlap-schedule + host trace (measurement only, not a lever row), started after p3v5 stops
source "$HOME/w/host/scripts/hx_lib.sh"
until grep -q P2_DONE "$H/p2.log"; do sleep 10; done
cc=$(mkcache q06_all p3c); c5=$(mkcache q06_all p3v5); c6=$(mkcache q06_all p3v6); c7=$(mkcache q06_all p3v7)
ct=$(mkcache q06_all p3t); mkdir -p "$R/htrace"; sudo chmod 777 "$R/htrace"
start host_p3c 06 32-35 30530 "$cc" "" ""
start host_p3v5 06 36-39 30531 "$c5" "--disable-overlap-schedule" ""
start host_p3v6 06 40-43 30532 "$c6" "" "SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=3"
start host_p3v7 06 44-47 30534 "$c7" "--context-length 2560" ""
r0=$(wait_ready host_p3c 30530); r5=$(wait_ready host_p3v5 30531); r6=$(wait_ready host_p3v6 30532)
say "ready ctrl=$r0 v5=$r5 v6=$r6"
pair() { bench "p3c_for_$1_p1" 30530 06; bench "$1_p1" "$2" 06 || { say "variant $1 bench failed, stopped"; return 1; }
  bench "p3c_for_$1_p2" 30530 06; bench "$1_p2" "$2" 06; }
[ "$r0" = FAIL ] && { say "control not ready"; for n in host_p3c host_p3v5 host_p3v6 host_p3v7; do stopsrv $n; done; exit 1; }
[ "$r5" != FAIL ] && pair p3v5 30531
stopsrv host_p3v5
start host_p3t 06 36-39 30533 "$ct" "--disable-overlap-schedule" "SGLANG_NEURON_HOST_TRACE=/htrace/p3t SGLANG_NEURON_HOST_TRACE_RANKS=0" "$R/htrace:/htrace"
[ "$r6" != FAIL ] && pair p3v6 30532
stopsrv host_p3v6
r7=$(wait_ready host_p3v7 30534); say "ready v7=$r7"
[ "$r7" != FAIL ] && pair p3v7 30534
stopsrv host_p3v7
stopsrv host_p3c
rt=$(wait_ready host_p3t 30533); say "ready t=$rt"
[ "$rt" != FAIL ] && bench p3t_t0_p1 30533 06
stopsrv host_p3t
say P3_DONE
