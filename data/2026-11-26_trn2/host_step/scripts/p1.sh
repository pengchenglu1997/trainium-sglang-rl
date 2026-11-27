#!/usr/bin/env bash
# Phase 1: control (all-on, as tonight) and measurement server (all-on + --decode-log-interval 1 + SGLANG_LOG_FORWARD_ITERS=1)
source "$HOME/w/host/scripts/hx_lib.sh"
ca=$(mkcache q06_all p1c); cb=$(mkcache q06_all p1m)
start host_p1c 06 32-35 30500 "$ca" "" ""
start host_p1m 06 36-39 30501 "$cb" "--decode-log-interval 1" "SGLANG_LOG_FORWARD_ITERS=1"
rc=$(wait_ready host_p1c 30500); rm=$(wait_ready host_p1m 30501); say "ready ctrl=$rc meas=$rm"
if [ "$rc" != FAIL ] && [ "$rm" != FAIL ]; then
  bench p1c_t0_p1 30500 06; bench p1m_t0_p1 30501 06; bench p1c_t0_p2 30500 06; bench p1m_t0_p2 30501 06
fi
stopsrv host_p1c; stopsrv host_p1m
say P1_DONE
