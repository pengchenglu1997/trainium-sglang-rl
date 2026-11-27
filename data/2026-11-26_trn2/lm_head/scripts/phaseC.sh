#!/usr/bin/env bash
# Phase C: the control the s1n row lacks: switch 0 without the sampler kernel (main as released sampler),
# cold compile during phase B3, then bench s0n and s1n again back to back (s1n from a copy of its own cache).
set -u
W=$HOME/w/lmh
X=$W/kit/s4_x_lmh.sh
P=$W/patch
L=$W/results/logs
say() { echo "== C $(date -u +%H:%M:%S) $*"; }
snap() { echo "-- docker ps $1 $(date -u +%H:%M:%S)"; docker ps --format '{{.Names}} {{.Status}}'; echo "-- load $(cut -d' ' -f1-3 /proc/loadavg)"; }
bench() { snap "before bench $1"; bash $X bench "$@"; snap "after bench $1"; }
S0N="SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0"
S1N="SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
say "wait for B2"
until grep -q "B2 done" $L/phaseB.log; do sleep 30; done
snap C1
bash $X prof lmh_q06_s0n 06 temperature 52-55 30552 $P "$S0N"
sudo -n cp -a $HOME/w/sgl_cache_lmh_q06_s1n $HOME/w/sgl_cache_lmh_q06_s1n_r2
say "copied s1n cache: $(sudo -n find $HOME/w/sgl_cache_lmh_q06_s1n_r2 -type f | wc -l) files"
until grep -q PHASE_B_DONE $L/phaseB.log; do sleep 30; done
say "phase B done"
bench lmh_q06_s0n 06 temperature 48-51 30550 $P "$S0N" t
bench lmh_q06_s1n_r2 06 temperature 48-51 30550 $P "$S1N" t
say "C done"
echo PHASE_C_DONE
