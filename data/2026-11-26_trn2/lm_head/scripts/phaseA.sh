#!/usr/bin/env bash
# Phase A: cold compiles in parallel on chips 48-63 (branch feat/lm-head-pretransposed-main, patch ~/w/lmh/patch)
set -u
X=$HOME/w/lmh/kit/s4_x_lmh.sh
P=$HOME/w/lmh/patch
L=$HOME/w/lmh/results/logs
say() { echo "== A $(date -u +%H:%M:%S) $*"; }
snap() { echo "-- docker ps $1 $(date -u +%H:%M:%S)"; docker ps --format '{{.Names}} {{.Status}}'; }
snap start
say start
bash $X prof lmh_q06_s0k 06 temperature 48-51 30550 $P "RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0" &
bash $X prof lmh_q06_s1k 06 temperature 52-55 30551 $P "RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1" &
bash $X prof lmh_q8_s1k 8 temperature 56-59 30552 $P "RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1" &
bash $X prof lmh_q8_s0k 8 temperature 60-63 30553 $P "RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0" &
wait
snap end
say end
echo PHASE_A_DONE
