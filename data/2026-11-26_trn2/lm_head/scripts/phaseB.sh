#!/usr/bin/env bash
# Phase B: after the cold compiles of phase A.
#  B1 (no timing): 20-prompt t0 + GSM8K 200 for 0.6B switch 0 and 1 (sampler kernel on), cold compile of 0.6B s1n
#  B2 (timing, sequential on chips 48-51): llmbench 0.6B s0k, s1k, s1n; 8B s0k, s1k (two t0 passes + one t08 each)
#  B3 (no timing): 20-prompt t0 for 8B switch 0 and 1; compare
set -u
W=$HOME/w/lmh
X=$W/kit/s4_x_lmh.sh
P=$W/patch
L=$W/results/logs
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
say() { echo "== B $(date -u +%H:%M:%S) $*"; }
snap() { echo "-- docker ps $1 $(date -u +%H:%M:%S)"; docker ps --format '{{.Names}} {{.Status}}'; echo "-- load $(cut -d' ' -f1-3 /proc/loadavg)"; }
cc() { (cd $W/cc && "$PY" lmh_cc_collect.py "$@" > results/logs/collect_$1.out 2>&1); say "collect $1 exit $?: $(tail -n 1 $W/cc/results/logs/collect_$1.out)"; }
S0K="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0"
S1K="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
S1N="SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
bench() { snap "before bench $1"; bash $X bench "$@"; snap "after bench $1"; }

say "wait for phase A"
until grep -q PHASE_A_DONE $L/phaseA.log; do sleep 30; done
say "phase A done"; snap B1
cc q06_s0k 56-59 30556 t0,gsm8k &
cc q06_s1k 60-63 30557 t0,gsm8k &
bash $X prof lmh_q06_s1n 06 temperature 52-55 30552 $P "$S1N" &
wait
say "B1 done"
bench lmh_q06_s0k 06 temperature 48-51 30550 $P "$S0K" t
bench lmh_q06_s1k 06 temperature 48-51 30550 $P "$S1K" t
bench lmh_q06_s1n 06 temperature 48-51 30550 $P "$S1N" t
bench lmh_q8_s0k 8 temperature 48-51 30550 $P "$S0K" t
bench lmh_q8_s1k 8 temperature 48-51 30550 $P "$S1K" t
say "B2 done"; snap B3
cc q8_s0k 56-59 30556 t0 &
cc q8_s1k 60-63 30557 t0 &
wait
(cd $W/cc && "$PY" lmh_cc_compare.py > results/logs/compare.out 2>&1); say "compare exit $?"
say "B3 done"
echo PHASE_B_DONE
