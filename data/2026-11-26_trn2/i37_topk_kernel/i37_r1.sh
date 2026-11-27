#!/usr/bin/env bash
# i37 round 1: kernel off vs kernel on, top_logprobs 20, both caches copied from the same source (q06_topk_koff).
set -u
cd "$HOME/w/i37" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
P="$HOME/w/sglang_rl/patches/sglang_neuron_int_topk"
mkdir -p results/logs
say() { echo "== I37 $(date -u +%H:%M:%S) $*"; }
run() { "$PY" i37_collect.py "$@" > "results/logs/collect_$1.out" 2>&1; say "collect $1 exit $?: $(tail -n 1 results/logs/collect_$1.out)"; }
say "r1 start"
run r1_koff 0-3 30300 q06_topk_koff "$P" "" 20 &
run r1_kon  4-7 30301 q06_topk_koff "$P" "RLKIT_SAMPLER_KERNEL_TOPK=1" 20 &
wait
say "r1 end"
echo "I37_R1_DONE"
