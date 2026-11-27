#!/usr/bin/env bash
# i37 round 2: kernel on with the single-line mask patch (patch_mask), and a second kernel-off server (A/A control
# for r1_koff). Caches copied from the same source as round 1 (q06_topk_koff), top_logprobs 20.
set -u
cd "$HOME/w/i37" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
P="$HOME/w/sglang_rl/patches/sglang_neuron_int_topk"
PM="$HOME/w/i37/patch_mask"
mkdir -p results/logs
say() { echo "== I37 $(date -u +%H:%M:%S) $*"; }
run() { "$PY" i37_collect.py "$@" > "results/logs/collect_$1.out" 2>&1; say "collect $1 exit $?: $(tail -n 1 results/logs/collect_$1.out)"; }
say "r2 start"
run r2_kon_mask 0-3 30302 q06_topk_koff "$PM" "RLKIT_SAMPLER_KERNEL_TOPK=1" 20 &
run r2_koff_b   4-7 30303 q06_topk_koff "$P" "" 20 &
wait
say "r2 end"
echo "I37_R2_DONE"
