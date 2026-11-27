#!/usr/bin/env bash
# GSM8K 200, T=0, on koff, kon (0.6B) and q8_rel, q8_all (8B); one server per chip, results in ~/w/cc/results_gsm8k2
set -u
cd "$HOME/w/cc" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1 CC_RES="$HOME/w/cc/results_gsm8k2"
mkdir -p "$CC_RES/logs"
say() { echo "== CC $(date -u +%H:%M:%S) $*"; }
run() { "$PY" cc_collect.py "$@" > "$CC_RES/logs/collect_$1.out" 2>&1; say "collect $1 exit $?: $(tail -n 1 $CC_RES/logs/collect_$1.out)"; }
say "gsm8k2 start"
run koff_a 48-51 30341 gsm8k &
run kon    52-55 30342 gsm8k &
run q8_rel 56-59 30343 gsm8k &
run q8_all 60-63 30344 gsm8k &
wait; say "gsm8k2 end"
for p in "koff_a kon" "q8_rel q8_all"; do
  set -- $p
  echo "COMPARE a=$1 b=$2: $("$PY" "$HOME/w/sglang_rl/21_gsm8k_acc.py" compare --a "$CC_RES/21_gsm8k_cc_$1.jsonl" --b "$CC_RES/21_gsm8k_cc_$2.jsonl")"
done
echo "CC_GSM8K2_DONE"
