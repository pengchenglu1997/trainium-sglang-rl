#!/usr/bin/env bash
# extra A/A controls for checks 3 and 4, on chips 56-59 and 60-63 (phase 3 holds 48-55)
set -u
cd "$HOME/w/cc" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
say() { echo "== CC $(date -u +%H:%M:%S) $*"; }
run() { "$PY" cc_collect.py "$@" > "results/logs/collect_$1.out" 2>&1; say "collect $1 exit $?: $(tail -n 1 results/logs/collect_$1.out)"; }
say "phase 4 start"
run rel_b    56-59 30331 t0,gsm8k &
run q8_rel_b 60-63 30332 t0 &
wait; say "phase 4 end"
echo "CC_PHASE4_DONE"
