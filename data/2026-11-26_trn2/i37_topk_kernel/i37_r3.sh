#!/usr/bin/env bash
# i37 round 3/4, cold caches (a patch folder with other kernel sources refuses a cache built from the original ones).
#   bash i37_r3.sh <round> <nameA> <patchA> "<envA>" <nameB> <patchB> "<envB>"
set -u
cd "$HOME/w/i37" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/logs
say() { echo "== I37 $(date -u +%H:%M:%S) $*"; }
run() { "$PY" i37_collect.py "$@" > "results/logs/collect_$1.out" 2>&1; say "collect $1 exit $?: $(tail -n 1 results/logs/collect_$1.out)"; }
r="$1"
say "$r start"
run "$2" 0-3 30304 none "$3" "$4" 20 &
run "$5" 4-7 30305 none "$6" "$7" 20 &
wait
say "$r end"
echo "I37_${r}_DONE"
