#!/usr/bin/env bash
# i37: one server run.   bash i37_one.sh <name> <cores> <port> <cache source tag or none> <patch dir> "<K=V ...>"
set -u
cd "$HOME/w/i37" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/logs
say() { echo "== I37 $(date -u +%H:%M:%S) $*"; }
say "$1 start"
"$PY" i37_collect.py "$1" "$2" "$3" "$4" "$5" "$6" 20 > "results/logs/collect_$1.out" 2>&1
say "collect $1 exit $?: $(tail -n 1 results/logs/collect_$1.out)"
echo "I37_ONE_DONE $1"
