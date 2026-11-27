#!/usr/bin/env bash
# cc_ correctness session: phases of at most four servers (chips 48-51, 52-55, 56-59, 60-63).
set -u
cd "$HOME/w/cc" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results/logs
say() { echo "== CC $(date -u +%H:%M:%S) $*"; }
run() { "$PY" cc_collect.py "$@" > "results/logs/collect_$1.out" 2>&1; say "collect $1 exit $?: $(tail -n 1 results/logs/collect_$1.out)"; }
phase="${1:-all}"
if [ "$phase" = all ] || [ "$phase" = 1 ]; then
  say "phase 1 start"
  run lmh_off_a 48-51 30301 t0,t08 &
  run lmh_off_b 52-55 30302 t0,t08 &
  run lmh_on    56-59 30303 t0,t08 &
  run all       60-63 30304 t0,gsm8k &
  wait; say "phase 1 end"
fi
if [ "$phase" = all ] || [ "$phase" = 2 ]; then
  say "phase 2 start"
  run koff_a 48-51 30311 t0,rl,minp,t08 &
  run koff_b 52-55 30312 t0,rl,minp,t08 &
  run kon    56-59 30313 t0,rl,minp,t08 &
  run rel    60-63 30314 t0,gsm8k &
  wait; say "phase 2 end"
fi
if [ "$phase" = all ] || [ "$phase" = 3 ]; then
  say "phase 3 start"
  run q8_all 48-51 30321 t0 &
  run q8_rel 52-55 30322 t0 &
  wait; say "phase 3 end"
fi
echo "CC_PHASES_DONE $phase"
