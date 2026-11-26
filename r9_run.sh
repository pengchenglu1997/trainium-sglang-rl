#!/usr/bin/env bash
# Session 3, step r9, started by hand (the driver on the machine predates r9): simple serving optimisations of
# SGLang-Neuron on Qwen3-0.6B, then the table. Usage: r9_run.sh [budget minutes, default 150]
# Marker lines "== S <time> ..."; last line S_DONE r9 ...
set -u
cd "$(dirname "$0")" || exit 1
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export DATA_ROOT="${DATA_ROOT:-$HOME/w}" PY="$VENV/bin/python" MODELS_ROOT="${MODELS_ROOT:-$HOME/w/models}"
unset SGL_PATCH_DIR SGL_ENV SGL_EXTRA SGL_SCRIPT SGL_TOP_LOGPROBS SGL_SAMPLER SGL_MOUNTS SGL_CORES SGL_CACHE
R="$DATA_ROOT/results_s3"; BUDGET="${1:-150}"
say() { echo "== S $(date -u +%H:%M:%S) $*"; }
if [ -n "$(docker ps -q)" ]; then say "r9 refused: containers are running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; echo "S_DONE r9 status=refused"; exit 2; fi
say "r9 start, budget $BUDGET min, ready cap 1800 s"
READY_CAP=1800 timeout -s TERM -k 90 $(( (BUDGET + 12) * 60 )) "$PY" 81_opt_sweep.py --model Qwen3-0.6B --results "$R" --tag r9 \
  --budget-minutes "$BUDGET" > "$R/logs/s3_r9.out" 2>&1
rc=$?
say "r9 sweep exit $rc: $(tail -n 1 "$R/logs/s3_r9.out" | cut -c1-300)"
for n in $(docker ps -a --format '{{.Names}}' | grep '^optsweep_'); do
  docker logs "$n" > "$R/server_logs/sweep_${n}_$(date +%s).log" 2>&1; docker rm -f "$n" > /dev/null 2>&1
done
"$PY" 81_opt_table.py --results "$R" --tag r9 > "$R/logs/s3_81_r9.out" 2>&1; tb=$?
say "r9 table exit $tb: $(tail -n 1 "$R/logs/s3_81_r9.out" | cut -c1-200)"
echo "S_DONE r9 status=$([ "$rc" = 0 ] && [ "$tb" = 0 ] && echo ok || echo partial) rc=$rc table=$tb"
