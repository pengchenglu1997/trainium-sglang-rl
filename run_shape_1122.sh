#!/usr/bin/env bash
# trn2, 2026-11-22: what the compiled shapes cost when the load does not fill them (60_shape_cost.py), four servers
# side by side on their own cores. Qwen3-0.6B, patch and cache of the night session.
#   A  one decode bucket (32), context 1024            cores 12-15
#   B  the default decode buckets, context 1024        cores 16-19
#   C  one decode bucket (32), context 4096            cores 20-23
#   D  one decode bucket (32), context 8192            cores 24-27
# Last line S_DONE.
set -u
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
export DATA_ROOT=$HOME/w PY=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python
export MODELS_ROOT=$DATA_ROOT/models SGL_CACHE=$DATA_ROOT/sgl_cache_v0.1.0_night1121
cd "$(dirname "$0")"
export SGL_PATCH_DIR="$PWD/patches/sglang_neuron_v0.1.0_inplace"
R=$DATA_ROOT/results_night; M=$MODELS_ROOT
one() {  # tag cores port extra
  "$PY" 60_shape_cost.py --model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag "$1" \
    --cores "$2" --port "$3" --extra "$4" --env "RLKIT_DECODE_LADDER=0" > "$R/logs/shape_$1.out" 2>&1
  echo "== S $(date -u +%H:%M:%S) $1 exit $?"
}
one A 12-15 30101 "--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768 --cuda-graph-bs-decode 32" &
one B 16-19 30102 "--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768" &
one C 20-23 30103 "--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 32" &
one D 24-27 30104 "--context-length 8192 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 262144 --cuda-graph-bs-decode 32" &
wait
echo S_DONE
