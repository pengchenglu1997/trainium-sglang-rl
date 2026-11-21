#!/usr/bin/env bash
# trn2.48xlarge: the 150-step whole-machine loop, one run per seed, first not overlapped, then overlapped.
# Layout: 8 TP4 server replicas (cores 0-31), trainer on core 48 plus 15 workers (49-63), compiled training,
# adapter-only in-place sync, 32 questions x 8 answers a step, lr 1e-4, evaluation of 500 every 25 steps.
# Expects under $DATA_ROOT (default ~/w): this repository as kit/, models/Qwen3-0.6B, models/gsm8k/{train,test}.jsonl.
# usage: run_trn2_seeds.sh "<seeds>"        marker lines "== Q ..." ; last line Q_DONE
set -u
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
export DATA_ROOT=${DATA_ROOT:-$HOME/w} PY=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python
export MODELS_ROOT=$DATA_ROOT/models SGL_CACHE=$DATA_ROOT/sgl_cache_v0.1.0_patch_seeds
cd "$DATA_ROOT/kit"
export SGL_PATCH_DIR="$PWD/patches/sglang_neuron_v0.1.0_inplace"
export SGL_EXTRA="--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768 --cuda-graph-bs-decode 32"
R=$DATA_ROOT/results_seeds; M=$MODELS_ROOT
mkdir -p "$R/logs"
say() { echo "== Q $(date -u +%H:%M:%S) $*"; }
ok_file() { [ -f "$1" ] && [ "$(tail -n 1 "$1")" = '{"done": true}' ] && ! grep -q '"status": "fail"' "$1"; }
quiet() {   # nothing of a previous loop left on the cores
  for i in $(seq 1 60); do
    [ -z "$(docker ps -q)" ] && ! pgrep -f "30_rl_loop.py" > /dev/null && return 0
    sleep 5
  done
  return 1
}
loop() {  # tag seed extra...
  local tag="$1" seed="$2"; shift 2
  ok_file "$R/30_rl_loop_$tag.jsonl" && { say "$tag skipped (done)"; return 0; }
  quiet || { say "$tag: cores still busy"; return 1; }
  sleep 10
  say "$tag start"
  mkdir -p "$DATA_ROOT/work_$tag"
  NEURON_RT_VISIBLE_CORES=48 "$PY" 30_rl_loop.py --run-tag "$tag" --backend sglang_neuron_inplace \
    --sync-method sglang_disk --sync-payload adapter --train-compile --max-limit-share 0.8 \
    --replicas 8 --replica-cores "0-3;4-7;8-11;12-15;16-19;20-23;24-27;28-31" \
    --train-cores "49,50,51,52,53,54,55,56,57,58,59,60,61,62,63" \
    --base-model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --eval-jsonl "$M/gsm8k/test.jsonl" \
    --device neuron --core 48 --results "$R" --work-dir "$DATA_ROOT/work_$tag" --ckpt-dir "$M/rl_$tag" \
    --port 30200 --steps 150 --questions-per-step 32 --lr 1e-4 --eval-n 500 --eval-every 25 --seed "$seed" "$@" \
    > "$R/logs/loop_$tag.out" 2>&1
  say "$tag exit $?"
}
for s in $1; do loop "plain_s$s" "$s"; done
say "plain done"
for s in $1; do loop "overlap_s$s" "$s" --overlap; done
echo Q_DONE
