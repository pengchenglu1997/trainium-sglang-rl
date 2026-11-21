#!/usr/bin/env bash
# One GPU: the CUDA baseline of the same RL loop. Standard SGLang in its container, Qwen3-0.6B, server and trainer on
# the one GPU, in-place weight sync through SGLang's own update_weights_from_disk (merged checkpoint).
# Expects under ~/w: this repository as kit/. Downloads the model and GSM8K. Last line C_DONE.
set -u
W=$HOME/w; R=$W/results_cuda; M=$W/models; PY=${PY:-/opt/pytorch/bin/python}
mkdir -p "$R/logs" "$M/gsm8k"
say() { echo "== C $(date -u +%H:%M:%S) $*"; }
I=lmsysorg/sglang:v0.5.23-cu130
docker pull -q "$I" > "$R/logs/docker_pull_sgl.out" 2>&1 || { say "image FAILED"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$R/logs/gpu.txt"
$PY -m pip install -q "transformers==5.18.0" "peft==0.21.2" "accelerate==1.15.0" safetensors > "$R/logs/pip.out" 2>&1 || { say "pip FAILED"; exit 1; }
[ -s "$M/Qwen3-0.6B/model.safetensors" ] || hf download Qwen/Qwen3-0.6B --local-dir "$M/Qwen3-0.6B" > "$R/logs/hf_06b.out" 2>&1 || { say "model FAILED"; exit 1; }
for s in train test; do
  [ -s "$M/gsm8k/$s.jsonl" ] || curl -sSfL -o "$M/gsm8k/$s.jsonl" \
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$s.jsonl" || { say "gsm8k FAILED"; exit 1; }
done
cd "$W/kit"
export MODELS_ROOT="$M" SGL_CUDA_IMAGE="$I"
# the server must leave room on the GPU for the trainer
export SGLANG_EXTRA="--mem-fraction-static 0.35 --max-running-requests 256"
loop() {  # tag steps extra...
  local tag="$1" steps="$2"; shift 2
  say "loop $tag start"
  "$PY" 30_rl_loop.py --run-tag "$tag" --backend sglang_cuda_docker --sync-method sglang_disk \
    --base-model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --eval-jsonl "$M/gsm8k/test.jsonl" \
    --device cuda --results "$R" --work-dir "$W/work_$tag" --ckpt-dir "$M/rl_$tag" --port 30020 \
    --questions-per-step 32 --lr 1e-4 --eval-n 500 --rollout-concurrency 32 --eval-concurrency 64 \
    --steps "$steps" "$@" > "$R/logs/loop_$tag.out" 2>&1
  say "loop $tag exit $?"
}
loop cuda_b4 25 --micro-bs 4
loop cuda_long_b4 150 --micro-bs 4 --eval-every 25
"$PY" 45_inplace_swap_probe.py --device cuda --model "$M/Qwen3-0.6B" --results "$R" --tag swap_cuda > "$R/logs/swap_cuda.out" 2>&1
echo C_DONE
