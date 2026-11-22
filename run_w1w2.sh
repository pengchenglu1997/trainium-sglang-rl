#!/usr/bin/env bash
# trn2 session 1 of the Qwen3 plan (deliverables/trainium-qwen3-experiment-plan-2026-11-22.md): W1 serving and sync,
# W2 8B training probes. Each row is its own command, so rows run side by side on their own cores.
#   run_w1w2.sh prep
#   run_w1w2.sh w1.1   8B adapter update with the shard-only delta                       cores 8-11
#   run_w1w2.sh w1.2   decode context ladder on, context 8192                             cores 12-15
#   run_w1w2.sh w1.3   PR 4 needle test at context 16384, with and without                cores 16-19, 20-23
#   run_w1w2.sh w1.4   one decode bucket against the default set, one group of 32         (read from w1.2 A/B of the night; rerun here) cores 24-27
#   run_w1w2.sh w1.5   update while long requests run                                     cores 28-31
#   run_w1w2.sh w1.6   8B rollout speed, TP4 x 2 replicas                                  cores 32-39
#   run_w1w2.sh w1.7   Miles disk-delta sequence through the pull_weights shim            cores 40-43
#   run_w1w2.sh w2.1 | w2.2 | w2.3 | w2.4 | w2.5                                          cores 48-55
# Environment-dependent values come from the environment, not from this file: SGL_IMAGE (default the v0.1.0 image),
# VENV (default the Neuron DLAMI torch-neuronx 2.14 venv). Marker lines "== W <time> ..."; last line W_DONE <row>.
# Not run anywhere yet: the flags of w1.3 (context 16384) and the 8B example script path are unverified.
set -u
VENV="${VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}"
source "$VENV/bin/activate"
export DATA_ROOT="${DATA_ROOT:-$HOME/w}" PY="$VENV/bin/python"
export MODELS_ROOT=$DATA_ROOT/models SGL_CACHE=$DATA_ROOT/sgl_cache_v0.1.0_w1
cd "$(dirname "$0")"
KIT="$PWD"
export SGL_PATCH_DIR="$KIT/patches/sglang_neuron_v0.1.0_inplace"
IMAGE="${SGL_IMAGE:-public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04}"
export SGL_IMAGE="$IMAGE"
R=$DATA_ROOT/results_w1w2; M=$MODELS_ROOT
SMALL="--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768 --cuda-graph-bs-decode 32"
Q8=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh
mkdir -p "$R/logs" "$M" "$SGL_CACHE"
say() { echo "== W $(date -u +%H:%M:%S) $*"; }
fetch() {
  if [ -f "$M/$2/.fetch_ok" ]; then say "model $2 present"; return 0; fi
  docker run --rm -v "$M":/models "$IMAGE" hf download "$1" --local-dir "/models/$2" > "$R/logs/hf_$2.out" 2>&1 \
    || { say "model $2 FAILED"; return 1; }
  sudo chown -R "$(id -u):$(id -g)" "$M/$2" 2>/dev/null || true
  touch "$M/$2/.fetch_ok"; say "model $2 ok $(du -sh "$M/$2" | cut -f1)"
}
WHAT="${1:?row}"
case "$WHAT" in
  prep)
    docker pull "$IMAGE" > "$R/logs/docker_pull.out" 2>&1 || { say "prep FAILED docker pull"; exit 1; }
    fetch Qwen/Qwen3-0.6B Qwen3-0.6B & fetch Qwen/Qwen3-8B Qwen3-8B & wait
    mkdir -p "$M/gsm8k"
    for s in train test; do
      [ -s "$M/gsm8k/$s.jsonl" ] || curl -sSfL -o "$M/gsm8k/$s.jsonl" \
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$s.jsonl" || say "gsm8k $s FAILED"
    done
    for m in Qwen3-0.6B Qwen3-8B; do "$PY" mk_alt.py --base "$M/$m" --alt "$M/${m}_alt" > "$R/logs/alt_$m.out" 2>&1 & done; wait
    "$PY" -m pip install -q zstandard xxhash > "$R/logs/pip.out" 2>&1 || say "pip zstandard/xxhash FAILED (w1.7 needs them)"
    docker run --rm --entrypoint cat "$IMAGE" "$Q8" > "$R/logs/q8_run_server.sh" 2>&1; say "8B example script: $(wc -l < "$R/logs/q8_run_server.sh") lines"
    "$PY" selftest_fake_server.py | tail -1
    ;;
  w1.1)
    SGL_CORES=8-11 SGL_SCRIPT=$Q8 SGL_EXTRA="$SMALL" "$PY" 50_adapter_server_test.py --base "$M/Qwen3-8B" \
      --work "$M/adapter_test_8b" --results "$R" --tag q8_lazy > "$R/logs/w1.1.out" 2>&1
    grep -h '"event": "update"\|"event": "check"\|"status": "fail"' "$R/50_adapter_q8_lazy.jsonl" | cut -c1-260
    grep -h "adapter update from" "$R"/server_logs/*q8_lazy*.log | grep -o "delta [0-9.]*s, copy [0-9.]*s, total [0-9.]*s; lazy.*" | sort | uniq -c | head -8
    ;;
  w1.2)
    "$PY" 60_shape_cost.py --model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag ladder8k \
      --cores 12-15 --port 30112 --env "RLKIT_DECODE_LADDER=1" \
      --extra "--context-length 8192 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 262144 --cuda-graph-bs-decode 32" \
      > "$R/logs/w1.2.out" 2>&1
    grep -h '"pass"' "$R/60_shape_cost_ladder8k.jsonl" | grep -o '"concurrent": [0-9]*\|"step_ms": [0-9.]*' | paste - -
    ;;
  w1.3)
    X="--max-running-requests 4 --max-total-tokens 65536 --chunked-prefill-size 2048 --cuda-graph-bs-decode 4"
    "$PY" 67_long_context_needle.py --model "$M/Qwen3-0.6B" --results "$R" --tag nopr4 --cores 16-19 --port 30113 \
      --context-length 16384 --patch-dir "$KIT/patches/sglang_neuron_v0.1.0_inplace" --extra "$X" > "$R/logs/w1.3_nopr4.out" 2>&1 &
    "$PY" 67_long_context_needle.py --model "$M/Qwen3-0.6B" --results "$R" --tag pr4 --cores 20-23 --port 30114 \
      --context-length 16384 --patch-dir "$KIT/patches/sglang_neuron_v0.1.0_inplace_pr4" --extra "$X" > "$R/logs/w1.3_pr4.out" 2>&1 &
    wait
    tail -n 12 "$R"/67_*needle*.jsonl 2>/dev/null | cut -c1-240
    ;;
  w1.4)
    for t in "one:--cuda-graph-bs-decode 32:30115:24-27" "default::30116:44-47"; do
      IFS=: read -r tag flag port cores <<< "$t"
      "$PY" 60_shape_cost.py --model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag "bucket_$tag" \
        --cores "$cores" --port "$port" --levels 32 --env "RLKIT_DECODE_LADDER=0" \
        --extra "--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768 $flag" \
        > "$R/logs/w1.4_$tag.out" 2>&1 &
    done; wait
    grep -h '"pass"' "$R"/60_shape_cost_bucket_*.jsonl | grep -o '"seconds": [0-9.]*\|"tokens_per_s": [0-9.]*' | paste - -
    ;;
  w1.5)
    SGL_EXTRA="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 8 --max-total-tokens 32768 --cuda-graph-bs-decode 8" \
      "$PY" 68_update_under_load.py --base "$M/Qwen3-0.6B" --alt "$M/Qwen3-0.6B_alt" --results "$R" --tag w15 --cores 28-31 \
      > "$R/logs/w1.5.out" 2>&1
    tail -n 6 "$R"/68_*w15*.jsonl | cut -c1-300
    ;;
  w1.6)
    SGL_SCRIPT=$Q8 SGL_EXTRA="$SMALL" "$PY" 55_rollout_bench.py --model "$M/Qwen3-8B" --train-jsonl "$M/gsm8k/train.jsonl" \
      --results "$R" --tag q8_tp4x2 --replica-cores "32-35;36-39" --questions 32 --k 8 --port 30160 > "$R/logs/w1.6.out" 2>&1
    grep -h '"event": "pass"' "$R/55_rollout_bench_q8_tp4x2.jsonl" | cut -c1-260
    ;;
  w1.7)
    # server on 30170, shim on 30171; the replay plays Miles's disk-delta sequence through the shim
    SGL_CORES=40-43 SGL_EXTRA="$SMALL" "$PY" 23_restart_time.py --backend sglang_neuron_inplace --model "$M/Qwen3-0.6B" \
      --results "$R" --tag miles_srv --port 30170 --name srv_miles --repeats 0 --leave-running > "$R/logs/w1.7_srv.out" 2>&1
    mkdir -p "$M/miles_local"
    nohup "$PY" 66_pull_weights_shim.py --port 30171 --upstream http://127.0.0.1:30170 --base-dir "$M/Qwen3-0.6B" \
      --host-root "$M" --served-root /models > "$R/logs/w1.7_shim.out" 2>&1 &
    SHIM=$!
    sleep 3
    "$PY" 64_miles_replay.py --help > "$R/logs/w1.7_replay_help.txt" 2>&1
    say "w1.7: server and shim are up (shim pid $SHIM); run 64_miles_replay.py against http://127.0.0.1:30171 with the options of logs/w1.7_replay_help.txt, then: kill $SHIM; bash srv_sglang_neuron.sh stop srv_miles"
    ;;
  w2.1)
    NEURON_RT_VISIBLE_CORES=48 "$PY" 70_mem_probe.py --model "$M/Qwen3-8B" --device neuron --core 48 --results "$R" --tag w21 \
      --stages load,forward > "$R/logs/w2.1.out" 2>&1; cut -c1-260 "$R"/70_*w21*.jsonl
    ;;
  w2.2)
    NEURON_RT_VISIBLE_CORES=49 "$PY" 70_mem_probe.py --model "$M/Qwen3-0.6B" --device neuron --core 49 --results "$R" --tag w22 \
      > "$R/logs/w2.2.out" 2>&1; cut -c1-260 "$R"/70_*w22*.jsonl
    NEURON_RT_VISIBLE_CORES=50 "$PY" 70_mem_probe.py --model "$M/Qwen3-0.6B" --device neuron --core 50 --results "$R" --tag w22c \
      --compile > "$R/logs/w2.2c.out" 2>&1; cut -c1-260 "$R"/70_*w22c*.jsonl
    ;;
  w2.3)
    NEURON_RT_VISIBLE_CORES=51 "$PY" 70_mem_probe.py --model "$M/Qwen3-8B" --device neuron --core 51 --results "$R" --tag w23 \
      > "$R/logs/w2.3.out" 2>&1; cut -c1-260 "$R"/70_*w23*.jsonl
    NEURON_RT_VISIBLE_CORES=52 "$PY" 70_mem_probe.py --model "$M/Qwen3-8B" --device neuron --core 52 --results "$R" --tag w23c \
      --compile > "$R/logs/w2.3c.out" 2>&1; cut -c1-260 "$R"/70_*w23c*.jsonl
    ;;
  w2.4)
    "$PY" 71_handoff_probe.py --device neuron --cores 53,54 --results "$R" --tag w24 > "$R/logs/w2.4.out" 2>&1
    cut -c1-300 "$R"/71_*w24*.jsonl | tail -n 6
    ;;
  w2.5)
    NEURON_LOGICAL_NC_CONFIG=2 TORCH_NEURONX_ENABLE_HOST_CC=0 OMP_NUM_THREADS=1 "$PY" 72_collective_backward_probe.py \
      --device neuron --world 4 --cores 56,57,58,59 --results "$R" --tag w25 > "$R/logs/w2.5.out" 2>&1
    cut -c1-300 "$R"/72_*w25*.jsonl | tail -n 8
    ;;
  *) echo "usage: run_w1w2.sh prep|w1.1..w1.7|w2.1..w2.5"; exit 2 ;;
esac
echo "W_DONE $WHAT"
