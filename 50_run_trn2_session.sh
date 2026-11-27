#!/usr/bin/env bash
# One trn2 session of the SGLang / RL plan (../NEXT_ROUND_PLAN_2026-11-20.md), in order.
#   bash 50_run_trn2_session.sh [step ...]      default: all steps
# Steps: prep sgl_up sgl_acc sgl_probes sgl_bench loop_sgl loop_hf loop_vllm ddp cache_record (last: size and file count of each cache folder)
# Start it detached: nohup setsid bash 50_run_trn2_session.sh > $R/session_$(date +%s).out 2>&1 &
# It can be started again: a step whose result file is finished without a fail line is skipped.
# One marker line per step: "== S <time> <step> ok|FAILED|skipped ...".
#
# Cores (logical, LNC 2: four per chip): SGLang server 0-3, vLLM server 16-19, trainer 8,
# DDP 32-39. Nothing here starts, stops or changes the machine itself.
# Required: DATA_ROOT (a disk with about 60 GB free). Optional: DEADLINE_UNIX (no step starts
# after it), STEPS_RL (default 20), PY (the host Python with torch-neuronx, transformers, peft).
set -u
cd "$(dirname "$0")"
: "${DATA_ROOT:?}"
DATA_ROOT="$(realpath "$DATA_ROOT")"; export DATA_ROOT
export MODELS_ROOT="$DATA_ROOT/models"
export SGL_CACHE="$DATA_ROOT/sgl_cache_v0.1.0"
export R="${R:-$DATA_ROOT/results_sgl}"
PY="${PY:-python}"
STEPS_RL="${STEPS_RL:-20}"
BASE="$MODELS_ROOT/Qwen3-0.6B"
GSM="$MODELS_ROOT/gsm8k"
WORK="$DATA_ROOT/work"
IMAGE="${SGL_IMAGE:-public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04}"
SRV=srv_sgl
PORT=30000
mkdir -p "$R/logs" "$MODELS_ROOT" "$WORK"

say() { echo "== S $(date -u +%H:%M:%S) $*"; }
ok_file() { [ -f "$1" ] && [ "$(tail -n 1 "$1")" = '{"done": true}' ] && ! grep -q '"status": "fail"' "$1"; }
in_time() { [ -z "${DEADLINE_UNIX:-}" ] || [ "$(date +%s)" -lt "$DEADLINE_UNIX" ]; }
sgl_alive() { [ "$(bash srv_sglang_neuron.sh alive "$SRV")" = true ]; }
sgl_served() { echo "/models/Qwen3-0.6B"; }

step_prep() {
  nproc; free -g | sed -n 2p; df -h "$DATA_ROOT" | tail -1
  neuron-ls 2>&1 | head -30 > "$R/logs/neuron_ls.txt"; ls /dev/neuron* | wc -l
  docker --version || { say "prep FAILED no docker"; return 1; }
  docker pull "$IMAGE" > "$R/logs/docker_pull.out" 2>&1 || { say "prep FAILED docker pull"; return 1; }
  if [ ! -s "$BASE/model.safetensors" ]; then
    docker run --rm -v "$MODELS_ROOT":/models "$IMAGE" hf download Qwen/Qwen3-0.6B \
      --local-dir /models/Qwen3-0.6B > "$R/logs/hf_download.out" 2>&1 || { say "prep FAILED model download"; return 1; }
    sudo chown -R "$(id -u):$(id -g)" "$MODELS_ROOT" 2>/dev/null || true
  fi
  mkdir -p "$GSM"
  for s in train test; do
    [ -s "$GSM/$s.jsonl" ] || curl -sSfL -o "$GSM/$s.jsonl" \
      "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$s.jsonl" \
      || { say "prep FAILED gsm8k $s"; return 1; }
  done
  wc -l "$GSM"/*.jsonl
  "$PY" -c "import torch, torch_neuronx, transformers, peft, safetensors; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" \
    || { say "prep FAILED host python packages"; return 1; }
  say "prep ok"
}

step_sgl_up() {   # first start (compiles), then two restarts on the same weights; left running
  if sgl_alive && ok_file "$R/23_restart_sgl.jsonl"; then say "sgl_up skipped (running)"; return 0; fi
  "$PY" 23_restart_time.py --backend sglang_neuron --model "$BASE" --results "$R" --tag sgl \
    --port "$PORT" --name "$SRV" --repeats 2 --leave-running > "$R/logs/sgl_up.out" 2>&1
  if ok_file "$R/23_restart_sgl.jsonl" && sgl_alive; then say "sgl_up ok"; return 0; fi
  say "sgl_up FAILED: $(grep -h '"status": "fail"' "$R/23_restart_sgl.jsonl" | tail -1 | cut -c1-300)"
  tail -n 40 "$R"/server_logs/server_"$SRV"_*.log 2>/dev/null | cut -c1-300 > "$R/logs/sgl_refusal_tail.txt"
  return 1
}

step_sgl_acc() {
  sgl_alive || { say "sgl_acc skipped (no server)"; return 1; }
  # the repo's own check: 5-shot, first 100 questions (its README states about 0.45 for 0.6B)
  [ -s "$R/logs/sgl_gsm8k_repo.out" ] || docker run --rm --network host -e PORT="$PORT" \
    -e GSM8K_DATA=/models/gsm8k/test.jsonl -v "$MODELS_ROOT":/models:ro "$IMAGE" \
    /opt/sglang-neuron/examples/qwen3_0p6b/run_gsm8k.sh > "$R/logs/sgl_gsm8k_repo.out" 2>&1
  grep -iE "accuracy|invalid|latency" "$R/logs/sgl_gsm8k_repo.out" | head -5
  ok_file "$R/21_gsm8k_sgl.jsonl" || "$PY" 21_gsm8k_acc.py --base-url "http://127.0.0.1:$PORT" \
    --model "$(sgl_served)" --data "$GSM/test.jsonl" --results "$R" --tag sgl > "$R/logs/sgl_acc.out" 2>&1
  ok_file "$R/21_gsm8k_sgl.jsonl" && say "sgl_acc ok" || { say "sgl_acc FAILED"; return 1; }
}

step_sgl_probes() {
  sgl_alive || { say "sgl_probes skipped (no server)"; return 1; }
  ok_file "$R/20_rl_probes_sgl.jsonl" || "$PY" 20_rl_probes.py --base-url "http://127.0.0.1:$PORT" \
    --model "$(sgl_served)" --results "$R" --tag sgl > "$R/logs/sgl_probes.out" 2>&1
  ok_file "$R/20_rl_probes_sgl.jsonl" && say "sgl_probes ok" || { say "sgl_probes FAILED"; return 1; }
}

step_sgl_bench() {
  sgl_alive || { say "sgl_bench skipped (no server)"; return 1; }
  [ -f "$R/llmbench_sgl_tp4.done" ] && { say "sgl_bench skipped (done)"; return 0; }
  bash 22_llmbench.sh sgl_tp4 "http://127.0.0.1:$PORT" "$(sgl_served)" "$BASE" "$R" 4 \
    > "$R/logs/sgl_bench.out" 2>&1 && touch "$R/llmbench_sgl_tp4.done"
  [ -f "$R/llmbench_sgl_tp4.done" ] && say "sgl_bench ok" || { say "sgl_bench FAILED"; return 1; }
}

loop() {  # tag backend
  local tag="$1" backend="$2"
  ok_file "$R/30_rl_loop_${tag}.jsonl" && { say "loop_$tag skipped (done)"; return 0; }
  NEURON_RT_VISIBLE_CORES=8 "$PY" 30_rl_loop.py --run-tag "$tag" --backend "$backend" \
    --base-model "$BASE" --train-jsonl "$GSM/train.jsonl" --eval-jsonl "$GSM/test.jsonl" \
    --device neuron --core 8 --results "$R" --work-dir "$WORK" --ckpt-dir "$MODELS_ROOT/rl" \
    --port 30010 --steps "$STEPS_RL" > "$R/logs/loop_${tag}.out" 2>&1
  ok_file "$R/30_rl_loop_${tag}.jsonl" && say "loop_$tag ok" \
    || { say "loop_$tag FAILED: $(grep -h '"status": "fail"' "$R/30_rl_loop_${tag}.jsonl" | tail -1 | cut -c1-300)"; return 1; }
}

step_loop_sgl() {   # the server of sgl_up holds cores 0-3: stop it, the loop starts its own
  bash srv_sglang_neuron.sh stop "$SRV"
  loop sgl sglang_neuron
}
step_loop_hf() { loop hf hf; }
step_loop_vllm() { loop vllm vllm_neuron; }

step_ddp() {
  local n fails=0
  ok_file "$R/40_ddp_single.jsonl" || NEURON_RT_VISIBLE_CORES=32 "$PY" 40_ddp_probe.py --device neuron \
    --model "$BASE" --data "$GSM/train.jsonl" --results "$R" --tag single --no-ddp --per-rank-bs 8 \
    > "$R/logs/ddp_single.out" 2>&1
  ok_file "$R/40_ddp_single.jsonl" || fails=$((fails + 1))
  for n in 1 2 4 8; do
    ok_file "$R/40_ddp_ddp$n.jsonl" && continue
    NEURON_RT_VISIBLE_CORES=32-39 "$PY" -m torch.distributed.run --standalone --nproc-per-node="$n" 40_ddp_probe.py \
      --device neuron --model "$BASE" --data "$GSM/train.jsonl" --results "$R" --tag "ddp$n" \
      --global-bs 8 > "$R/logs/ddp_$n.out" 2>&1
    ok_file "$R/40_ddp_ddp$n.jsonl" || { fails=$((fails + 1)); say "ddp $n FAILED"; }
  done
  [ "$fails" = 0 ] && say "ddp ok" || { say "ddp FAILED ($fails of 5)"; return 1; }
}

step_cache_record() {   # end of session: bytes and file count of every compile-cache folder under DATA_ROOT
  # the session's own server is stopped first (a cache a container still writes has no final count); KEEP_SGL=1 keeps it and
  # the numbers are then not final
  [ "${KEEP_SGL:-0}" = 1 ] || bash srv_sglang_neuron.sh stop "$SRV" 2>/dev/null
  "$PY" 82_cache_record.py --results "$R" --tag session --root "$DATA_ROOT" > "$R/logs/cache_record.out" 2>&1
  cat "$R/logs/cache_record.out"
  ok_file "$R/82_cache_record_session.jsonl" && say "cache_record ok" || { say "cache_record FAILED"; return 1; }
}

ALL="prep sgl_up sgl_acc sgl_probes sgl_bench loop_sgl loop_hf loop_vllm ddp cache_record"
stop_loop_servers() {   # a loop killed hard can leave its container on cores 0-3
  bash srv_sglang_neuron.sh stop rl_sgl 2>/dev/null
  pkill -f "vllm serve .*--served-model-name rl_" 2>/dev/null || true
}
case " ${*:-$ALL} " in *" loop_"*) stop_loop_servers ;; esac
for s in ${*:-$ALL}; do
  in_time || { say "$s skipped (deadline)"; continue; }
  case " $ALL " in *" $s "*) ;; *) say "unknown step $s"; continue ;; esac
  say "$s start"
  "step_$s" || true
  [ "$s" = prep ] && ! [ -s "$BASE/model.safetensors" ] && { say "stop: no base model"; break; }
done
case " ${*:-$ALL} " in *" loop_"*) stop_loop_servers ;; esac
# the server of sgl_up stays up between partial runs; it is stopped when the loops begin
[ "${KEEP_SGL:-0}" = 1 ] || case " ${*:-$ALL} " in *" loop_sgl "*|*" ddp "*) bash srv_sglang_neuron.sh stop "$SRV" 2>/dev/null ;; esac
say "session end"
