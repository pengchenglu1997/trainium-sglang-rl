#!/usr/bin/env bash
# trn2 session of 2026-11-21 (night): four checks, each a separate command so they can run side by side on their own
# cores. Patch: patches/sglang_neuron_v0.1.0_inplace of this kit (cache folder sgl_cache_v0.1.0_night1121).
#   run_night_1121.sh prep                 image, models (Qwen3-0.6B, Qwen3-8B, GLM-4-9B-0414, DeepSeek-V2-Lite), GSM8K
#   run_night_1121.sh gen glm4|dsv2        check 1: in-place update on a second model family, against a fresh start
#   run_night_1121.sh sampler              check 2: topk_topp sampler on Qwen3-0.6B
#   run_night_1121.sh q8                   check 3: Qwen3-8B update, host graph built each time, then kept
#   run_night_1121.sh profile              check 4: system profile of one decode pass, Qwen3-0.6B
# Cores (logical): glm4 0-3, dsv2 4-7, q8 8-11 then 20-23, sampler 12-15, profile 16-19.
# Marker lines "== N <time> ..."; each command ends with N_DONE <what>.
set -u
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
export DATA_ROOT=$HOME/w PY=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python
export MODELS_ROOT=$DATA_ROOT/models SGL_CACHE=$DATA_ROOT/sgl_cache_v0.1.0_night1121
cd "$(dirname "$0")"
export SGL_PATCH_DIR="$PWD/patches/sglang_neuron_v0.1.0_inplace"
IMAGE="public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04"
R=$DATA_ROOT/results_night; M=$MODELS_ROOT
SMALL="--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768 --cuda-graph-bs-decode 32"
mkdir -p "$R/logs" "$M" "$SGL_CACHE"
say() { echo "== N $(date -u +%H:%M:%S) $*"; }
ok_file() { [ -f "$1" ] && [ "$(tail -n 1 "$1")" = '{"done": true}' ] && ! grep -q '"status": "fail"' "$1"; }
fetch() {  # hub id, folder
  if ls "$M/$2"/*.safetensors > /dev/null 2>&1 && [ -f "$M/$2/.fetch_ok" ]; then say "model $2 present"; return 0; fi
  docker run --rm -v "$M":/models "$IMAGE" hf download "$1" --local-dir "/models/$2" > "$R/logs/hf_$2.out" 2>&1 \
    || { say "model $2 FAILED: $(tail -n 2 "$R/logs/hf_$2.out" | cut -c1-200)"; return 1; }
  sudo chown -R "$(id -u):$(id -g)" "$M/$2" 2>/dev/null || true
  touch "$M/$2/.fetch_ok"
  say "model $2 ok $(du -sh "$M/$2" | cut -f1)"
}
inplace() {  # tag, model folder, cores, port, script, extra flags, more args of 48...
  local tag="$1" model="$2" cores="$3" port="$4" script="$5" extra="$6"; shift 6
  ok_file "$R/48_inplace_$tag.jsonl" && { say "$tag skipped (done)"; return 0; }
  rm -f "$R/48_inplace_$tag.jsonl"
  [ -d "$M/${model}_alt" ] || { say "alt for $model is missing (run: prep)"; return 1; }
  SGL_CORES="$cores" SGL_SCRIPT="$script" SGL_EXTRA="$extra" "$PY" 48_inplace_server_test.py \
    --backend sglang_neuron_inplace --base "$M/$model" --alt "$M/${model}_alt" --results "$R" --tag "$tag" \
    --port "$port" "$@" > "$R/logs/inplace_$tag.out" 2>&1
  local rc=$?
  say "$tag exit $rc: $(grep -h '"event": "check"\|"status": "fail"' "$R/48_inplace_$tag.jsonl" 2>/dev/null | cut -c1-300 | tr '\n' ' ')"
  say "$tag updates: $(grep -h '"event": "update"' "$R/48_inplace_$tag.jsonl" 2>/dev/null | grep -o '"seconds": [0-9.]*' | cut -d' ' -f2 | tr '\n' ' ')"
  ok_file "$R/48_inplace_$tag.jsonl"
}
WHAT="${1:?prep|gen|sampler|q8|profile|boundary|adapter8|devprofile}"
case "$WHAT" in
  prep)
    nproc; free -g | sed -n 2p; df -h "$DATA_ROOT" | tail -1; ls /dev/neuron* | wc -l
    docker pull "$IMAGE" > "$R/logs/docker_pull.out" 2>&1 || { say "prep FAILED docker pull"; echo N_DONE prep; exit 1; }
    say "image ok"
    fetch Qwen/Qwen3-0.6B Qwen3-0.6B &
    fetch Qwen/Qwen3-8B Qwen3-8B &
    fetch zai-org/GLM-4-9B-0414 GLM-4-9B-0414 &
    fetch deepseek-ai/DeepSeek-V2-Lite DeepSeek-V2-Lite &
    wait
    mkdir -p "$M/gsm8k"
    for s in train test; do
      [ -s "$M/gsm8k/$s.jsonl" ] || curl -sSfL -o "$M/gsm8k/$s.jsonl" \
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$s.jsonl" \
        || say "gsm8k $s FAILED"
    done
    wc -l "$M"/gsm8k/*.jsonl | tail -1
    for m in Qwen3-8B GLM-4-9B-0414 DeepSeek-V2-Lite; do
      "$PY" mk_alt.py --base "$M/$m" --alt "$M/${m}_alt" > "$R/logs/alt_$m.out" 2>&1 &
    done
    wait
    for m in Qwen3-8B GLM-4-9B-0414 DeepSeek-V2-Lite; do say "$(tail -n 1 "$R/logs/alt_$m.out" | cut -c1-200)"; done
    which neuron-explorer neuron-profile 2>&1 | head -3
    "$PY" selftest_fake_server.py | tail -1
    ;;
  gen)
    case "${2:?glm4|dsv2}" in
      glm4)
        # dense, on the shared transformer; reduced shapes first, the example's own flags if that start is refused
        inplace glm4 GLM-4-9B-0414 0-3 30005 /opt/sglang-neuron/examples/glm4_9b/run_server.sh "$SMALL" \
          || inplace glm4_full GLM-4-9B-0414 0-3 30005 /opt/sglang-neuron/examples/glm4_9b/run_server.sh ""
        ;;
      dsv2)
        # its own graph (MLA, MoE); the recipe wants context a multiple of 512 and chunked prefill = context
        inplace dsv2 DeepSeek-V2-Lite 4-7 30006 /opt/sglang-neuron/examples/deepseek_v2_lite/run_server.sh "$SMALL" \
          || inplace dsv2_full DeepSeek-V2-Lite 4-7 30006 /opt/sglang-neuron/examples/deepseek_v2_lite/run_server.sh ""
        ;;
    esac
    ;;
  sampler)
    ok_file "$R/58_sampler_s1.jsonl" || SGL_EXTRA="$SMALL" SGL_ENV="RLKIT_DECODE_LADDER=0" "$PY" 58_sampler_probe.py \
      --model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag s1 --cores 12-15 \
      > "$R/logs/sampler.out" 2>&1
    say "sampler: $(grep -c '"event"' "$R/58_sampler_s1.jsonl") lines; $(grep -h '"event": "accept"' "$R/58_sampler_s1.jsonl" | grep -o '"mode": "[a-z_]*"\|"status": [0-9]*' | tr '\n' ' ' | cut -c1-400)"
    ;;
  q8)
    # A: as the patch was (host graph built at every update). B: host graph kept between updates. Six updates each
    # (two of the test, four extra); B has the fresh-start comparison, A does not repeat it.
    inplace q8_built Qwen3-8B 8-11 30007 /opt/sglang-neuron/examples/qwen3_8b/run_server.sh "$SMALL" \
      --extra-updates 4 --skip-fresh &
    SGL_ENV="RLKIT_KEEP_STAGED=1" inplace q8_kept Qwen3-8B 20-23 30009 \
      /opt/sglang-neuron/examples/qwen3_8b/run_server.sh "$SMALL" --extra-updates 4 &
    wait
    for t in q8_built q8_kept; do
      say "$t stage lines: $(grep -h 'in-place weight update from' "$R"/server_logs/server_srv_${t}_*.log 2>/dev/null | grep -o 'stage [0-9.]*s ([^)]*), copy [0-9.]*s' | sort | uniq -c | head -12 | tr '\n' ';')"
    done
    ;;
  profile)
    ok_file "$R/59_decode_profile_p1.jsonl" || SGL_EXTRA="$SMALL" SGL_ENV="RLKIT_DECODE_LADDER=0" "$PY" \
      59_decode_profile.py --model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag p1 \
      --cores 16-19 > "$R/logs/profile.out" 2>&1
    grep -h '"event": "pass"\|profile' "$R/59_decode_profile_p1.jsonl" | cut -c1-260
    sudo find "$SGL_CACHE/profile_p1" -maxdepth 3 | head -20
    ;;
  boundary)   # check 2b: where the top-p filter cuts, on prompts with a flatter first-token distribution
    SGL_EXTRA="$SMALL" SGL_ENV="RLKIT_DECODE_LADDER=0" "$PY" 58_sampler_probe.py --model "$M/Qwen3-0.6B" \
      --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag b1 --cores 12-15 --modes topk_topp --no-speed \
      > "$R/logs/boundary.out" 2>&1
    grep -h '"event": "boundary"' "$R/58_sampler_b1.jsonl" | cut -c1-420
    ;;
  adapter8)   # check 3b: adapter-only update on Qwen3-8B (what a LoRA loop sends), against a restart on the merge
    SGL_CORES=8-11 SGL_SCRIPT=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh SGL_EXTRA="$SMALL" \
      "$PY" 50_adapter_server_test.py --base "$M/Qwen3-8B" --work "$M/adapter_test_8b" --results "$R" --tag q8 \
      > "$R/logs/adapter8.out" 2>&1
    grep -h '"event": "update"\|"event": "check"\|"status": "fail"' "$R/50_adapter_q8.jsonl" | cut -c1-300
    ;;
  devprofile) # check 4b: device profile of 4 decode steps of TP rank 0 (launch-time window of the profiling guide)
    SGL_EXTRA="$SMALL" SGL_ENV="RLKIT_DECODE_LADDER=0 SGLANG_NEURON_PROFILE_SERVE_RANKS=0 SGLANG_NEURON_PROFILE_DIR=/cache/devprof_p2" \
      "$PY" 59_decode_profile.py --model "$M/Qwen3-0.6B" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" \
      --tag p2 --cores 16-19 --port 30091 > "$R/logs/devprofile.out" 2>&1
    cut -c1-240 "$R/59_decode_profile_p2.jsonl"
    sudo find "$SGL_CACHE/devprof_p2" -maxdepth 3 | head -20
    ;;
  *) echo "usage: run_night_1121.sh prep|gen glm4|gen dsv2|sampler|q8|profile|boundary|adapter8|devprofile"; exit 2 ;;
esac
echo "N_DONE $*"
