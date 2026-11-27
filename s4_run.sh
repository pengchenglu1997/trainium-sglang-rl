#!/usr/bin/env bash
# Session 4 driver (2026-11-26). Usage: s4_run.sh phaseA | phaseB | b1 | b2 | b3 | pack
#   phaseA  five servers at once on five chips, each with its own compile cache: cold compile, three rollout passes of 32
#           requests (256 tokens each, ignore_eos), a device profile of 4 decode steps of TP rank 0 starting at step 300.
#             q8_rel   Qwen3-8B,   released temperature sampler            cores 0-3
#             q8_main  Qwen3-8B,   fork main (stamp fix + sampler kernel)  cores 4-7
#             q06_topk Qwen3-0.6B, released topk_topp, top-k 50 top-p 0.95 cores 8-11
#             q06_main Qwen3-0.6B, fork main (stamp fix + sampler kernel)  cores 12-15
#             q06_rel  Qwen3-0.6B, released temperature sampler            cores 16-19
#   phaseB  b1, b2, b3 one after the other (one server at a time, warm caches of phaseA, llmbench prompt 1024, conc 1/8/32):
#             b1 q06_main: temperature 0 twice, then temperature 0.8 without filters
#             b2 q06_topk: temperature 0.8 top-k 50 top-p 0.95 twice, then temperature 0.8 without filters
#             b3 q06_rel : temperature 0.8 without filters twice, then temperature 0
# No profile switches in phaseB. Marker lines "== S <UTC> ..."; every command ends with S_DONE.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
VENV="${VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export DATA_ROOT="${DATA_ROOT:-$HOME/w}"
export PY="$VENV/bin/python"; [ -x "$PY" ] || export PY=python3
export MODELS_ROOT="$DATA_ROOT/models"
M="$MODELS_ROOT"; R="$DATA_ROOT/results_s4"; PROF="$DATA_ROOT/profiles_s4"
unset SGL_PATCH_DIR SGL_ENV SGL_EXTRA SGL_SCRIPT SGL_TOP_LOGPROBS SGL_SAMPLER SGL_MOUNTS SGL_CORES SGL_CACHE
FAST="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32"
S06=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh
S8=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh
PATCH="$PWD/patches/sglang_neuron_main_2c6d86d"
mkdir -p "$R/logs" "$R/server_logs" "$PROF"
say() { echo "== S $(date -u +%H:%M:%S) $*"; }

prof_one() {  # tag model script sampler cores port patch(0|1) [extra args of s4_profile.py]
  local tag="$1" model="$2" script="$3" sampler="$4" cores="$5" port="$6" patch="$7"; shift 7
  local env="SGLANG_NEURON_PROFILE_SERVE_RANKS=0 SGLANG_NEURON_PROFILE_SERVE_START=300 SGLANG_NEURON_PROFILE_SERVE_STEPS=4 SGLANG_NEURON_PROFILE_DIR=/profiles/$tag"
  [ "${PROF_NODGE:-0}" = 1 ] || env="$env NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1"
  mkdir -p "$PROF/$tag"
  (
    export SGL_CACHE="$DATA_ROOT/sgl_cache_s4_${PROF_CACHE_TAG:-$tag}" SGL_SCRIPT="$script" SGL_SAMPLER="$sampler" SGL_EXTRA="$FAST"
    export SGL_MOUNTS="$PROF:/profiles"
    if [ "$patch" = 1 ]; then export SGL_PATCH_DIR="$PATCH"; env="$env RLKIT_SAMPLER_KERNEL=1"
    elif [ "$patch" != 0 ]; then export SGL_PATCH_DIR="$patch"; fi   # any other value: a patch folder; switches via PROF_XENV
    export SGL_ENV="$env ${PROF_XENV:-}"
    "$PY" s4_profile.py --model "$M/$model" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag "$tag" \
      --cores "$cores" --port "$port" --temperature 0.8 --ignore-eos "$@" > "$R/logs/prof_$tag.out" 2>&1
  )
  say "prof $tag exit $?: $(tail -n 1 "$R/logs/prof_$tag.out" | cut -c1-200); profile files: $(find "$PROF/$tag" -type f 2>/dev/null | wc -l)"
}

phaseA() {
  say "phaseA start; df $(df -h "$DATA_ROOT" | tail -n 1 | tr -s ' ')"
  prof_one q8_rel   Qwen3-8B   "$S8"  temperature 0-3   30101 0 &
  prof_one q8_main  Qwen3-8B   "$S8"  temperature 4-7   30102 1 &
  prof_one q06_topk Qwen3-0.6B "$S06" topk_topp   8-11  30103 0 --top-k 50 --top-p 0.95 &
  prof_one q06_main Qwen3-0.6B "$S06" temperature 12-15 30104 1 &
  prof_one q06_rel  Qwen3-0.6B "$S06" temperature 16-19 30105 0 &
  wait
  for t in q8_rel q8_main q06_topk q06_main q06_rel; do
    say "cache $t: $(du -sb "$DATA_ROOT/sgl_cache_s4_$t" 2>/dev/null | cut -f1) bytes, $(find "$DATA_ROOT/sgl_cache_s4_$t" -type f 2>/dev/null | wc -l) files"
  done
  say "phaseA end"
}

phaseA2() {  # lm_head: the integration branch with the pre-transposed LM head change, switch off and on
  local LP="$PWD/patches/sglang_neuron_lmhead_int_26213e7"
  say "phaseA2 start"
  PROF_XENV="SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0" prof_one q06_lmh_off Qwen3-0.6B "$S06" temperature 20-23 30106 "$LP" &
  PROF_XENV="SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1" prof_one q06_lmh_on  Qwen3-0.6B "$S06" temperature 24-27 30107 "$LP" &
  wait
  say "phaseA2 end"
}

phaseA3() {  # the same captures again WITHOUT NEURON_RT_ENABLE_DGE_NOTIFICATIONS (it slowed every pass), warm caches, tags *_nd
  local LP="$PWD/patches/sglang_neuron_lmhead_int_26213e7"
  export PROF_NODGE=1
  say "phaseA3 start ($*)"
  for t in "$@"; do
    case "$t" in
      q8_rel)      PROF_CACHE_TAG=q8_rel   prof_one q8_rel_nd   Qwen3-8B   "$S8"  temperature 0-3   30121 0 & ;;
      q8_main)     PROF_CACHE_TAG=q8_main  prof_one q8_main_nd  Qwen3-8B   "$S8"  temperature 4-7   30122 1 & ;;
      q06_topk)    PROF_CACHE_TAG=q06_topk prof_one q06_topk_nd Qwen3-0.6B "$S06" topk_topp   8-11  30123 0 --top-k 50 --top-p 0.95 & ;;
      q06_main)    PROF_CACHE_TAG=q06_main prof_one q06_main_nd Qwen3-0.6B "$S06" temperature 12-15 30124 1 & ;;
      q06_rel)     PROF_CACHE_TAG=q06_rel  prof_one q06_rel_nd  Qwen3-0.6B "$S06" temperature 16-19 30125 0 & ;;
      q06_lmh_off) PROF_CACHE_TAG=q06_lmh_off PROF_XENV="SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0" prof_one q06_lmh_off_nd Qwen3-0.6B "$S06" temperature 20-23 30126 "$LP" & ;;
      q06_lmh_on)  PROF_CACHE_TAG=q06_lmh_on  PROF_XENV="SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1" prof_one q06_lmh_on_nd  Qwen3-0.6B "$S06" temperature 24-27 30127 "$LP" & ;;
    esac
  done
  wait
  say "phaseA3 end"
}

phaseA4() {  # integration with the topk_topp sampler kernel, switch on and off; no DGE notifications
  local TP="$PWD/patches/sglang_neuron_int_topk"
  export PROF_NODGE=1
  say "phaseA4 start"
  PROF_XENV="RLKIT_SAMPLER_KERNEL_TOPK=1" prof_one q06_topk_kon  Qwen3-0.6B "$S06" topk_topp 8-11  30151 "$TP" --top-k 50 --top-p 0.95 &
  PROF_XENV=""                            prof_one q06_topk_koff Qwen3-0.6B "$S06" topk_topp 12-15 30152 "$TP" --top-k 50 --top-p 0.95 &
  wait
  say "phaseA4 end"
}

G="env LB_PROMPTS=1024 LB_CONC=1,8,32"
NOF="LB_TEMP=0.8 LB_EXTRA_BODY='top_k=-1 top_p=1.0 min_p=0.0'"
bench_one() {  # tag sampler patch(0|1) port bench1 bench2 bench3
  local tag="$1" sampler="$2" patch="$3" port="$4"
  (
    export SGL_CACHE="$DATA_ROOT/sgl_cache_s4_$tag" SGL_SCRIPT="$S06" SGL_SAMPLER="$sampler" SGL_EXTRA="$FAST" SGL_CORES=0-3
    if [ "$patch" = 1 ]; then export SGL_PATCH_DIR="$PATCH" SGL_ENV="RLKIT_SAMPLER_KERNEL=1"; fi
    "$PY" 76_serve_and_bench.py --backend sglang_neuron --model-dir "$M/Qwen3-0.6B" --name "pair_$tag" --port "$port" \
      --results "$R" --tag "b_$tag" --ready-timeout-s 2400 --cache-dir "$SGL_CACHE" \
      --bench "$5" --bench "$6" --bench "$7" > "$R/logs/bench_$tag.out" 2>&1
  )
  say "bench $tag exit $?: $(tail -n 1 "$R/logs/bench_$tag.out" | cut -c1-240)"
}
b1() { bench_one q06_main temperature 1 30111 \
  "$G bash 22b_llmbench_grid.sh {tag}_t0_p1 {url} {served} {tok} {results} 4" \
  "$G bash 22b_llmbench_grid.sh {tag}_t0_p2 {url} {served} {tok} {results} 4" \
  "$G $NOF bash 22b_llmbench_grid.sh {tag}_t08 {url} {served} {tok} {results} 4"; }
b2() { bench_one q06_topk topk_topp 0 30112 \
  "$G LB_TEMP=0.8 LB_EXTRA_BODY='top_k=50 top_p=0.95' bash 22b_llmbench_grid.sh {tag}_k50_p1 {url} {served} {tok} {results} 4" \
  "$G LB_TEMP=0.8 LB_EXTRA_BODY='top_k=50 top_p=0.95' bash 22b_llmbench_grid.sh {tag}_k50_p2 {url} {served} {tok} {results} 4" \
  "$G $NOF bash 22b_llmbench_grid.sh {tag}_t08 {url} {served} {tok} {results} 4"; }
bench_x() {  # tag sampler patchdir env port : three benches at the RL setting (twice) and without filters
  local tag="$1" sampler="$2" pdir="$3" xenv="$4" port="$5"
  (
    export SGL_CACHE="$DATA_ROOT/sgl_cache_s4_$tag" SGL_SCRIPT="$S06" SGL_SAMPLER="$sampler" SGL_EXTRA="$FAST" SGL_CORES=0-3
    export SGL_PATCH_DIR="$pdir" SGL_ENV="$xenv"
    "$PY" 76_serve_and_bench.py --backend sglang_neuron --model-dir "$M/Qwen3-0.6B" --name "pair_$tag" --port "$port"       --results "$R" --tag "b_$tag" --ready-timeout-s 2400 --cache-dir "$SGL_CACHE"       --bench "$G LB_TEMP=0.8 LB_EXTRA_BODY='top_k=50 top_p=0.95' bash 22b_llmbench_grid.sh {tag}_k50_p1 {url} {served} {tok} {results} 4"       --bench "$G LB_TEMP=0.8 LB_EXTRA_BODY='top_k=50 top_p=0.95' bash 22b_llmbench_grid.sh {tag}_k50_p2 {url} {served} {tok} {results} 4"       --bench "$G $NOF bash 22b_llmbench_grid.sh {tag}_t08 {url} {served} {tok} {results} 4" > "$R/logs/bench_$tag.out" 2>&1
  )
  say "bench $tag exit $?: $(tail -n 1 "$R/logs/bench_$tag.out" | cut -c1-240)"
}
b4() { bench_x q06_topk_kon  topk_topp "$PWD/patches/sglang_neuron_int_topk" "RLKIT_SAMPLER_KERNEL_TOPK=1" 30114; }
b5() { bench_x q06_topk_koff topk_topp "$PWD/patches/sglang_neuron_int_topk" "" 30115; }
b6() { bench_x q06_lmh_off temperature "$PWD/patches/sglang_neuron_lmhead_int_26213e7" "SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0" 30116; }
b7() { bench_x q06_lmh_on  temperature "$PWD/patches/sglang_neuron_lmhead_int_26213e7" "SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1" 30117; }
b3() { bench_one q06_rel temperature 0 30113 \
  "$G $NOF bash 22b_llmbench_grid.sh {tag}_t08_p1 {url} {served} {tok} {results} 4" \
  "$G $NOF bash 22b_llmbench_grid.sh {tag}_t08_p2 {url} {served} {tok} {results} 4" \
  "$G bash 22b_llmbench_grid.sh {tag}_t0 {url} {served} {tok} {results} 4"; }

case "${1:?phaseA|phaseB|b1|b2|b3|pack}" in
  phaseA) phaseA; echo "S_DONE phaseA" ;;
  phaseA2) phaseA2; echo "S_DONE phaseA2" ;;
  phaseA3) shift; phaseA3 "$@"; echo "S_DONE phaseA3" ;;
  phaseB)
    if [ -n "$(docker ps -q)" ]; then say "phaseB refused: containers running: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; echo "S_DONE phaseB status=refused"; exit 2; fi
    b1; b2; b3; echo "S_DONE phaseB" ;;
  b1|b2|b3|b4|b5|b6|b7) "$1"; echo "S_DONE $1" ;;
  phaseA4) phaseA4; echo "S_DONE phaseA4" ;;
  pack)
    cd "$DATA_ROOT" || exit 2
    tar czf s4_results.tgz results_s4 --exclude='results_s4/llmbench_*/*/requests*' 2>/dev/null
    sha256sum s4_results.tgz; ls -la s4_results.tgz; du -sh profiles_s4; echo "S_DONE pack" ;;
  *) echo "usage: s4_run.sh phaseA|phaseB|b1|b2|b3|pack"; exit 2 ;;
esac
