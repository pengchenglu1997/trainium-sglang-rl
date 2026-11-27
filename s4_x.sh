#!/usr/bin/env bash
# Session 4, generic steps (2026-11-26). Same server flags, load and bench as s4_run.sh.
#   s4_x.sh prof  <tag> <06|8> <sampler> <cores> <port> <patch dir or -> "<K=V ...>" [s4_profile.py args]
#       cold or warm start with its own cache sgl_cache_s4_<tag>, three rollout passes, device profile of 4 steps from
#       step 300 (no DGE notifications)
#   s4_x.sh bench <tag> <06|8> <sampler> <cores> <port> <patch dir or -> "<K=V ...>" <t|k>
#       llmbench prompt 1024, conc 1/8/32. t: temperature 0 twice, then temperature 0.8 without filters.
#       k: temperature 0.8 top-k 50 top-p 0.95 twice, then temperature 0.8 without filters.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export DATA_ROOT="$HOME/w" MODELS_ROOT="$HOME/w/models"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
M="$MODELS_ROOT"; R="$DATA_ROOT/results_s4"; PROF="$DATA_ROOT/profiles_s4"
unset SGL_PATCH_DIR SGL_ENV SGL_EXTRA SGL_SCRIPT SGL_TOP_LOGPROBS SGL_SAMPLER SGL_MOUNTS SGL_CORES SGL_CACHE
FAST="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32"
say() { echo "== S $(date -u +%H:%M:%S) $*"; }
what="${1:?prof|bench}"; tag="${2:?tag}"; size="${3:?06|8}"; sampler="${4:?sampler}"; cores="${5:?cores}"; port="${6:?port}"
pdir="${7:?patch dir or -}"; xenv="${8-}"
case "$size" in
  06) model=Qwen3-0.6B; script=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh ;;
  8)  model=Qwen3-8B;   script=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh ;;
  *) echo "size must be 06 or 8"; exit 2 ;;
esac
export SGL_CACHE="$DATA_ROOT/sgl_cache_s4_$tag" SGL_SCRIPT="$script" SGL_SAMPLER="$sampler" SGL_EXTRA="$FAST" SGL_CORES="$cores"
[ "$pdir" = - ] || export SGL_PATCH_DIR="$pdir"
G="env LB_PROMPTS=1024 LB_CONC=1,8,32"
NOF="LB_TEMP=0.8 LB_EXTRA_BODY='top_k=-1 top_p=1.0 min_p=0.0'"
K50="LB_TEMP=0.8 LB_EXTRA_BODY='top_k=50 top_p=0.95'"
case "$what" in
  prof)
    shift 8
    mkdir -p "$PROF/$tag"
    export SGL_MOUNTS="$PROF:/profiles"
    export SGL_ENV="SGLANG_NEURON_PROFILE_SERVE_RANKS=0 SGLANG_NEURON_PROFILE_SERVE_START=300 SGLANG_NEURON_PROFILE_SERVE_STEPS=4 SGLANG_NEURON_PROFILE_DIR=/profiles/$tag $xenv"
    "$PY" s4_profile.py --model "$M/$model" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag "$tag" \
      --cores "$cores" --port "$port" --temperature 0.8 --ignore-eos "$@" > "$R/logs/prof_$tag.out" 2>&1
    say "prof $tag exit $?: $(tail -n 1 "$R/logs/prof_$tag.out" | cut -c1-200); profile files: $(find "$PROF/$tag" -type f 2>/dev/null | wc -l)"
    echo "S_DONE prof $tag" ;;
  bench)
    mode="${9:?t|k}"
    export SGL_ENV="$xenv"
    if [ "$mode" = k ]; then
      b1="$G $K50 bash 22b_llmbench_grid.sh {tag}_k50_p1 {url} {served} {tok} {results} 4"
      b2="$G $K50 bash 22b_llmbench_grid.sh {tag}_k50_p2 {url} {served} {tok} {results} 4"
    else
      b1="$G bash 22b_llmbench_grid.sh {tag}_t0_p1 {url} {served} {tok} {results} 4"
      b2="$G bash 22b_llmbench_grid.sh {tag}_t0_p2 {url} {served} {tok} {results} 4"
    fi
    b3="$G $NOF bash 22b_llmbench_grid.sh {tag}_t08 {url} {served} {tok} {results} 4"
    "$PY" 76_serve_and_bench.py --backend sglang_neuron --model-dir "$M/$model" --name "pair_$tag" --port "$port" \
      --results "$R" --tag "b_$tag" --ready-timeout-s 2400 --cache-dir "$SGL_CACHE" \
      --bench "$b1" --bench "$b2" --bench "$b3" > "$R/logs/bench_$tag.out" 2>&1
    say "bench $tag exit $?: $(tail -n 1 "$R/logs/bench_$tag.out" | cut -c1-240)"
    echo "S_DONE bench $tag" ;;
  *) echo "prof|bench"; exit 2 ;;
esac
