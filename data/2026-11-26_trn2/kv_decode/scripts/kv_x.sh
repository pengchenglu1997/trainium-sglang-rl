#!/usr/bin/env bash
# kv slot copy of s4_x.sh (2026-11-26). Differences from s4_x.sh: results under ~/w/kv/results, profiles under
# ~/w/kv/profiles, cache ~/w/sgl_cache_kv_<tag>, container names kv_pair_<tag> / kv_prof_<tag> (s4_profile.py copy),
# extra launch flags appended from KV_EXTRA, and bench = two temperature-0 llmbench passes + a 20-prompt
# temperature-0 token dump (kv_tokdump.py) instead of the t08 pass.
#   kv_x.sh prof  <tag> <06|8> <sampler> <cores> <port> <patch dir or -> "<K=V ...>"
#   kv_x.sh bench <tag> <06|8> <sampler> <cores> <port> <patch dir or -> "<K=V ...>"
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export DATA_ROOT="$HOME/w" MODELS_ROOT="$HOME/w/models"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
M="$MODELS_ROOT"; R="$HOME/w/kv/results"; PROF="$HOME/w/kv/profiles"
mkdir -p "$R/logs" "$PROF"
unset SGL_PATCH_DIR SGL_ENV SGL_EXTRA SGL_SCRIPT SGL_TOP_LOGPROBS SGL_SAMPLER SGL_MOUNTS SGL_CORES SGL_CACHE
FAST="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32 ${KV_EXTRA:-}"
say() { echo "== S $(date -u +%H:%M:%S) $*"; }
what="${1:?prof|bench}"; tag="${2:?tag}"; size="${3:?06|8}"; sampler="${4:?sampler}"; cores="${5:?cores}"; port="${6:?port}"
pdir="${7:?patch dir or -}"; xenv="${8-}"
case "$size" in
  06) model=Qwen3-0.6B; script=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh ;;
  8)  model=Qwen3-8B;   script=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh ;;
  *) echo "size must be 06 or 8"; exit 2 ;;
esac
export SGL_CACHE="$HOME/w/sgl_cache_kv_$tag" SGL_SCRIPT="$script" SGL_SAMPLER="$sampler" SGL_EXTRA="$FAST" SGL_CORES="$cores"
[ "$pdir" = - ] || export SGL_PATCH_DIR="$pdir"
G="env LB_PROMPTS=1024 LB_CONC=1,8,32"
case "$what" in
  prof)
    mkdir -p "$PROF/$tag"
    export SGL_MOUNTS="$PROF:/profiles"
    export SGL_ENV="SGLANG_NEURON_PROFILE_SERVE_RANKS=0 SGLANG_NEURON_PROFILE_SERVE_START=300 SGLANG_NEURON_PROFILE_SERVE_STEPS=4 SGLANG_NEURON_PROFILE_DIR=/profiles/$tag $xenv"
    "$PY" s4_profile.py --model "$M/$model" --train-jsonl "$M/gsm8k/train.jsonl" --results "$R" --tag "$tag" \
      --cores "$cores" --port "$port" --temperature 0.8 --ignore-eos > "$R/logs/prof_$tag.out" 2>&1
    say "prof $tag exit $?: $(tail -n 1 "$R/logs/prof_$tag.out" | cut -c1-200); profile files: $(find "$PROF/$tag" -type f 2>/dev/null | wc -l)"
    echo "S_DONE prof $tag" ;;
  bench)
    export SGL_ENV="$xenv"
    b1="$G bash 22b_llmbench_grid.sh {tag}_t0_p1 {url} {served} {tok} {results} 4"
    b2="$G bash 22b_llmbench_grid.sh {tag}_t0_p2 {url} {served} {tok} {results} 4"
    b3="$PY kv_tokdump.py {url} {served} $M/gsm8k/train.jsonl {results}/tokdump_{tag}.json"
    "$PY" 76_serve_and_bench.py --backend sglang_neuron --model-dir "$M/$model" --name "kv_pair_$tag" --port "$port" \
      --results "$R" --tag "b_$tag" --ready-timeout-s 2400 --cache-dir "$SGL_CACHE" \
      --bench "$b1" --bench "$b2" --bench "$b3" > "$R/logs/bench_$tag.out" 2>&1
    say "bench $tag exit $?: $(tail -n 1 "$R/logs/bench_$tag.out" | cut -c1-240)"
    echo "S_DONE bench $tag" ;;
  *) echo "prof|bench"; exit 2 ;;
esac
