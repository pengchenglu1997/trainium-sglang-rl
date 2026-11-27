#!/usr/bin/env bash
# dp_ slot (2026-11-27): replica layout (--dp-size) on top of the "all-on" configuration. Copy of the bench mode of
# ~/w/sglang_rl/s4_x.sh with: results in ~/w/dp/results, container dp_<tag>, own cache ~/w/sgl_cache_dp_<cachetag>,
# and an extra-flags argument appended AFTER s4_x.sh's FAST flags (the last occurrence of a flag wins).
#   dp_x.sh start <tag> <cachetag> <06|8> <cores> <port> "<extra flags>"        start, wait ready, stop (no bench)
#   dp_x.sh bench <tag> <cachetag> <06|8> <cores> <port> "<extra flags>"        t0 p1, t0 p2, t08, then cc (20 prompts t0)
# Fixed: sampler temperature, patch folder sglang_neuron_int_topk, env RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1
set -u
KIT="$HOME/w/sglang_rl"
cd "$KIT" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export DATA_ROOT="$HOME/w" MODELS_ROOT="$HOME/w/models"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
M="$MODELS_ROOT"; R="$HOME/w/dp/results"; mkdir -p "$R/logs"
unset SGL_PATCH_DIR SGL_ENV SGL_EXTRA SGL_SCRIPT SGL_TOP_LOGPROBS SGL_SAMPLER SGL_MOUNTS SGL_CORES SGL_CACHE
FAST="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32"
say() { echo "== D $(date -u +%H:%M:%S) $*"; }
what="${1:?start|bench}"; tag="${2:?tag}"; ctag="${3:?cachetag}"; size="${4:?06|8}"; cores="${5:?cores}"; port="${6:?port}"
xflags="${7-}"
case "$size" in
  06) model=Qwen3-0.6B; script=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh ;;
  8)  model=Qwen3-8B;   script=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh ;;
  *) echo "size must be 06 or 8"; exit 2 ;;
esac
export SGL_CACHE="$DATA_ROOT/sgl_cache_dp_$ctag" SGL_SCRIPT="$script" SGL_SAMPLER=temperature SGL_CORES="$cores"
export SGL_EXTRA="$FAST${xflags:+ $xflags}"
export SGL_PATCH_DIR="$KIT/patches/sglang_neuron_int_topk"
export SGL_ENV="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
snap() { { echo "--- $(date -u +%FT%TZ) $1 $tag"; docker ps --format '{{.Names}}\t{{.Status}}'; } >> "$R/logs/docker_ps.log"; }
G="env LB_PROMPTS=1024 LB_CONC=1,8,32"
NOF="LB_TEMP=0.8 LB_EXTRA_BODY='top_k=-1 top_p=1.0 min_p=0.0'"
benches=()
if [ "$what" = bench ]; then
  benches+=(--bench "$G bash 22b_llmbench_grid.sh {tag}_t0_p1 {url} {served} {tok} {results} 4")
  benches+=(--bench "$G bash 22b_llmbench_grid.sh {tag}_t0_p2 {url} {served} {tok} {results} 4")
  benches+=(--bench "$G $NOF bash 22b_llmbench_grid.sh {tag}_t08 {url} {served} {tok} {results} 4")
  benches+=(--bench "$PY $HOME/w/dp/dp_cc.py {url} {served} {results}/cc_out_{tag}_t0.json")
elif [ "$what" != start ]; then echo "start|bench"; exit 2; fi
say "$what $tag cache $SGL_CACHE cores $cores port $port extra: $SGL_EXTRA"
snap "before"
"$PY" 76_serve_and_bench.py --backend sglang_neuron --model-dir "$M/$model" --name "dp_$tag" --port "$port" \
  --results "$R" --tag "d_$tag" --ready-timeout-s 2400 --cache-dir "$SGL_CACHE" \
  ${benches[@]+"${benches[@]}"} > "$R/logs/${what}_$tag.out" 2>&1
rc=$?
snap "after"
say "$what $tag exit $rc: $(tail -n 1 "$R/logs/${what}_$tag.out" | cut -c1-240)"
echo "D_DONE $what $tag"
