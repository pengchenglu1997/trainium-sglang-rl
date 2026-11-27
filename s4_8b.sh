#!/usr/bin/env bash
# Session 4, Qwen3-8B: llmbench prompt 1024, concurrency 1/8/32, temperature 0, two passes each.
#   vllm    the vllm-neuron server already running on port 30121 (cores 28-31, started by an earlier step of the same session)
#   sgl     SGLang-Neuron released temperature sampler, then fork main with the sampler kernel, cores 4-7, warm caches
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
VENV="${VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export DATA_ROOT="${DATA_ROOT:-$HOME/w}"; export MODELS_ROOT="$DATA_ROOT/models"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
M="$MODELS_ROOT"; R="$DATA_ROOT/results_s4"
say() { echo "== S $(date -u +%H:%M:%S) $*"; }
FAST="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32"
G="env LB_PROMPTS=1024 LB_CONC=1,8,32"
case "${1:?vllm|sgl}" in
  vllm)
    url=http://127.0.0.1:30121
    served=$(curl -s "$url/v1/models" | "$PY" -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
    say "vllm served name: $served"
    for p in p1 p2; do
      LB_PROMPTS=1024 LB_CONC=1,8,32 bash 22b_llmbench_grid.sh "b_q8_vllm_$p" "$url" "$served" "$M/Qwen3-8B" "$R" 4 2>&1 | tail -n 1
    done
    echo "S_DONE vllm" ;;
  sgl)
    for v in rel main; do
      (
        export SGL_CACHE="$DATA_ROOT/sgl_cache_s4_q8_$v" SGL_SCRIPT=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh
        export SGL_SAMPLER=temperature SGL_EXTRA="$FAST" SGL_CORES=4-7
        if [ "$v" = main ]; then export SGL_PATCH_DIR="$PWD/patches/sglang_neuron_main_2c6d86d" SGL_ENV="RLKIT_SAMPLER_KERNEL=1"; fi
        "$PY" 76_serve_and_bench.py --backend sglang_neuron --model-dir "$M/Qwen3-8B" --name "pair_q8_$v" --port 3017$([ "$v" = rel ] && echo 1 || echo 2) \
          --results "$R" --tag "b_q8_$v" --ready-timeout-s 2400 --cache-dir "$SGL_CACHE" \
          --bench "$G bash 22b_llmbench_grid.sh {tag}_t0_p1 {url} {served} {tok} {results} 4" \
          --bench "$G bash 22b_llmbench_grid.sh {tag}_t0_p2 {url} {served} {tok} {results} 4" > "$R/logs/bench_q8_$v.out" 2>&1
      )
      say "bench q8_$v exit $?: $(tail -n 1 "$R/logs/bench_q8_$v.out" | cut -c1-240)"
    done
    echo "S_DONE sgl" ;;
esac
