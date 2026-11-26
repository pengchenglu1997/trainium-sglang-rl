#!/usr/bin/env bash
# R4, the baseline pair: vllm-neuron 0.29.0.1.2.0 and SGLang-Neuron v0.1.0 as released, one model, one client.
#   76_baseline_pair.sh <model folder name under $MODELS_ROOT> <tag> [--arms "vllm sgl"] [--ready-vllm S] [--ready-sgl S]
# Example, session driver (host venv active or PY set; docker and /dev/neuron* present):
#   bash 76_baseline_pair.sh Qwen3-0.6B q06
#   READY_VLLM_S=1200 LB_CONC=1,8,32 LB_PROMPTS=1024 bash 76_baseline_pair.sh Qwen3-8B q8     (8B: vllm-neuron capped at 20 min)
# One server at a time: the arms run one after the other (vllm, then sgl), each started, pre-checked, benchmarked
# (grid pass, then the repeat pass) and stopped by 76_serve_and_bench.py. Then 77_pair_table.py writes the table.
#
# Arm 1, vllm-neuron (srv_vllm_neuron_docker.sh, backend vllm_neuron_docker, container pair_vllm_<tag>):
#   untie_ckpt.py first (vllm-neuron's Qwen3 loader needs lm_head.weight; Qwen3-0.6B is tied, Qwen3-8B is not),
#   TP4, --max-model-len 4096, --max-num-seqs 32, compile timeout 1800 (VLLM_COMPILE_TIMEOUT), devices VLLM_DEVICES
#   (default 16,17,18,19 = NEURON_VISIBLE_DEVICES, a LIST of logical cores, one per TP rank; the launcher's header: the
#   runtime refuses NEURON_RT_VISIBLE_CORES for vLLM and "4" resolved to one core). The launcher also passes
#   --no-enable-prefix-caching, which the parity repository's script does not (run_inference.sh line 77); llmbench varies the
#   prompt per request, so it should not matter, but it is a difference.
# Arm 2, SGLang-Neuron as released (srv_sglang_neuron.sh, backend sglang_neuron, container pair_sgl_<tag>): no
#   SGL_PATCH_DIR, SGL_ENV, SGL_EXTRA, SGL_TOP_LOGPROBS, SGL_SAMPLER (the launcher's default sampler applies); the
#   image's own example script with its own flags (TP4, context 4096, 64 running requests, read in the repo at tag
#   v0.1.0: examples/qwen3_0p6b/run_server.sh and examples/qwen3_8b/run_server.sh), cores SGL_CORES (default 0-3 =
#   NEURON_RT_VISIBLE_CORES, a range), its own compile cache $DATA_ROOT/sgl_cache_v0.1.0_release. The two arms use
#   disjoint core sets, so one arm's container cannot hold the other's cores; they still never run together.
# Environment (defaults): DATA_ROOT ($HOME/w)  MODELS_ROOT ($DATA_ROOT/models)  R (results, $DATA_ROOT/results_pair)
#   PY (host python with torch and safetensors; default the Neuron venv /opt/aws_neuronx_venv_torch_neuronx_2_14_0)
#   VLLM_DEVICES (16,17,18,19)  SGL_CORES (0-3)  VLLM_CACHE ($DATA_ROOT/vllm_cache_0.29.0.1.2.0)
#   SGL_CACHE_PAIR ($DATA_ROOT/sgl_cache_v0.1.0_release)  PORT_VLLM (30071)  PORT_SGL (30072)
#   READY_VLLM_S (1800)  READY_SGL_S (3000): give up on that arm when its server is not ready after this many seconds
#   (SGLang-Neuron's example flags took 1224 s to the first start at 0.6B on 2026-11-20; 8B has no record)
#   BENCH_TIMEOUT_S (7200) per bench command   ARMS ("vllm sgl")   LB_CONC LB_PROMPTS LB_OUT LB_REQ LB_WARM LB_MAXLEN
#   (the grid, see 22b_llmbench_grid.sh; the repeat pass overrides LB_PROMPTS with 1024 and keeps the rest)
# Files in $R: 76_serve_<tag>_vllm.jsonl and 76_serve_<tag>_sgl.jsonl (one per arm), llmbench_<tag>_vllm.csv,
#   llmbench_<tag>_vllm_rep.csv, llmbench_<tag>_sgl.csv, llmbench_<tag>_sgl_rep.csv, 77_pair_<tag>.md, 76_pair_<tag>.jsonl
#   (this script's own result, JSON lines ending with {"done": true}), server_logs/, logs/.
#   A failed pre-check of an arm (76_serve_and_bench.py prints PRECHECK_FAILED: 64 tokens asked with ignore_eos, not
#   64 returned) does not change the exit codes or the P_DONE line: the bench runs, the pair counts as done, and the line
#   in 76_pair_<tag>.jsonl carries "precheck_failed": true and "precheck_failed_arms": "vllm sgl"; this script prints
#   "PRECHECK FAILED 76 ..." before its last line; 77_pair_table.py marks every row of that arm.
# Marker lines "== P <time> ..."; last line: "P_DONE 76 <model> <tag> vllm=<code> sgl=<code>" when every requested arm
# ended well, else "P_FAILED 76 ..." (exit 1). A code: 0 done, 2 server not ready, 3 other error, 9 not run.
set -u
cd "$(dirname "$0")"
model="${1:?model folder name under MODELS_ROOT}"; tag="${2:?tag}"; shift 2
ARMS="${ARMS:-vllm sgl}"; READY_VLLM_S="${READY_VLLM_S:-1800}"; READY_SGL_S="${READY_SGL_S:-3000}"
while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="${2:?}"; shift 2 ;;
    --ready-vllm) READY_VLLM_S="${2:?}"; shift 2 ;;
    --ready-sgl) READY_SGL_S="${2:?}"; shift 2 ;;
    *) echo "P_FAILED 76 unknown argument $1"; exit 2 ;;
  esac
done
case "$tag" in *[!A-Za-z0-9.-]*|"") echo "P_FAILED 76 tag: letters, digits, . - only"; exit 2 ;; esac
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
PY="${PY:-$VENV/bin/python}"; [ -x "$PY" ] || PY=python3
export DATA_ROOT="${DATA_ROOT:-$HOME/w}"
export MODELS_ROOT="${MODELS_ROOT:-$DATA_ROOT/models}"
R="${R:-$DATA_ROOT/results_pair}"; PORT_VLLM="${PORT_VLLM:-30071}"; PORT_SGL="${PORT_SGL:-30072}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-7200}"
mkdir -p "$R/logs"
say() { echo "== P $(date -u +%H:%M:%S) $*"; }
src="$MODELS_ROOT/$model"
[ -f "$src/config.json" ] || { echo "P_FAILED 76 no $src/config.json"; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "P_FAILED 76 no docker"; exit 2; }
BENCH1="bash 22b_llmbench_grid.sh {tag} {url} {served} {tok} {results} 4"
BENCH2="env LB_PROMPTS=1024 bash 22b_llmbench_grid.sh {tag}_rep {url} {served} {tok} {results} 4"
rc_vllm=9; rc_sgl=9; pc_failed=""
case "$model" in
  Qwen3-0.6B) SGL_SCRIPT_PAIR=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh ;;
  Qwen3-8B) SGL_SCRIPT_PAIR=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh ;;
  *) SGL_SCRIPT_PAIR="${SGL_SCRIPT_PAIR:-}" ;;
esac
say "pair $model $tag arms [$ARMS] ready caps vllm ${READY_VLLM_S}s sgl ${READY_SGL_S}s grid conc=${LB_CONC:-default} prompts=${LB_PROMPTS:-default}"

for arm in $ARMS; do
  case "$arm" in
    vllm)
      vname="pair_vllm_${tag//./_}"
      say "vllm: untie check on $src"
      if ! served_dir=$("$PY" untie_ckpt.py "$src" 2> "$R/logs/76_${tag}_untie.err" | tail -n 1) || [ -z "$served_dir" ]; then
        say "vllm: untie FAILED: $(tail -n 2 "$R/logs/76_${tag}_untie.err" | tr '\n' ' ' | cut -c1-300)"; rc_vllm=3; continue
      fi
      say "vllm: serving folder $served_dir ($(tr '\n' ' ' < "$R/logs/76_${tag}_untie.err" | cut -c1-200))"
      [ "$(bash srv_sglang_neuron.sh alive "pair_sgl_${tag//./_}")" = true ] && { say "vllm: the sgl container is still running, not starting"; rc_vllm=3; continue; }
      unset VLLM_EXTRA
      export VLLM_TP=4 VLLM_MAX_SEQS=32 VLLM_MAX_LEN=4096 VLLM_COMPILE_TIMEOUT=1800
      export VLLM_DEVICES="${VLLM_DEVICES:-16,17,18,19}"
      export VLLM_CACHE="${VLLM_CACHE:-$DATA_ROOT/vllm_cache_0.29.0.1.2.0}"
      "$PY" 76_serve_and_bench.py --backend vllm_neuron_docker --model-dir "$served_dir" --tokenizer "$src" \
        --name "$vname" --port "$PORT_VLLM" --results "$R" --tag "${tag}_vllm" --ready-timeout-s "$READY_VLLM_S" \
        --cache-dir "$VLLM_CACHE" --bench-timeout-s "$BENCH_TIMEOUT_S" --bench "$BENCH1" --bench "$BENCH2" \
        > "$R/logs/76_${tag}_vllm.out" 2>&1
      rc_vllm=$?
      say "vllm: exit $rc_vllm: $(tail -n 1 "$R/logs/76_${tag}_vllm.out" | cut -c1-300)"
      if grep -q '^PRECHECK_FAILED' "$R/logs/76_${tag}_vllm.out" 2>/dev/null; then
        pc_failed="$pc_failed vllm"; say "vllm: PRECHECK FAILED (the bench ran anyway; the arm still counts as done): $(grep -m1 '^PRECHECK_FAILED' "$R/logs/76_${tag}_vllm.out" | cut -c1-200)"
      fi
      ;;
    sgl)
      sname="pair_sgl_${tag//./_}"
      [ -n "$SGL_SCRIPT_PAIR" ] || { say "sgl: no example script known for $model: set SGL_SCRIPT_PAIR"; rc_sgl=3; continue; }
      [ "$(bash srv_vllm_neuron_docker.sh alive "pair_vllm_${tag//./_}")" = true ] && { say "sgl: the vllm container is still running, not starting"; rc_sgl=3; continue; }
      unset SGL_PATCH_DIR SGL_ENV SGL_EXTRA SGL_TOP_LOGPROBS SGL_SAMPLER
      export SGL_SCRIPT="$SGL_SCRIPT_PAIR" SGL_CORES="${SGL_CORES:-0-3}" SGL_CACHE="${SGL_CACHE_PAIR:-$DATA_ROOT/sgl_cache_v0.1.0_release}"
      "$PY" 76_serve_and_bench.py --backend sglang_neuron --model-dir "$src" --tokenizer "$src" \
        --name "$sname" --port "$PORT_SGL" --results "$R" --tag "${tag}_sgl" --ready-timeout-s "$READY_SGL_S" \
        --cache-dir "$SGL_CACHE" --bench-timeout-s "$BENCH_TIMEOUT_S" --bench "$BENCH1" --bench "$BENCH2" \
        > "$R/logs/76_${tag}_sgl.out" 2>&1
      rc_sgl=$?
      say "sgl: exit $rc_sgl: $(tail -n 1 "$R/logs/76_${tag}_sgl.out" | cut -c1-300)"
      if grep -q '^PRECHECK_FAILED' "$R/logs/76_${tag}_sgl.out" 2>/dev/null; then
        pc_failed="$pc_failed sgl"; say "sgl: PRECHECK FAILED (the bench ran anyway; the arm still counts as done): $(grep -m1 '^PRECHECK_FAILED' "$R/logs/76_${tag}_sgl.out" | cut -c1-200)"
      fi
      ;;
    *) say "unknown arm $arm (vllm or sgl)" ;;
  esac
done

table_args=()
[ "$rc_vllm" != 9 ] && table_args+=(--arm "vllm-neuron=${tag}_vllm")
[ "$rc_sgl" != 9 ] && table_args+=(--arm "sglang-neuron=${tag}_sgl")
if [ "${#table_args[@]}" -gt 0 ]; then
  "$PY" 77_pair_table.py --results "$R" "${table_args[@]}" --title "Baseline pair $model ($tag)" --out "$R/77_pair_${tag}.md" \
    > "$R/logs/77_${tag}.out" 2>&1 && say "table: $(tail -n 1 "$R/logs/77_${tag}.out")" || say "table FAILED: $(tail -n 1 "$R/logs/77_${tag}.out" | cut -c1-200)"
fi
want_ok=1
for arm in $ARMS; do
  case "$arm" in vllm) [ "$rc_vllm" = 0 ] || want_ok=0 ;; sgl) [ "$rc_sgl" = 0 ] || want_ok=0 ;; esac
done
pc_json=false; [ -n "$pc_failed" ] && pc_json=true
{
  # a failed pre-check does not change the exit codes (no re-run of the step); it is only recorded here and in 77's table
  printf '{"event": "pair", "model": "%s", "tag": "%s", "arms": "%s", "vllm_exit": %s, "sgl_exit": %s, "precheck_failed": %s, "precheck_failed_arms": "%s"}\n' \
    "$model" "$tag" "$ARMS" "$rc_vllm" "$rc_sgl" "$pc_json" "${pc_failed# }"
  echo '{"done": true}'
} > "$R/76_pair_${tag}.jsonl"
[ -n "$pc_failed" ] && echo "PRECHECK FAILED 76 $model $tag arm(s):$pc_failed (ignore_eos or usage not confirmed; the bench ran anyway)"
if [ "$want_ok" = 1 ]; then
  echo "P_DONE 76 $model $tag vllm=$rc_vllm sgl=$rc_sgl"
else
  echo "P_FAILED 76 $model $tag vllm=$rc_vllm sgl=$rc_sgl"; exit 1
fi
