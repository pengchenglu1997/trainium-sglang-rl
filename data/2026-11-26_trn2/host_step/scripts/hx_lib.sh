#!/usr/bin/env bash
# host_ slot helpers (cores 32-47, ports 30500-30549, folder ~/w/host). Sourced by the phase scripts.
set -u
KIT="$HOME/w/sglang_rl"; H="$HOME/w/host"; R="$H/results"; mkdir -p "$R/server_logs" "$R/ps"
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export DATA_ROOT="$HOME/w" MODELS_ROOT="$HOME/w/models"
FAST="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32"
ALLENV="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
PATCH="$KIT/patches/sglang_neuron_int_topk"
say() { echo "== H $(date -u +%H:%M:%S) $*"; }
# mkcache <src tag> <dst name>: copy a warm cache of tonight to ~/w/sgl_cache_host_<dst>
mkcache() { local d="$HOME/w/sgl_cache_host_$2"; [ -d "$d" ] || sudo cp -a "$HOME/w/sgl_cache_s4_$1" "$d"; echo "$d"; }
# start <name> <06|8> <cores> <port> <cache dir> "<extra flags after FAST>" "<extra env after ALLENV>"
start() {
  local name="$1" size="$2" cores="$3" port="$4" cache="$5" extra="$6" xenv="$7" mounts="${8-}" model script
  case "$size" in 06) model=Qwen3-0.6B; script=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh ;;
                  8) model=Qwen3-8B; script=/opt/sglang-neuron/examples/qwen3_8b/run_server.sh ;; esac
  case "$name" in host_*) ;; *) echo "bad name $name"; return 2 ;; esac
  SGL_CACHE="$cache" SGL_SCRIPT="$script" SGL_SAMPLER=temperature SGL_CORES="$cores" SGL_PATCH_DIR="$PATCH" \
    SGL_EXTRA="$FAST $extra" SGL_ENV="$ALLENV $xenv" SGL_MOUNTS="$mounts" bash "$KIT/srv_sglang_neuron.sh" start "$MODELS_ROOT/$model" "$port" "$name" >/dev/null
  echo "{\"event\":\"start\",\"name\":\"$name\",\"size\":\"$size\",\"cores\":\"$cores\",\"port\":$port,\"cache\":\"$cache\",\"extra\":\"$FAST $extra\",\"env\":\"$ALLENV $xenv\",\"patch\":\"$PATCH\",\"mounts\":\"$mounts\",\"unix\":$(date +%s)}" >> "$R/events.jsonl"
  date +%s > "$R/.t0_$name"
}
# wait_ready <name> <port>: up to 40 min; prints ready seconds or FAIL
wait_ready() {
  local name="$1" port="$2" t0 now
  t0=$(cat "$R/.t0_$name")
  while true; do
    now=$(date +%s)
    if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
      echo "{\"event\":\"dead\",\"name\":\"$name\",\"after_s\":$((now-t0))}" >> "$R/events.jsonl"; echo FAIL; return 1; fi
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" --max-time 3)" = 200 ]; then
      echo "{\"event\":\"ready\",\"name\":\"$name\",\"ready_s\":$((now-t0))}" >> "$R/events.jsonl"; echo $((now-t0)); return 0; fi
    [ $((now-t0)) -gt 2400 ] && { echo "{\"event\":\"timeout\",\"name\":\"$name\"}" >> "$R/events.jsonl"; echo FAIL; return 1; }
    sleep 5
  done
}
snap() { { echo "# $(date -u +%FT%TZ) $1"; docker ps --format '{{.Names}} {{.Status}} {{.Image}}'; uptime; } >> "$R/ps/ps.txt"; }
# bench <tag> <port> <06|8>: llmbench prompt 1024, conc 1/8/32, temperature 0 (as s4_x.sh bench t)
bench() {
  local tag="$1" port="$2" size="$3" model
  case "$size" in 06) model=Qwen3-0.6B ;; 8) model=Qwen3-8B ;; esac
  snap "before $tag"
  echo "{\"event\":\"bench_start\",\"tag\":\"$tag\",\"port\":$port,\"unix\":$(date +%s)}" >> "$R/events.jsonl"
  ( cd "$KIT" && env LB_PROMPTS=1024 LB_CONC=1,8,32 bash 22b_llmbench_grid.sh "$tag" "http://127.0.0.1:$port" "/models/$model" "$MODELS_ROOT/$model" "$R" 4 ) > "$R/bench_$tag.out" 2>&1
  local rc=$?
  echo "{\"event\":\"bench_end\",\"tag\":\"$tag\",\"rc\":$rc,\"last\":\"$(tail -n 1 "$R/bench_$tag.out" | tr -d '"' | cut -c1-200)\",\"unix\":$(date +%s)}" >> "$R/events.jsonl"
  snap "after $tag"
  say "bench $tag rc=$rc $(tail -n 1 "$R/bench_$tag.out" | cut -c1-160)"
  return $rc
}
stopsrv() { docker logs "$1" > "$R/server_logs/$1.log" 2>&1; bash "$KIT/srv_sglang_neuron.sh" stop "$1"; echo "{\"event\":\"stopped\",\"name\":\"$1\",\"unix\":$(date +%s)}" >> "$R/events.jsonl"; }
