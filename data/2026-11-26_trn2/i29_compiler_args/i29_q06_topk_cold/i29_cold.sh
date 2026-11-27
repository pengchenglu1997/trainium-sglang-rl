#!/usr/bin/env bash
# i29 Part A: cold start of released topk_topp sampler, Qwen3-0.6B, cores 12-15, own empty cache.
# Same server flags as s4_x.sh (FAST), no patch dir (image as released), no extra env.
# A host-side poller records every neuronx-cc command line (from /proc/<pid>/cmdline) with its cgroup.
set -u
W=$HOME/w/i29; mkdir -p "$W"
cd $HOME/w/sglang_rl || exit 2
export MODELS_ROOT=$HOME/w/models
export SGL_CACHE=$HOME/w/sgl_cache_s4_i29_q06_topk_cold
export SGL_SCRIPT=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh
export SGL_SAMPLER=topk_topp
export SGL_EXTRA="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32"
export SGL_CORES=12-15
unset SGL_PATCH_DIR SGL_ENV SGL_MOUNTS SGL_TOP_LOGPROBS
NAME=i29_q06_topk_cold; PORT=30340
echo "== $(date -u +%FT%TZ) cache before: $(ls -A $SGL_CACHE 2>/dev/null | wc -l) entries"
bash srv_sglang_neuron.sh start "$MODELS_ROOT/Qwen3-0.6B" $PORT $NAME
CID=$(docker inspect -f '{{.Id}}' $NAME); echo "== container $CID"
echo "$CID" > "$W/cid.txt"
# poller
(
  : > "$W/cc_cmdlines.txt"
  while [ -e "$W/poll.on" ]; do
    for p in $(pgrep -f neuronx-cc 2>/dev/null) $(pgrep -f neuronxcc 2>/dev/null); do
      [ -e "$W/seen_$p" ] && continue
      cl=$(sudo cat /proc/$p/cmdline 2>/dev/null | tr '\0' ' ')
      [ -n "$cl" ] || continue
      cg=$(cat /proc/$p/cgroup 2>/dev/null | tr '\n' ' ')
      cwd=$(sudo readlink /proc/$p/cwd 2>/dev/null)
      touch "$W/seen_$p"
      echo "$(date -u +%FT%TZ) pid=$p cgroup=[$cg] cwd=$cwd CMD: $cl" >> "$W/cc_cmdlines.txt"
    done
    sleep 0.3
  done
) &
touch "$W/poll.on"
t0=$(date +%s); ready=0
while [ $(( $(date +%s) - t0 )) -lt 2400 ]; do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && curl -s -m 5 "http://127.0.0.1:$PORT/get_model_info" | grep -q model_path; then ready=1; break; fi
  [ "$(bash srv_sglang_neuron.sh alive $NAME)" = true ] || { echo "== container exited"; break; }
  sleep 10
done
echo "== ready=$ready after $(( $(date +%s) - t0 )) s"
if [ $ready = 1 ]; then
  curl -s -m 120 "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' -d '{"text":"The capital of France is","sampling_params":{"temperature":0.8,"top_k":50,"top_p":0.95,"max_new_tokens":16}}' > "$W/smoke.json"; echo; echo "== smoke done"
  sleep 20
fi
rm -f "$W/poll.on"; sleep 1
bash srv_sglang_neuron.sh logs $NAME > "$W/server_${NAME}.log" 2>&1
docker inspect $NAME > "$W/inspect_${NAME}.json" 2>&1
bash srv_sglang_neuron.sh stop $NAME
find "$SGL_CACHE" -type f | sort > "$W/cache_files.txt"
rm -f "$W"/seen_*
echo "I29_COLD_DONE ready=$ready"
