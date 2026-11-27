#!/usr/bin/env bash
# i29 Part A, follow-up cold starts: i29_cold2.sh <tag> <sampler>
# Released image (no patch dir), Qwen3-0.6B, same server flags as s4_x.sh, cores 12-15, port 30341, own empty cache.
# The neuronx-cc poller (poll2.sh) starts before the container, so every compile command is recorded.
set -u
tag="${1:?tag}"; sampler="${2:?sampler}"
W=$HOME/w/i29; OUT=$W/$tag; mkdir -p "$OUT"
cd $HOME/w/sglang_rl || exit 2
export MODELS_ROOT=$HOME/w/models
export SGL_CACHE=$HOME/w/sgl_cache_s4_$tag
export SGL_SCRIPT=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh
export SGL_SAMPLER=$sampler
export SGL_EXTRA="--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32"
export SGL_CORES=12-15
unset SGL_PATCH_DIR SGL_ENV SGL_MOUNTS SGL_TOP_LOGPROBS
NAME=$tag; PORT=30341
echo "== $(date -u +%FT%TZ) $tag sampler=$sampler cache before: $(ls -A $SGL_CACHE 2>/dev/null | wc -l) entries"
rm -f "$OUT/poll.stop"; : > "$OUT/cc_cmdlines.txt"
(setsid nohup bash $W/poll2.sh "$OUT" "$NAME" > "$OUT/poll.log" 2>&1 < /dev/null &)
sleep 1
bash srv_sglang_neuron.sh start "$MODELS_ROOT/Qwen3-0.6B" $PORT $NAME
t0=$(date +%s); ready=0
while [ $(( $(date +%s) - t0 )) -lt 2400 ]; do
  if curl -s -m 5 "http://127.0.0.1:$PORT/get_model_info" | grep -q model_path; then ready=1; break; fi
  [ "$(bash srv_sglang_neuron.sh alive $NAME)" = true ] || { echo "== container exited"; break; }
  sleep 10
done
echo "== ready=$ready after $(( $(date +%s) - t0 )) s"
if [ $ready = 1 ]; then
  curl -s -m 120 "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' -d '{"text":"The capital of France is","sampling_params":{"temperature":0.8,"max_new_tokens":16}}' > "$OUT/smoke.json"; echo "== smoke done"
fi
touch "$OUT/poll.stop"; sleep 2
bash srv_sglang_neuron.sh logs $NAME > "$OUT/server_${NAME}.log" 2>&1
bash srv_sglang_neuron.sh stop $NAME
sudo find "$SGL_CACHE" -type f | sort > "$OUT/cache_files.txt"
echo "I29_COLD2_DONE $tag ready=$ready"
