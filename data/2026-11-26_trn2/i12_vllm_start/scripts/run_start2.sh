#!/usr/bin/env bash
# run_start.sh <name> <cores e.g. 28,29,30,31> <port> <cache dir> <mode keep|stop_on_pass> <cap_s> [VLLM_ENV items...]
# One start with the run-repo launcher copy; stops at the first NCC_EVRF059 line.
set -u
name=$1; cores=$2; port=$3; cache=$4; mode=$5; cap=$6; shift 6
case "$name" in i12_*) ;; *) echo "name must start with i12_"; exit 2 ;; esac
case "$cache" in /home/ubuntu/w/i12_cache_*) ;; *) echo "cache must be ~/w/i12_cache_*"; exit 2 ;; esac
R=/home/ubuntu/w/i12/results; L=$R/logs; mkdir -p "$L"
ev() { echo "{\"name\": \"$name\", \"unix\": $(date +%s.%N | cut -c1-14), \"utc\": \"$(date -u +%H:%M:%S)\", $1}" >> "$R/starts.jsonl"; }
if [ "${REUSE_CACHE:-0}" != 1 ]; then [ -e "$cache" ] && { echo "cache folder exists: $cache"; exit 2; }; fi
mkdir -p "$cache"; CF=$(sudo find "$cache" -type f | wc -l)
export MODELS_ROOT=/home/ubuntu/w/models VLLM_CACHE="$cache" VLLM_DEVICES="$cores"
export VLLM_TP=4 VLLM_MAX_SEQS=32 VLLM_MAX_LEN=4096 VLLM_COMPILE_TIMEOUT=1800
unset VLLM_EXTRA
export VLLM_ENV="$*"
ev "\"event\": \"config\", \"model\": \"${MODEL:-Qwen3-8B}\", \"cores\": \"$cores\", \"port\": $port, \"cache\": \"$cache\", \"cache_files_before\": $CF, \"reuse_cache\": \"${REUSE_CACHE:-0}\", \"vllm_env\": \"$VLLM_ENV\", \"mode\": \"$mode\", \"cap_s\": $cap"
t0=$(date +%s)
bash /home/ubuntu/w/i12/srv_vllm_neuron_docker.sh start "/home/ubuntu/w/models/${MODEL:-Qwen3-8B}" "$port" "$name" > "$L/${name}.launch.out" 2>&1
ev "\"event\": \"launched\", \"rc\": $?, \"container_started\": \"$(docker inspect -f '{{.State.StartedAt}}' "$name" 2>/dev/null)\""
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$name" > "$L/${name}.env.txt" 2>&1
verdict=""; passed_at=""
while :; do
  sleep 5
  now=$(date +%s); el=$((now - t0))
  docker logs "$name" > "$L/${name}.log" 2>&1
  if [ -z "$verdict" ] && grep -q "NCC_EVRF059" "$L/${name}.log"; then
    verdict=fail; ev "\"event\": \"evrf059\", \"elapsed_s\": $el, \"line_no\": $(grep -n -m1 NCC_EVRF059 "$L/${name}.log" | cut -d: -f1)"
    break
  fi
  if [ -z "$passed_at" ] && grep -q "Parallel trace finished" "$L/${name}.log"; then
    passed_at=$el; ev "\"event\": \"parallel_trace_finished\", \"elapsed_s\": $el, \"line_no\": $(grep -n -m1 'Parallel trace finished' "$L/${name}.log" | cut -d: -f1)"
    [ "$mode" = stop_on_pass ] && { verdict=pass_stopped; break; }
  fi
  if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" > "$L/${name}.models.json" 2>/dev/null; then
    verdict=ready; ev "\"event\": \"ready\", \"elapsed_s\": $el"; break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    verdict=exited; ev "\"event\": \"container_exited\", \"elapsed_s\": $el, \"exit_code\": \"$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null)\""; break
  fi
  [ "$el" -ge "$cap" ] && { verdict=cap; ev "\"event\": \"cap\", \"elapsed_s\": $el"; break; }
done
docker logs "$name" > "$L/${name}.log" 2>&1
# list the container's /var/tmp NKI folders (names and counts only) before stopping
docker exec "$name" sh -c 'for d in /var/tmp/*; do echo "$d $(find "$d" -type f | wc -l)"; done' > "$L/${name}.vartmp.txt" 2>&1
find "$cache" -maxdepth 3 > "$L/${name}.cache_tree.txt" 2>&1
ev "\"event\": \"end\", \"verdict\": \"$verdict\", \"cache_files_after\": $(sudo find "$cache" -type f 2>/dev/null | wc -l)"
if [ "$verdict" != ready ]; then
  bash /home/ubuntu/w/i12/srv_vllm_neuron_docker.sh stop "$name"; ev "\"event\": \"stopped\""
fi
