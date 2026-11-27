#!/usr/bin/env bash
# run_one.sh <name> <cache dir> <old|new> <port> <cap_s>
# One Qwen3-8B vllm-neuron start on cores 12-15 with the given launcher copy; to ready, NCC_EVRF059, exit or cap.
# On ready: one completion. Always: save the log, list the index before/after, stop the container.
set -u
name=$1; cache=$2; which=$3; port=$4; cap=$5
case "$name" in i34_*) ;; *) echo "name must start with i34_"; exit 2 ;; esac
case "$cache" in /home/ubuntu/w/i34/cache_*) ;; *) echo "cache must be ~/w/i34/cache_*"; exit 2 ;; esac
W=/home/ubuntu/w/i34; R=$W/results; L=$R/logs; mkdir -p "$L"
ev() { echo "{\"name\": \"$name\", \"unix\": $(date +%s.%N | cut -c1-14), \"utc\": \"$(date -u +%H:%M:%S)\", $1}" >> "$R/starts.jsonl"; }
sudo python3 $W/nki_entries.py "$cache" > "$L/${name}.nki_before.txt" 2>&1
CF=$(sudo find "$cache" -type f | wc -l)
export MODELS_ROOT=/home/ubuntu/w/models VLLM_CACHE="$cache" VLLM_DEVICES=12,13,14,15
export VLLM_TP=4 VLLM_MAX_SEQS=32 VLLM_MAX_LEN=4096 VLLM_COMPILE_TIMEOUT=1800
unset VLLM_EXTRA VLLM_ENV VLLM_NKI_INDEX_CHECK
ev "\"event\": \"config\", \"model\": \"Qwen3-8B\", \"launcher\": \"$which\", \"cores\": \"12,13,14,15\", \"port\": $port, \"cache\": \"$cache\", \"cache_files_before\": $CF, \"nki_entries_before\": \"$(head -1 "$L/${name}.nki_before.txt")\", \"cap_s\": $cap"
t0=$(date +%s.%N)
bash $W/$which/srv_vllm_neuron_docker.sh start /home/ubuntu/w/models/Qwen3-8B "$port" "$name" > "$L/${name}.launch.out" 2>&1
rc=$?; t1=$(date +%s.%N)
ev "\"event\": \"launched\", \"rc\": $rc, \"launcher_s\": $(python3 -c "print(round($t1-$t0,1))"), \"check_line\": \"$(grep 'nki index check' "$L/${name}.launch.out" | tr -d '"')\", \"container_started\": \"$(docker inspect -f '{{.State.StartedAt}}' "$name" 2>/dev/null)\""
sudo python3 $W/nki_entries.py "$cache" > "$L/${name}.nki_after_check.txt" 2>&1
verdict=""
while :; do
  sleep 5
  el=$(python3 -c "print(round($(date +%s.%N)-$t0))")
  docker logs "$name" > "$L/${name}.log" 2>&1
  if grep -q "NCC_EVRF059" "$L/${name}.log"; then
    verdict=fail; ev "\"event\": \"evrf059\", \"elapsed_s\": $el, \"line_no\": $(grep -n -m1 NCC_EVRF059 "$L/${name}.log" | cut -d: -f1)"; break
  fi
  if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" > "$L/${name}.models.json" 2>/dev/null; then
    verdict=ready; ev "\"event\": \"ready\", \"elapsed_s\": $el, \"ready_line_no\": \"$(grep -n -m1 'Application startup complete' "$L/${name}.log" | cut -d: -f1)\""; break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    verdict=exited; ev "\"event\": \"container_exited\", \"elapsed_s\": $el, \"exit_code\": \"$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null)\""; break
  fi
  [ "$el" -ge "$cap" ] && { verdict=cap; ev "\"event\": \"cap\", \"elapsed_s\": $el"; break; }
done
if [ "$verdict" = ready ]; then
  curl -s -m 120 "http://127.0.0.1:$port/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\": \"$name\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}" > "$R/completion_$name.json"
  ev "\"event\": \"completion\", \"bytes\": $(wc -c < "$R/completion_$name.json")"
fi
docker logs "$name" > "$L/${name}.log" 2>&1
bash $W/old/srv_vllm_neuron_docker.sh stop "$name"
ev "\"event\": \"end\", \"verdict\": \"$verdict\", \"evrf059_lines\": $(grep -c NCC_EVRF059 "$L/${name}.log"), \"cache_files_after\": $(sudo find "$cache" -type f | wc -l)"
sudo python3 $W/nki_entries.py "$cache" > "$L/${name}.nki_after_stop.txt" 2>&1
ev "\"event\": \"stopped\", \"still_listed\": \"$(docker ps -a --format '{{.Names}}' | grep -x "$name")\""
