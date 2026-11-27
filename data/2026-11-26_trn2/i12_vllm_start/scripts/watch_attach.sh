#!/usr/bin/env bash
# watch_attach.sh <name> <port> <cache dir> <mode keep|stop_on_pass> <cap_s> <t0 unix>
# Re-attaches the watch loop of run_start.sh to a container that is already running (the first watcher was killed
# because its script file was replaced while it ran). Same stop rules: first NCC_EVRF059 line -> stop.
set -u
name=$1; port=$2; cache=$3; mode=$4; cap=$5; t0=$6
case "$name" in i12_*) ;; *) echo "name must start with i12_"; exit 2 ;; esac
R=/home/ubuntu/w/i12/results; L=$R/logs
ev() { echo "{\"name\": \"$name\", \"unix\": $(date +%s.%N | cut -c1-14), \"utc\": \"$(date -u +%H:%M:%S)\", $1}" >> "$R/starts.jsonl"; }
ev "\"event\": \"watch_reattached\", \"t0\": $t0"
verdict=""
grep -q "Parallel trace finished" "$L/${name}.log" 2>/dev/null && passed_at=already || passed_at=""
while :; do
  sleep 5
  now=$(date +%s); el=$((now - t0))
  docker logs "$name" > "$L/${name}.log" 2>&1
  if grep -q "NCC_EVRF059" "$L/${name}.log"; then
    verdict=fail; ev "\"event\": \"evrf059\", \"elapsed_s\": $el, \"line_no\": $(grep -n -m1 NCC_EVRF059 "$L/${name}.log" | cut -d: -f1)"; break
  fi
  if [ -z "$passed_at" ] && grep -q "Parallel trace finished" "$L/${name}.log"; then
    passed_at=$el; ev "\"event\": \"parallel_trace_finished\", \"elapsed_s\": $el"
    [ "$mode" = stop_on_pass ] && { verdict=pass_stopped; break; }
  fi
  if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" > "$L/${name}.models.json" 2>/dev/null; then
    verdict=ready; ev "\"event\": \"ready\", \"elapsed_s\": $el, \"ready_line_no\": $(grep -n -m1 -E 'Application startup complete' "$L/${name}.log" | cut -d: -f1 || echo null)"; break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    verdict=exited; ev "\"event\": \"container_exited\", \"elapsed_s\": $el"; break
  fi
  [ "$el" -ge "$cap" ] && { verdict=cap; ev "\"event\": \"cap\", \"elapsed_s\": $el"; break; }
done
docker logs "$name" > "$L/${name}.log" 2>&1
docker exec "$name" sh -c 'for d in /var/tmp/*; do echo "$d $(find "$d" -type f | wc -l)"; done' > "$L/${name}.vartmp.txt" 2>&1
sudo find "$cache" -maxdepth 3 > "$L/${name}.cache_tree.txt" 2>&1
ev "\"event\": \"end\", \"verdict\": \"$verdict\", \"cache_files_after\": $(sudo find "$cache" -type f 2>/dev/null | wc -l)"
if [ "$verdict" != ready ]; then
  bash /home/ubuntu/w/i12/srv_vllm_neuron_docker.sh stop "$name"; ev "\"event\": \"stopped\""
fi
