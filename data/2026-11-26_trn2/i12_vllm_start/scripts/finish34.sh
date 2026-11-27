#!/usr/bin/env bash
# After both starts are ready: one completion each, save logs, list /var/tmp, stop.
set -u
cd /home/ubuntu/w/i12; R=results
for p in "i12_f34a 30142" "i12_f34b 30143"; do
  set -- $p; n=$1; port=$2
  curl -s -m 120 http://127.0.0.1:$port/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\": \"$n\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}" > $R/completion_$n.json
  echo "$n: $(cat $R/completion_$n.json | cut -c1-300)"
  docker logs $n > $R/logs/$n.log 2>&1
  docker exec $n sh -c 'for d in /var/tmp/*; do echo "$d $(find "$d" -type f | wc -l)"; done' > $R/logs/$n.vartmp.txt 2>&1
  echo "$n evrf=$(grep -c NCC_EVRF059 $R/logs/$n.log) ready_line=$(grep -n -m1 'Application startup complete' $R/logs/$n.log | cut -d: -f1)"
done
for n in i12_f34a i12_f34b; do bash srv_vllm_neuron_docker.sh stop $n; echo "{\"name\": \"$n\", \"utc\": \"$(date -u +%T)\", \"event\": \"stopped_after_ready\"}" >> $R/starts.jsonl; done
docker ps -a --format '{{.Names}}' | grep i12_ || echo "no i12_ containers"
