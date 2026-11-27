#!/usr/bin/env bash
# Order of the session before: 8B on a cache folder that a stopped 0.6B container left behind.
# r1  : plain launch (attempt-1 condition)        cores 32-35
# r1b : copy of the same folder, NKI_ENABLE_TRACE_CACHE=0   cores 36-39
set -u
cd /home/ubuntu/w/i12
R=results
bash srv_vllm_neuron_docker.sh stop i12_r1_q06
echo "{\"name\": \"i12_r1_q06\", \"utc\": \"$(date -u +%T)\", \"event\": \"stopped_after_ready\"}" >> $R/starts.jsonl
docker ps -a --format '{{.Names}}' | grep -q '^i12_r1_q06$' && { echo "i12_r1_q06 still exists"; exit 1; }
C=/home/ubuntu/w/i12_cache_r1
sudo find $C -type f | sed "s#^$C/##" | sort > $R/logs/i12_cache_r1_files_after_q06.txt
[ -e /home/ubuntu/w/i12_cache_r1b ] && { echo "r1b exists"; exit 1; }
sudo cp -a $C /home/ubuntu/w/i12_cache_r1b
echo "r1 files $(sudo find $C -type f | wc -l), r1b files $(sudo find /home/ubuntu/w/i12_cache_r1b -type f | wc -l)"
REUSE_CACHE=1 nohup bash run_start2.sh i12_r1_q8 32,33,34,35 30128 $C stop_on_pass 900 > $R/logs/run_r1_q8.out 2>&1 &
REUSE_CACHE=1 nohup bash run_start2.sh i12_r1b_q8 36,37,38,39 30129 /home/ubuntu/w/i12_cache_r1b stop_on_pass 900 NKI_ENABLE_TRACE_CACHE=0 > $R/logs/run_r1b_q8.out 2>&1 &
sleep 2
tail -n 4 $R/starts.jsonl
