#!/usr/bin/env bash
# Follow-up A: three copies of a fresh 0.6B-then-nothing folder.
#  p1: neuron/compile_cache/nki deleted -> i12_a1_fix   (cores 32-35)
#  p2: untouched                         -> i12_a1_ctl   (cores 36-39), control, expected to fail
#  p3: untouched, VLLM_NEURON_DISABLE_PARALLEL_TRACE=1 -> i12_a2_nopool (cores 40-43)
set -u
cd /home/ubuntu/w/i12
R=results; W=/home/ubuntu/w
bash srv_vllm_neuron_docker.sh stop i12_p_q06
echo "{\"name\": \"i12_p_q06\", \"utc\": \"$(date -u +%T)\", \"event\": \"stopped_after_ready\"}" >> $R/starts.jsonl
docker ps -a --format '{{.Names}}' | grep -q '^i12_p_q06$' && { echo "i12_p_q06 still exists"; exit 1; }
P0=$W/i12_cache_p0
sudo find $P0 -type f | sed "s#^$P0/##" | sort > $R/logs/i12_cache_p0_files_after_q06.txt
echo "p0 files: $(wc -l < $R/logs/i12_cache_p0_files_after_q06.txt)"
echo "names equal to the r1 list after 0.6B: $(diff -q $R/logs/i12_cache_p0_files_after_q06.txt $R/logs/i12_cache_r1_files_after_q06.txt > /dev/null && echo yes || echo no)"
diff $R/logs/i12_cache_p0_files_after_q06.txt $R/logs/i12_cache_r1_files_after_q06.txt > $R/logs/i12_cache_p0_vs_r1_names.diff
sudo python3 -I nki_entries.py $P0 > $R/nki_entries_p0_q06.txt
for d in p1 p2 p3; do [ -e $W/i12_cache_$d ] && { echo "$d exists"; exit 1; }; sudo cp -a $P0 $W/i12_cache_$d; done
sudo rm -rf $W/i12_cache_p1/neuron/compile_cache/nki
for d in p1 p2 p3; do echo "$d files $(sudo find $W/i12_cache_$d -type f | wc -l), nki dir: $(sudo ls -d $W/i12_cache_$d/neuron/compile_cache/nki 2>/dev/null || echo absent)"; done | tee $R/logs/poison_copies.txt
REUSE_CACHE=1 nohup bash run_start2.sh i12_a1_fix 32,33,34,35 30132 $W/i12_cache_p1 stop_on_pass 900 > $R/logs/run_a1_fix.out 2>&1 &
REUSE_CACHE=1 nohup bash run_start2.sh i12_a1_ctl 36,37,38,39 30133 $W/i12_cache_p2 stop_on_pass 900 > $R/logs/run_a1_ctl.out 2>&1 &
REUSE_CACHE=1 nohup bash run_start2.sh i12_a2_nopool 40,41,42,43 30134 $W/i12_cache_p3 keep 1500 VLLM_NEURON_DISABLE_PARALLEL_TRACE=1 > $R/logs/run_a2_nopool.out 2>&1 &
sleep 2; tail -n 6 $R/starts.jsonl | cut -c1-230
