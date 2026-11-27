#!/usr/bin/env bash
# Two passes of the kit benchmark against i12_b_q06, one after the other, as the coordinator wrote them.
set -u
R=/home/ubuntu/w/i12/results
URL=http://127.0.0.1:30141; NAME=i12_b_q06; TOK=/home/ubuntu/w/models/Qwen3-0.6B
cd /home/ubuntu/w/sglang_rl || exit 1
for t in b_q06_vllm_p1 b_q06_vllm_p2; do
  echo "{\"name\": \"$NAME\", \"utc\": \"$(date -u +%T)\", \"event\": \"bench_start\", \"tag\": \"$t\"}" >> $R/starts.jsonl
  LB_PROMPTS=1024 LB_CONC=1,8,32 bash 22b_llmbench_grid.sh $t $URL $NAME $TOK /home/ubuntu/w/results_s4 4 > $R/logs/bench_$t.out 2>&1
  rc=$?
  echo "{\"name\": \"$NAME\", \"utc\": \"$(date -u +%T)\", \"event\": \"bench_end\", \"tag\": \"$t\", \"rc\": $rc, \"last\": \"$(tail -n 1 $R/logs/bench_$t.out | tr -d '"' | cut -c1-200)\"}" >> $R/starts.jsonl
done
echo BENCH_ALL_DONE >> $R/logs/bench_b06_done.txt
