#!/usr/bin/env bash
cd /home/ubuntu/w/i12/results; date -u +%T
for n in i12_s1a i12_s1b i12_s2 i12_s3 ${EXTRA:-}; do
  f=logs/$n.log; [ -f $f ] || continue
  echo "$n lines=$(wc -l < $f) warn=$(grep -c 'NKI_ENABLE_TRACE_CACHE=1' $f) evrf=$(grep -c NCC_EVRF059 $f) ptstart=$(grep -c 'Parallel trace: jobs' $f) ptfin=$(grep -c 'Parallel trace finished' $f) err=$(grep -c -E 'ERROR|Traceback' $f) last=$(tail -n1 $f | cut -c1-160)"
done
echo "--- events"; tail -n ${EV:-8} starts.jsonl
