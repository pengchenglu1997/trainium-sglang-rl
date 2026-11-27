#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src2 --entrypoint sh $IMG -c '
S=/usr/local/lib/python3.14/site-packages/sglang/srt
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
M=$S/managers/scheduler_components/metrics_reporter.py
grep -n "def \|gap_latency\|perf_counter\|decode_log_interval\|Decode batch\|forward_pass" $M | head -80
echo ====OBS
grep -n "forward_pass\|decode_log\|request_time_stats\|export_metrics" -A3 $S/arg_groups/fields/observability.py | head -60
echo ====POLICYFILE
ls $N; grep -rln "def current" $N | head
'
