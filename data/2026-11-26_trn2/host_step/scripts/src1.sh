#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker ps --format '{{.Names}} {{.Status}}'
docker run --rm --name host_src1 --entrypoint sh $IMG -c '
S=/usr/local/lib/python3.14/site-packages/sglang/srt
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
grep -rn "def log_decode_stats" -A 60 $S/managers/scheduler_metrics_mixin.py | head -120
echo =====FPM
grep -rln "enable_forward_pass_metrics" $S $N | head
echo =====POLICY
grep -rn "decode_result_lag\|async_execution" $N --include=*.py | head -40
'
