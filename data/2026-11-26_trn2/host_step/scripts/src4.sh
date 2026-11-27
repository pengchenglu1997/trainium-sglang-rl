#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src4 --entrypoint sh $IMG -c '
S=/usr/local/lib/python3.14/site-packages/sglang/srt
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
grep -rn "LOG_FORWARD_ITERS =\|RECORD_STEP_TIME =\|LOG_FORWARD_ITERS=\|RECORD_STEP_TIME=" $S --include=*.py | head
echo ===POL; sed -n 1,80p $N/serving_policy.py | grep -n "env\|SGLANG\|def \|field\|:" | head -60
echo ===NCDS; grep -rn "num_continuous_decode_steps\|scheduler_recv_interval" $N --include=*.py | head -20
'
