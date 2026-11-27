#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src5 --entrypoint sh $IMG -c '
S=/usr/local/lib/python3.14/site-packages/sglang/srt
grep -rn "num_continuous_decode_steps" $S --include=*.py | head -20
echo ===RECV; grep -rn "scheduler_recv_interval" $S --include=*.py | head
echo ===MIXED; grep -rn "enable_mixed_chunk" /usr/local/lib/python3.14/site-packages/sglang_neuron --include=*.py | head
'
