#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src11 --entrypoint sh $IMG -c '
S=/usr/local/lib/python3.14/site-packages/sglang/srt
sed -n 2120,2200p $S/managers/scheduler.py
'
