#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src7 --entrypoint sh $IMG -c '
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
wc -l $N/managers/host_loop.py; sed -n 1,60p $N/managers/host_loop.py
grep -n "def \|_trace\|add(" $N/managers/host_loop.py | head -60
'
