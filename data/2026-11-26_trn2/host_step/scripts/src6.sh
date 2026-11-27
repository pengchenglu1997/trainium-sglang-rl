#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src6 --entrypoint sh $IMG -c '
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
grep -rln "HOST_TRACE" $N --include=*.py
f=$(grep -rln "SGLANG_NEURON_HOST_TRACE_SPANS" $N --include=*.py | grep -v env_registry | head -1); echo FILE $f
sed -n 1,140p $f
'
