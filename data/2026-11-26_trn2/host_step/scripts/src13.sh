#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src13 --entrypoint sh $IMG -c '
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
G=$N/model_executor/graph_runner.py
sed -n 160,240p $G
echo "== SUPPORTED_KV_SEGMENT_SIZES / context ladder"
grep -rnE "SUPPORTED_KV_SEGMENT_SIZES *=|decode_contexts|context_ladder|kv_segment" $N/model_executor/*.py | head -20
'
