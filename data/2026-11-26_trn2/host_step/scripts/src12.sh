#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
P=$HOME/w/sglang_rl/patches/sglang_neuron_int_topk
echo "== patch env_registry bucket/context names"
grep -nE "BUCKET|CONTEXT|PREFILL" "$P/env_registry.py" | head -30
docker run --rm --name host_src12 --entrypoint sh $IMG -c '
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
echo "== warm decode / prefill bucket source"
grep -rnE "warm decode bucket|warm prefill bucket" $N --include=*.py | head
echo "== context ladder"
grep -rnE "def .*context_buckets|def .*context_ladder|context_buckets *=|prefill_buckets *=|def .*prefill_buckets" $N --include=*.py | head -20
'
