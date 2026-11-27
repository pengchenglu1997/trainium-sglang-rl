#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
P=$HOME/w/sglang_rl/patches/sglang_neuron_int_topk
echo "== patch dir"
grep -rnE "prefill_max_requests|max_prefill_bs|prefill_batch_size|PREFILL_BS|prefill_bs|max_prefill_reqs" --include=*.py "$P" | head -20
echo "== image"
docker run --rm --name host_src8 --entrypoint sh $IMG -c '
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
grep -rnE "prefill_max_requests|max_prefill_bs|prefill_batch_size|PREFILL_BS|prefill_bs|max_prefill_reqs" $N --include=*.py | head -30
'
