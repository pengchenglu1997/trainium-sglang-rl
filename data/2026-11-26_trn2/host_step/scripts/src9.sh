#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
P=$HOME/w/sglang_rl/patches/sglang_neuron_int_topk
docker run --rm --name host_src9 --entrypoint sh $IMG -c '
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
grep -rnE "PrefillAdder|add_one_req|can_run_list|one sequence|single sequence|one prompt|prefill_one" $N --include=*.py | head -30
echo ==== grid
sed -n 1,60p $N/sglang_patches/prefill_chunk_grid.py
'
