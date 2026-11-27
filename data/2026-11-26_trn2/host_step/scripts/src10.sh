#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
P=$HOME/w/sglang_rl/patches/sglang_neuron_int_topk
echo "== patch dir"
grep -rnE "CONSECUTIVE_PREFILL|disable_overlap_for_batch" --include=*.py "$P" | head
docker run --rm --name host_src10 --entrypoint sh $IMG -c '
S=/usr/local/lib/python3.14/site-packages/sglang/srt
N=/usr/local/lib/python3.14/site-packages/sglang_neuron
echo "== sglang"
grep -rnE "CONSECUTIVE_PREFILL|disable_overlap_for_batch" $S --include=*.py | head
f=$(grep -rlE "def is_disable_overlap_for_batch" $S --include=*.py | head -1); echo FILE $f
n=$(grep -n "def is_disable_overlap_for_batch" $f | cut -d: -f1); sed -n "${n},$((n+30))p" $f
echo "== neuron"
grep -rnE "CONSECUTIVE_PREFILL|disable_overlap_for_batch|is_disable_overlap" $N --include=*.py | head
'
