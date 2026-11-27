#!/usr/bin/env bash
IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
docker run --rm --name host_src3 --entrypoint sh $IMG -c '
S=/usr/local/lib/python3.14/site-packages/sglang/srt
M=$S/managers/scheduler_components/metrics_reporter.py
sed -n 716,760p $M; echo ...; sed -n 898,1000p $M; echo ...; sed -n 1144,1195p $M
echo ====NGT; grep -rn "num_generated_tokens" $S/managers | head
'
