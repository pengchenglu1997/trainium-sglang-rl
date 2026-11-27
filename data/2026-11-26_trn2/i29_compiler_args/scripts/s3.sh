IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
cat > /tmp/i29_in3.sh <<'EOF'
P=/usr/local/lib/python3.14/site-packages/sglang_neuron
echo "### host.py 95-125"; sed -n '95,125p' $P/models/host.py
echo "### which model class for qwen3"; grep -rn -E "Qwen3ForCausalLM" --include=*.py $P/models/catalog.py | head
echo "### sampler in graph"; grep -rn -E "SGLANG_NEURON_SAMPLER|topk_topp" --include=*.py $P | grep -v /diffusion/ | head -20
echo "### torch_neuronx compiler cmd"; grep -rn -E "NEURON_COMPILER_OPT_LEVEL|NEURON_CC_FLAGS|\"neuronx-cc\"|'neuronx-cc'|--model-type|--target|--lnc|logical-nc" --include=*.py /workspace/torch_neuronx/torch_neuronx | head -40
EOF
docker run --rm -v /tmp/i29_in3.sh:/i29_in3.sh:ro --entrypoint bash $IMG /i29_in3.sh
