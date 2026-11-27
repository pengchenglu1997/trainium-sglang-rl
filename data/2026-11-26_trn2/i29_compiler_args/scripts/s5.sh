IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
cat > /tmp/i29_in5.sh <<'EOF'
P=/usr/local/lib/python3.14/site-packages/sglang_neuron
echo "### host.py 1-30"; sed -n '1,30p' $P/models/host.py
echo "### host.py 200-245"; sed -n '200,245p' $P/models/host.py
echo "### SampledExecution users"; grep -rn "SampledExecution" --include=*.py $P | grep -v /diffusion/ | head
echo "### qwen3 sglang_model 35-80"; sed -n '35,80p' $P/models/qwen3/sglang_model.py
echo "### SCRATCHPAD"; grep -rn -E "scratchpad_page_mb|SCRATCHPAD_PAGE_MB *=" --include=*.py $P/serving_policy.py $P/runtime_init.py 2>/dev/null | head
EOF
docker run --rm -v /tmp/i29_in5.sh:/i29_in5.sh:ro --entrypoint bash $IMG /i29_in5.sh
