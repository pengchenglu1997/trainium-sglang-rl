IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
cat > /tmp/i29_in4.sh <<'EOF'
P=/usr/local/lib/python3.14/site-packages/sglang_neuron
echo "### qwen3 package"; ls $P/models/qwen3; grep -n -E "^class |import .*host|neuron_compiler_args" $P/models/qwen3/*.py | head -20
grep -n -E "^class " $P/models/host.py | head
T=/workspace/torch_neuronx/torch_neuronx
echo "### compiler_subprocess 200-300"; sed -n '200,300p' $T/kernels/compiler_subprocess.py
echo "### compiler_config 50-110"; sed -n '50,110p' $T/kernels/compiler_config.py
echo "### cache_utils 120-160"; sed -n '120,160p' $T/kernels/cache_utils.py
echo "### logging of command"; grep -rn -E "logger\.(info|debug|warning)\(.*(cmd|command|args)" $T/kernels/compiler_subprocess.py $T/kernels/neuronx_cc_wrapper.py $T/neuron_dynamo_backend/*.py 2>/dev/null | head -20
EOF
docker run --rm -v /tmp/i29_in4.sh:/i29_in4.sh:ro --entrypoint bash $IMG /i29_in4.sh
