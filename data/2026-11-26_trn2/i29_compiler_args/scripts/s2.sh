IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
cat > /tmp/i29_in2.sh <<'EOF'
P=/usr/local/lib/python3.14/site-packages/sglang_neuron
echo "### graph_runner.py 80-110"; sed -n '80,110p' $P/model_executor/graph_runner.py
echo "### graph_runner.py 600-650"; sed -n '600,650p' $P/model_executor/graph_runner.py
echo "### graph_runner.py 715-730"; sed -n '715,730p' $P/model_executor/graph_runner.py
echo "### stack.py 180-335"; sed -n '180,335p' $P/backend/stack.py
echo "### neuron_compiler_args defs"; grep -rn "def neuron_compiler_args" --include=*.py $P | grep -v /diffusion/
echo "### sampler compile"; grep -rn -E "compile_scope|compiler_args|torch.compile" --include=*.py $P/sampling.py $P/model_executor $P/models/transformer $P/general_plugin.py 2>/dev/null | head -40
echo "### torch_neuronx compile call"; T=$(python3 -c "import torch_neuronx,os;print(os.path.dirname(torch_neuronx.__file__))"); echo T=$T
grep -rn -E "NEURON_COMPILER_OPT_LEVEL|NEURON_CC_FLAGS|model-type|neuronx-cc" --include=*.py $T | head -30
EOF
docker run --rm -v /tmp/i29_in2.sh:/i29_in2.sh:ro --entrypoint bash $IMG /i29_in2.sh
