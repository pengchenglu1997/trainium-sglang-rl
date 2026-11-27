IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
cat > /tmp/i29_in6.sh <<'EOF'
P=/usr/local/lib/python3.14/site-packages/sglang_neuron
grep -n -E "^COMPILER_ARGS|\"--verbose=35\"|\"-O1\",|modular-flow-mac-threshold=10|enable-verifier|hbm-scratchpad-page-size=\{" $P/model_executor/graph_runner.py | head
grep -n -E "def neuron_compiler_args|topk_topp\":|arg = \"-O2\"" $P/models/host.py
grep -n -E "^SCRATCHPAD_PAGE_MB" $P/serving_policy.py
grep -n -E "def torchneuronx_compiler_env|GENERIC_MODEL_TYPE_ARG = |SPLIT_FINDER_OPTION = |part of the NEFF" $P/backend/stack.py
grep -n -E "def get_neuronx_cc_args|\"compile\",|self.optimization_level,|args.extend\(self.extra_flags\)" /workspace/torch_neuronx/torch_neuronx/kernels/compiler_config.py
grep -n -E "cmd = \[|subprocess.run\(cmd" /workspace/torch_neuronx/torch_neuronx/kernels/compiler_subprocess.py
grep -n -E "class SampledExecution|sample_filtered_sharded_logits|sample_sharded_logits\(" $P/models/host.py
EOF
docker run --rm -v /tmp/i29_in6.sh:/i29_in6.sh:ro --entrypoint bash $IMG /i29_in6.sh
pgrep -fa "poll2.sh|poll.sh|chain.sh|i29_cold" || echo no_i29_procs
date -u
