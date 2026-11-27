IMG=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
cat > /tmp/i29_in.sh <<'EOF'
P=/usr/local/lib/python3.14/site-packages
grep -rn -E "compiler_args|NEURON_CC_FLAGS|COMPILER_OPT|optlevel|\"-O[123]\"|--model-type|--lnc|neuronx-cc|neuronx_cc" --include=*.py $P/sglang_neuron | grep -v /diffusion/ | head -60
echo ===TNX
T=$(python3 -c "import torch_neuronx,os;print(os.path.dirname(torch_neuronx.__file__))" 2>/dev/null); echo $T
grep -rln -E "neuronx-cc|NEURON_CC_FLAGS" --include=*.py $T | head -30
EOF
docker run --rm -v /tmp/i29_in.sh:/i29_in.sh:ro --entrypoint bash $IMG /i29_in.sh
echo ===CACHE
C=~/w/sgl_cache_s4_q06_topk_koff
ls $C; du -sh $C; find $C -maxdepth 3 | head -30; find $C -type f | sed "s/.*\.//" | sort | uniq -c | sort -rn | head
