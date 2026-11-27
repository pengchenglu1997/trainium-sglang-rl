#!/usr/bin/env bash
# Raw outputs for lessons C15 and C12: image code excerpts with line numbers, and a read-only look at one running
# SGLang-Neuron container (env of its processes, /var/tmp, docker inspect env). Writes into ~/w/i12/results/c15/.
set -u
O=/home/ubuntu/w/i12/results/c15; mkdir -p $O
SGL=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
VL=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.1-ubuntu26.04
cat > $O/excerpt_sgl.sh <<'EOS'
SP=/usr/local/lib/python3.14/site-packages
show() { echo "=== $1 lines $2-$3"; awk -v a=$2 -v b=$3 'NR>=a && NR<=b {printf "%d: %s\n", NR, $0}' "$1"; }
show /workspace/torch_neuronx/torch_neuronx/__init__.py 160 172
show /workspace/torch_neuronx/torch_neuronx/nki_kernel.py 1 30
show /workspace/torch_neuronx/torch_neuronx/nki_kernel.py 49 90
show $SP/nki/compiler/_trace_cache.py 254 312
show $SP/nki/_config.py 180 200
show $SP/nki/_config.py 436 470
show $SP/nki/framework/compiled.py 315 330
show $SP/nki/compiler/_disk_cache.py 60 80
show /opt/sglang-neuron/sglang_neuron/backend/envs.py 1 20
show /opt/sglang-neuron/sglang_neuron/backend/stack.py 276 300
echo "=== stack.py /opt vs site-packages"; cmp /opt/sglang-neuron/sglang_neuron/backend/stack.py $SP/sglang_neuron/backend/stack.py && echo SAME
echo "=== nki version"; pip show nki 2>/dev/null | head -2
EOS
cat > $O/excerpt_vllm.sh <<'EOS'
SP=/opt/conda/lib/python3.14/site-packages
show() { echo "=== $1 lines $2-$3"; awk -v a=$2 -v b=$3 'NR>=a && NR<=b {printf "%d: %s\n", NR, $0}' "$1"; }
L=$SP/libtorch_neuronx_lite
show $L/_compiler/__init__.py 148 156
show $L/_compiler/nki_kernel.py 15 30
show $L/nki/nki_hop.py 600 640
show $L/nki/nki_cache.py 36 40
show $L/nki/nki_cache.py 97 190
show $L/nki/nki_cache.py 256 262
show $L/nki/nki_cache.py 335 370
show $L/envs.py 105 114
show $L/envs.py 186 196
show $SP/nki/compiler/_disk_cache.py 60 80
show $SP/nki/compiler/_trace_cache.py 254 312
EOS
docker run --rm --name i12_inspect -v $O/excerpt_sgl.sh:/x.sh:ro --entrypoint sh $SGL /x.sh > $O/sgl_image_code_excerpts.txt 2>&1
docker run --rm --name i12_inspect -v $O/excerpt_vllm.sh:/x.sh:ro --entrypoint sh $VL /x.sh > $O/vllm_image_code_excerpts.txt 2>&1
C=$(docker ps --filter ancestor=$SGL --format '{{.Names}}' | head -1)
{
  echo "container: $C (read-only: docker inspect, docker exec of ls and cat /proc/<pid>/environ)"; date -u
  echo "=== docker inspect env (NKI, TRACE, NEURON_RT_VISIBLE)"; docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$C" | grep -E 'NKI|TRACE|NEURON_RT_VISIBLE'
  echo "=== ls -la /var/tmp"; docker exec "$C" ls -la /var/tmp
  echo "=== NKI variables in /proc/<pid>/environ of processes matching sglang"
  docker exec "$C" sh -c 'for p in $(pgrep -f sglang); do echo "pid $p $(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -E "^NKI_" | sort | tr "\n" " ")"; done'
  echo "=== trace-cache warning lines in docker logs"; docker logs "$C" 2>&1 | grep -n 'NKI_ENABLE_TRACE_CACHE=1' | cut -c1-120 | head -8
} > $O/sgl_running_container.txt 2>&1
ls -la $O
