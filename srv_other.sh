#!/usr/bin/env bash
# Foreground servers for the comparison runs. Each runs until it gets SIGTERM.
#   srv_other.sh vllm_neuron <model dir> <port> <served name>
#   srv_other.sh vllm_cuda   <model dir> <port> <served name>
#   srv_other.sh sglang_cuda <model dir> <port>
#
# vllm_neuron: the DLAMI's vLLM Neuron environment (vllm-neuron docs, setup guide, option B).
#   Qwen3-0.6B is not in the vllm-neuron model list; it was run elsewhere at TP 1/2/4/8,
#   but the exact command was not available. The command below is put
#   together from the documented flags and is NOT a tested recipe: confirm it with the teammate
#   who ran it, or expect to adjust VLLM_EXTRA.
#   VLLM_TP (default 1), VLLM_CORES (NEURON_RT_VISIBLE_CORES, default 16-19, away from the
#   SGLang server and the trainer), VLLM_MAX_SEQS (default 32), VLLM_MAX_LEN (default 1024).
set -u
kind="${1:?vllm_neuron|vllm_cuda|sglang_cuda}"; model="${2:?model dir}"; port="${3:?port}"; name="${4:-rl}"
case "$kind" in
  vllm_neuron)
    venv=$(ls -d /opt/aws_neuronx_venv_pytorch_inference_vllm_* 2>/dev/null | head -1)
    [ -n "$venv" ] || { echo "no vLLM Neuron environment under /opt"; exit 2; }
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export NEURON_RT_VISIBLE_CORES="${VLLM_CORES:-16-19}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/w/vllm_cache}"
    export NEURON_LIBTORCH_COMPILATION_TIMEOUT=1200 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
    # shellcheck disable=SC2086
    exec vllm serve "$model" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
      --tensor-parallel-size "${VLLM_TP:-1}" --max-num-seqs "${VLLM_MAX_SEQS:-32}" \
      --max-model-len "${VLLM_MAX_LEN:-1024}" --no-enable-prefix-caching ${VLLM_EXTRA:-}
    ;;
  vllm_cuda)
    # shellcheck disable=SC2086
    exec vllm serve "$model" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
      --max-model-len "${VLLM_MAX_LEN:-4096}" --dtype bfloat16 ${VLLM_EXTRA:-}
    ;;
  sglang_cuda)
    # shellcheck disable=SC2086
    # SRV_PY: the Python of the environment that holds sglang (its own venv on the GPU machine)
    exec "${SRV_PY:-python}" -m sglang.launch_server --model-path "$model" --host 127.0.0.1 --port "$port" \
      --context-length 4096 --dtype bfloat16 ${SGLANG_EXTRA:-}
    ;;
  *) echo "usage: srv_other.sh vllm_neuron|vllm_cuda|sglang_cuda <model dir> <port> [name]"; exit 2 ;;
esac
