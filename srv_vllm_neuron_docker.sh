#!/usr/bin/env bash
# vLLM Neuron server in the AWS container (vllm-neuron setup guide, option C), for hosts whose
# AMI has no vLLM environment (the Neuron PyTorch 2.14 DLAMI of 2026-11-20 has none).
#   srv_vllm_neuron_docker.sh start <model dir under $MODELS_ROOT> <host port> <container name>
#   srv_vllm_neuron_docker.sh alive|logs|stop <container name>
# Image: vllm-neuron 0.29.0.1.2.0, SDK 2.33.1. The flags below are the documented ones (setup guide,
# quickstart).
#
# Environment:
#   MODELS_ROOT   host folder mounted read-only at /models
#   VLLM_CACHE    host folder of the compile cache, the same on every start
#   VLLM_IMAGE    default: the 0.29.0.1.2.0 / SDK 2.33.1 image
#   VLLM_DEVICES (16,17,18,19: one logical core per tensor-parallel rank; vLLM Neuron refuses
#   NEURON_RT_VISIBLE_CORES with its multi-process workers, and "4" resolved to one core,
#   both seen 2026-11-20)
#   VLLM_TP (4)VLLM_MAX_SEQS (32)  VLLM_MAX_LEN (1024)  VLLM_EXTRA
#   VLLM_COMPILE_TIMEOUT  seconds (default 1200), set in the container under two names, side by side:
#   VLLM_NEURON_COMPILATION_TIMEOUT (the name the vllm-neuron quickstart exports; vLLM core logs it as unknown) and
#   NEURON_LIBTORCH_COMPILATION_TIMEOUT (the name the vllm-neuron configuration reference lists, "individual NEFF
#   compilation"). Which of the two the compiler reads is UNVERIFIED (the docs name both; no vllm-neuron Python file
#   reads either), so both are set.
#   VLLM_ENV      extra environment variables for the container: "NAME=value NAME2=value2", items separated by spaces.
#                 A value holds no space. Each item is checked to be NAME=value with NAME of letters, digits and
#                 underscores, and is passed as one docker argument (no shell evaluation, no globbing). It comes after the
#                 built-in variables, so it overrides one of the same name. Default empty = nothing added. A bad item
#                 stops the start (exit 2) before anything is started.
#   VLLM_DEV_GLOB (/dev/neuron*) pattern of the device files to pass in; only a test needs to change it
set -u
IMAGE="${VLLM_IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.1-ubuntu26.04}"
cmd="${1:?start|alive|logs|stop}"
case "$cmd" in
  start)
    model="${2:?model dir}"; port="${3:?port}"; name="${4:?name}"
    : "${MODELS_ROOT:?}"
    VLLM_CACHE="${VLLM_CACHE:-$(dirname "$MODELS_ROOT")/vllm_cache_0.29.0.1.2.0}"
    case "$model" in
      "$MODELS_ROOT"/*) inside="/models/${model#"$MODELS_ROOT"/}" ;;
      *) echo "model dir must be under MODELS_ROOT ($MODELS_ROOT)"; exit 2 ;;
    esac
    # VLLM_ENV: split on spaces without globbing, one -e per item
    extra_env=()
    read -r -a env_items <<< "${VLLM_ENV:-}"
    for kv in ${env_items[@]+"${env_items[@]}"}; do
      case "$kv" in
        [A-Za-z_]*=*) ;;
        *) echo "VLLM_ENV item '$kv' is not NAME=value"; exit 2 ;;
      esac
      case "${kv%%=*}" in
        *[!A-Za-z0-9_]*) echo "VLLM_ENV item '$kv': the name has a character other than letters, digits, underscore"; exit 2 ;;
      esac
      extra_env+=(-e "$kv")
    done
    mkdir -p "$VLLM_CACHE"
    devices=()
    for d in ${VLLM_DEV_GLOB:-/dev/neuron*}; do [ -e "$d" ] && devices+=(--device "$d"); done
    [ "${#devices[@]}" -gt 0 ] || { echo "no /dev/neuron* devices"; exit 2; }
    docker rm -f "$name" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "$name" --init --ipc=host --cap-add SYS_NICE "${devices[@]}" \
      -v "$MODELS_ROOT":/models:ro -v "$VLLM_CACHE":/cache \
      -p "127.0.0.1:${port}:8000" \
      -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache -e NEURON_SKIP_EFA_AFFINITY=1 \
      -e VLLM_NEURON_COMPILATION_TIMEOUT="${VLLM_COMPILE_TIMEOUT:-1200}" \
      -e NEURON_LIBTORCH_COMPILATION_TIMEOUT="${VLLM_COMPILE_TIMEOUT:-1200}" -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
      -e NEURON_VISIBLE_DEVICES="${VLLM_DEVICES:-16,17,18,19}" ${extra_env[@]+"${extra_env[@]}"} \
      --entrypoint vllm "$IMAGE" serve "$inside" --served-model-name "$name" \
      --host 0.0.0.0 --port 8000 --tensor-parallel-size "${VLLM_TP:-4}" \
      --max-num-seqs "${VLLM_MAX_SEQS:-32}" --max-model-len "${VLLM_MAX_LEN:-1024}" \
      --no-enable-prefix-caching ${VLLM_EXTRA:-}
    ;;
  alive)
    [ "$(docker inspect -f '{{.State.Running}}' "${2:?name}" 2>/dev/null)" = true ] && echo true || echo false
    ;;
  logs)
    docker logs "${2:?name}" 2>&1
    ;;
  stop)
    name="${2:?name}"
    docker stop -t 120 "$name" >/dev/null 2>&1 || true
    docker rm "$name" >/dev/null 2>&1 || true
    ;;
  *) echo "usage: srv_vllm_neuron_docker.sh start|alive|logs|stop ..."; exit 2 ;;
esac
