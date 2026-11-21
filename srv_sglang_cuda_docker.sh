#!/usr/bin/env bash
# Standard SGLang on a CUDA machine, in the project's own container (lmsysorg/sglang).
#   srv_sglang_cuda_docker.sh start <model dir under $MODELS_ROOT> <host port> <container name>
#   srv_sglang_cuda_docker.sh alive|logs|stop <container name>
# Why a container: on the AWS PyTorch GPU AMI of 2026-11-20 a pip install of sglang 0.5.23 did not
# start (no CUDA home, then no ninja, then nvcc 13.5 against 13.0 headers in flashinfer's JIT build).
# MODELS_ROOT is mounted at the SAME path inside, so a checkpoint path given to
# /update_weights_from_disk means the same folder on both sides.
# Environment: MODELS_ROOT, SGL_CUDA_IMAGE (default below), SGLANG_EXTRA (extra launch flags).
set -u
IMAGE="${SGL_CUDA_IMAGE:-lmsysorg/sglang:latest}"
cmd="${1:?start|alive|logs|stop}"
case "$cmd" in
  start)
    model="${2:?model dir}"; port="${3:?port}"; name="${4:?name}"
    : "${MODELS_ROOT:?}"
    case "$model" in "$MODELS_ROOT"/*) ;; *) echo "model dir must be under MODELS_ROOT ($MODELS_ROOT)"; exit 2 ;; esac
    docker rm -f "$name" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "$name" --device nvidia.com/gpu=all --ipc=host --shm-size 16g \
      -v "$MODELS_ROOT":"$MODELS_ROOT" -p "127.0.0.1:${port}:30000" -e HF_HUB_OFFLINE=1 \
      "$IMAGE" python3 -m sglang.launch_server --model-path "$model" --host 0.0.0.0 --port 30000 \
      --context-length 4096 --dtype bfloat16 ${SGLANG_EXTRA:-}
    ;;
  alive)
    [ "$(docker inspect -f '{{.State.Running}}' "${2:?name}" 2>/dev/null)" = true ] && echo true || echo false
    ;;
  logs) docker logs "${2:?name}" 2>&1 ;;
  stop)
    name="${2:?name}"
    docker stop -t 60 "$name" >/dev/null 2>&1 || true
    docker rm "$name" >/dev/null 2>&1 || true
    ;;
  *) echo "usage: srv_sglang_cuda_docker.sh start|alive|logs|stop ..."; exit 2 ;;
esac
