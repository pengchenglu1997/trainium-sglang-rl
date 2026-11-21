#!/usr/bin/env bash
# SGLang-Neuron server in the release container (sgl-project/sglang-neuron v0.1.0, public preview).
#   srv_sglang_neuron.sh start <model dir under $MODELS_ROOT> <host port> <container name>
#   srv_sglang_neuron.sh alive <container name>      prints true or false
#   srv_sglang_neuron.sh logs  <container name>
#   srv_sglang_neuron.sh stop  <container name>
# The docker command is the one of the repo's docs/guides/container.md with the example script
# examples/qwen3_0p6b/run_server.sh (TP4, context 4096, 64 running requests). Changed here: the
# sampler (SGLANG_NEURON_SAMPLER, greedy in the docs) and the cores. The docs validate the
# language models on Trn3 only; on trn2 this start may be refused.
# Environment:
#   MODELS_ROOT   host folder mounted read-only at /models (base model and merged checkpoints)
#   SGL_CACHE     host folder of the compile cache, the same on every start
#   SGL_IMAGE     default: the v0.1.0 release image
#   SGL_SAMPLER   greedy | temperature | topk_topp | host    (default temperature)
#   SGL_CORES     NEURON_RT_VISIBLE_CORES of the server (default 0-3)
#   SGL_TOP_LOGPROBS  SGLANG_NEURON_MAX_TOP_LOGPROBS (default unset = 0)
#   SGL_SCRIPT    script inside the image (default the Qwen3-0.6B example)
#   SGL_EXTRA     extra launch_server flags appended (changes the compiled shapes)
set -u
IMAGE="${SGL_IMAGE:-public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04}"
cmd="${1:?start|alive|logs|stop}"
case "$cmd" in
  start)
    model="${2:?model dir}"; port="${3:?port}"; name="${4:?name}"
    : "${MODELS_ROOT:?}" "${SGL_CACHE:?}"
    case "$model" in
      "$MODELS_ROOT"/*) inside="/models/${model#"$MODELS_ROOT"/}" ;;
      *) echo "model dir must be under MODELS_ROOT ($MODELS_ROOT)"; exit 2 ;;
    esac
    mkdir -p "$SGL_CACHE"
    devices=()
    for d in /dev/neuron*; do [ -e "$d" ] && devices+=(--device "$d"); done
    [ "${#devices[@]}" -gt 0 ] || { echo "no /dev/neuron* devices"; exit 2; }
    extra_env=()
    [ -n "${SGL_TOP_LOGPROBS:-}" ] && extra_env+=(-e "SGLANG_NEURON_MAX_TOP_LOGPROBS=$SGL_TOP_LOGPROBS")
    # SGL_PATCH_DIR: a folder with experimental replacements of plugin files (host.py, general_plugin.py),
    # mounted read-only over the installed ones. Unset = the image as released.
    patch=()
    if [ -n "${SGL_PATCH_DIR:-}" ]; then
      pkg=/usr/local/lib/python3.14/site-packages/sglang_neuron
      # every .py of the folder replaces the file of the same relative path in the package (orig/ is left out)
      while IFS= read -r rel; do
        patch+=(-v "$SGL_PATCH_DIR/$rel:$pkg/$rel:ro")
      done < <(cd "$SGL_PATCH_DIR" && find . -name "*.py" -not -path "./orig/*" -not -path "*/__pycache__/*" | sed 's#^\./##' | sort)
    fi
    # SGL_ENV: extra environment for the server, "K=V K2=V2" (switches of the experimental patch)
    for kv in ${SGL_ENV:-}; do extra_env+=(-e "$kv"); done
    docker rm -f "$name" >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    docker run -d --name "$name" --init --ipc=host --cap-add SYS_NICE "${devices[@]}" \
      -v "$MODELS_ROOT":/models:ro -v "$SGL_CACHE":/cache ${patch[@]+"${patch[@]}"} \
      -p "127.0.0.1:${port}:30000" -e HOST=0.0.0.0 -e PORT=30000 -e MODEL_PATH="$inside" \
      -e HF_HUB_OFFLINE=1 \
      -e SGLANG_NEURON_SAMPLER="${SGL_SAMPLER:-temperature}" \
      -e NEURON_RT_VISIBLE_CORES="${SGL_CORES:-0-3}" ${extra_env[@]+"${extra_env[@]}"} \
      "$IMAGE" "${SGL_SCRIPT:-/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh}" ${SGL_EXTRA:-}
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
  *) echo "usage: srv_sglang_neuron.sh start|alive|logs|stop ..."; exit 2 ;;
esac
