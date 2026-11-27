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
#   NEURON_LIBTORCH_COMPILATION_TIMEOUT. The two vllm-neuron doc pages that list the second name describe it differently:
#   the configuration reference says "Timeout in seconds for individual NEFF compilation", the compilation-cache page says
#   "Seconds to wait for another process to finish compiling before timing out" (a lock wait). So the meaning of the name is
#   itself unsettled, and which of the two names the compiler reads is UNVERIFIED (the docs name both; no vllm-neuron source
#   was read here), so both are set.
#   VLLM_ENV      extra environment variables for the container: "NAME=value NAME2=value2", items separated by spaces.
#                 A value holds no space. Each item is checked to be NAME=value with NAME of letters, digits and
#                 underscores, and is passed as one docker argument (no shell evaluation, no globbing). It comes after the
#                 built-in variables; that a repeated name then takes the last value is docker behaviour that was NOT RUN here. Default empty = nothing added. A bad item
#                 stops the start (exit 2) before anything is started.
#   VLLM_DEV_GLOB (/dev/neuron*) pattern of the device files to pass in; only a test needs to change it
#   VLLM_NKI_INDEX_CHECK (1) before the start, check $VLLM_CACHE/neuron/compile_cache/nki with nki_index_check.py
#                 (next to this script), run in a throwaway container of the same image with the same mounts. If an
#                 entry names a kernel file that does not exist there (vllm-neuron writes them to the container's own
#                 /var/tmp, gone with that container; a later start can fail with NCC_EVRF059), only that index
#                 folder is removed. With this image every entry names such a file, so in practice any non-empty
#                 index is removed. When the check is on, exactly one line starting "nki index check:" says what
#                 was done (also for a missing index folder or a missing helper file). Skipped (and said so)
#                 while another running container mounts the cache folder, a folder inside it or a parent of it.
#                 Exactly 0 = no check (any other value leaves it on).
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
    if [ "${VLLM_NKI_INDEX_CHECK:-1}" != 0 ]; then
      helper="$(dirname "$(readlink -f "$0")")/nki_index_check.py"
      if [ ! -f "$helper" ]; then
        echo "nki index check: skipped, helper not found at $helper"
      else
        # a running container uses the cache when one of its mount sources is the cache folder, inside it,
        # or a parent of it (one source per line, so paths with spaces stay whole)
        cache_real="$(realpath -m "$VLLM_CACHE")"; users=""
        for c in $(docker ps -q); do
          while IFS= read -r src; do
            [ -n "$src" ] || continue
            s="$(realpath -m "$src" 2>/dev/null || printf '%s' "$src")"
            case "$s/" in "$cache_real"/*) users="$users $c"; break ;; esac
            p="${s%/}"  # "" for a mount source of /; the pattern "${s%/}"/* does not match then, "$p"/* does
            case "$cache_real/" in "$p"/*) users="$users $c"; break ;; esac
          done < <(docker inspect -f '{{range .Mounts}}{{println .Source}}{{end}}' "$c")
        done
        if [ -n "$users" ]; then
          echo "nki index check: skipped, cache folder in use by running container(s):$users"
        else
          # the existence test of the index folder is done by the helper, as root in the container
          docker run --rm -v "$MODELS_ROOT":/models:ro -v "$VLLM_CACHE":/cache \
            --mount "type=bind,src=$helper,dst=/nki_index_check.py,readonly" \
            --entrypoint python3 "$IMAGE" /nki_index_check.py /cache/neuron/compile_cache/nki
          rc=$?
          [ "$rc" -eq 0 ] || echo "nki index check: the check container failed (exit $rc); index state unknown, see the output above"
        fi
      fi
    fi
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
