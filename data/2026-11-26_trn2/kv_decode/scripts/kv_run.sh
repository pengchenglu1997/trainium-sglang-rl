#!/usr/bin/env bash
# kv slot driver (2026-11-26): control (all-on, as q06_all / q8_all) vs variant (same + --cuda-graph-bs-prefill 2048,
# which leaves only the 2048-token prefill bucket: chunk_num_blocks = 2048/128 = 16, the K-cache fast path).
# Steps given as arguments, run in order: c06 v06 c8 v8 pv06
set -u
K="$HOME/w/kv/kit"; R="$HOME/w/kv/results"; mkdir -p "$R/logs"
CORES=16-19; PORT=30450
P="$HOME/w/sglang_rl/patches/sglang_neuron_int_topk"
ALLON="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
snap() { { echo "## $(date -u +%FT%TZ) $1"; docker ps --format '{{.Names}} {{.Status}} {{.Image}}'; } >> "$R/docker_ps.txt"; }
copy() { [ -d "$HOME/w/sgl_cache_kv_$2" ] || sudo cp -a "$HOME/w/sgl_cache_s4_$1" "$HOME/w/sgl_cache_kv_$2"; }
for step in "$@"; do
  case "$step" in
    c06) copy q06_all q06_ctl; snap "before bench q06_ctl"
         KV_EXTRA="" bash "$K/kv_x.sh" bench q06_ctl 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q06_ctl" ;;
    v06) copy q06_all q06_p2k; snap "before bench q06_p2k"
         KV_EXTRA="--cuda-graph-bs-prefill 2048" bash "$K/kv_x.sh" bench q06_p2k 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q06_p2k" ;;
    c8)  copy q8_all q8_ctl; snap "before bench q8_ctl"
         KV_EXTRA="" bash "$K/kv_x.sh" bench q8_ctl 8 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q8_ctl" ;;
    v8)  copy q8_all q8_p2k; snap "before bench q8_p2k"
         KV_EXTRA="--cuda-graph-bs-prefill 2048" bash "$K/kv_x.sh" bench q8_p2k 8 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q8_p2k" ;;
    pv06) copy q06_all q06_p2k_prof; snap "before prof q06_p2k_prof"
         KV_EXTRA="--cuda-graph-bs-prefill 2048" bash "$K/kv_x.sh" prof q06_p2k_prof 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after prof q06_p2k_prof" ;;
    *) echo "unknown step $step" ;;
  esac
done
echo "KV_RUN_DONE $*"
