#!/usr/bin/env bash
# kv slot driver, part 2 (2026-11-26): decode K-cache length by flag. Control = all-on as q06_all / q8_all
# (--context-length 4096, one decode context graph of 4096 tokens since the integration host.py has
# decode_context_ladder False). Variant = same + --context-length 2048 (appended last, so it wins): the decode
# graph covers 2048 tokens; llmbench rows are 1024 + 256 = 1280 tokens, so they fit.
# Steps: c06b (control repeat, tag q06_ctl2) L06 (q06_ctx2k) c8b (q8_ctl2) L8 (q8_ctx2k)
set -u
K="$HOME/w/kv/kit"; R="$HOME/w/kv/results"; mkdir -p "$R/logs"
CORES=16-19; PORT=30451
P="$HOME/w/sglang_rl/patches/sglang_neuron_int_topk"
ALLON="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
snap() { { echo "## $(date -u +%FT%TZ) $1"; docker ps --format '{{.Names}} {{.Status}} {{.Image}}'; } >> "$R/docker_ps.txt"; }
copy() { [ -d "$HOME/w/sgl_cache_kv_$2" ] || sudo cp -a "$HOME/w/sgl_cache_s4_$1" "$HOME/w/sgl_cache_kv_$2"; }
for step in "$@"; do
  case "$step" in
    c06b) copy q06_all q06_ctl2; snap "before bench q06_ctl2"
          KV_EXTRA="" bash "$K/kv_x.sh" bench q06_ctl2 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q06_ctl2" ;;
    L06)  copy q06_all q06_ctx2k; snap "before bench q06_ctx2k"
          KV_EXTRA="--context-length 2048" bash "$K/kv_x.sh" bench q06_ctx2k 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q06_ctx2k" ;;
    c8b)  copy q8_all q8_ctl2; snap "before bench q8_ctl2"
          KV_EXTRA="" bash "$K/kv_x.sh" bench q8_ctl2 8 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q8_ctl2" ;;
    L8)   copy q8_all q8_ctx2k; snap "before bench q8_ctx2k"
          KV_EXTRA="--context-length 2048" bash "$K/kv_x.sh" bench q8_ctx2k 8 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q8_ctx2k" ;;
    *) echo "unknown step $step" ;;
  esac
done
echo "KV_RUN2_DONE $*"
