#!/usr/bin/env bash
# kv slot driver, part 3 (2026-11-26): decode K-cache block geometry by flag. The decode kernel shrinks the 128-token
# page to gcd(128, rung / (shards * 128)) tokens per block (nkilib attention_tkg_utils.py resize), and the decode
# context rungs are context/4, /2, /1 rounded up to 256 (graph_runner BucketConfig.build). llmbench rows are
# 1025..1280 tokens during decode.
#   control  --context-length 4096  -> rows on the 2048 rung (blocks of 8 or 16 tokens)
#   C3k      --context-length 3072  -> rows on the 1536 rung (25 % fewer prior bytes, smaller blocks)
#   C16k     --context-length 16384 -> rows on the 4096 rung (2x prior bytes, blocks of 16 or 32 tokens)
# Steps: c06c (control, tag q06_ctl3) C3k (q06_ctx3k) C16k (q06_ctx16k)
set -u
K="$HOME/w/kv/kit"; R="$HOME/w/kv/results"; mkdir -p "$R/logs"
CORES=16-19; PORT=30452
P="$HOME/w/sglang_rl/patches/sglang_neuron_int_topk"
ALLON="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
snap() { { echo "## $(date -u +%FT%TZ) $1"; docker ps --format '{{.Names}} {{.Status}} {{.Image}}'; } >> "$R/docker_ps.txt"; }
copy() { [ -d "$HOME/w/sgl_cache_kv_$2" ] || sudo cp -a "$HOME/w/sgl_cache_s4_$1" "$HOME/w/sgl_cache_kv_$2"; }
for step in "$@"; do
  case "$step" in
    c06c) copy q06_all q06_ctl3; snap "before bench q06_ctl3"
          KV_EXTRA="" bash "$K/kv_x.sh" bench q06_ctl3 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q06_ctl3" ;;
    C3k)  copy q06_all q06_ctx3k; snap "before bench q06_ctx3k"
          KV_EXTRA="--context-length 3072" bash "$K/kv_x.sh" bench q06_ctx3k 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q06_ctx3k" ;;
    C16k) copy q06_all q06_ctx16k; snap "before bench q06_ctx16k"
          KV_EXTRA="--context-length 16384" bash "$K/kv_x.sh" bench q06_ctx16k 06 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q06_ctx16k" ;;
    *) echo "unknown step $step" ;;
  esac
done
echo "KV_RUN3_DONE $*"
