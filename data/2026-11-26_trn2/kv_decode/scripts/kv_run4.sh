#!/usr/bin/env bash
# kv slot driver, part 4 (2026-11-26): the 1536-rung geometry on Qwen3-8B (--context-length 3072, rows on the 1536
# rung) and a fresh all-on control right after it (tag q8_ctl3, warm copy of q8_all).
set -u
K="$HOME/w/kv/kit"; R="$HOME/w/kv/results"; mkdir -p "$R/logs"
CORES=16-19; PORT=30453
P="$HOME/w/sglang_rl/patches/sglang_neuron_int_topk"
ALLON="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"
snap() { { echo "## $(date -u +%FT%TZ) $1"; docker ps --format '{{.Names}} {{.Status}} {{.Image}}'; } >> "$R/docker_ps.txt"; }
copy() { [ -d "$HOME/w/sgl_cache_kv_$2" ] || sudo cp -a "$HOME/w/sgl_cache_s4_$1" "$HOME/w/sgl_cache_kv_$2"; }
copy q8_all q8_ctx3k; snap "before bench q8_ctx3k"
KV_EXTRA="--context-length 3072" bash "$K/kv_x.sh" bench q8_ctx3k 8 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q8_ctx3k"
copy q8_all q8_ctl3; snap "before bench q8_ctl3"
KV_EXTRA="" bash "$K/kv_x.sh" bench q8_ctl3 8 temperature $CORES $PORT "$P" "$ALLON"; snap "after bench q8_ctl3"
echo "KV_RUN4_DONE"
