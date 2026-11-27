#!/usr/bin/env bash
# dp_ phase 1: cold compiles (start, ready, stop; no bench) of the three replica layouts, in parallel on chips 1-3 of
# the slot; and the warm cache copies of the two controls (originals untouched).
set -u
D="$HOME/w/dp"
say() { echo "== P1 $(date -u +%H:%M:%S) $*"; }
say "copy caches"
for t in q06_all q8_all; do
  [ -e "$HOME/w/sgl_cache_dp_ctl_$t" ] || sudo -n cp -a "$HOME/w/sgl_cache_s4_$t" "$HOME/w/sgl_cache_dp_ctl_$t"
  say "cache $t: original $(sudo -n find "$HOME/w/sgl_cache_s4_$t" -type f | wc -l) files, copy $(sudo -n find "$HOME/w/sgl_cache_dp_ctl_$t" -type f | wc -l) files"
done
bash "$D/dp_x.sh" start c_dp4_06 dp4_06 06 4-7 30401 "--tp-size 1 --dp-size 4 --max-total-tokens 65536" &
bash "$D/dp_x.sh" start c_dp4_8 dp4_8 8 8-11 30402 "--tp-size 1 --dp-size 4 --max-total-tokens 16384" &
bash "$D/dp_x.sh" start c_dp2_06 dp2_06 06 12-15 30403 "--tp-size 2 --dp-size 2" &
wait
say "phase 1 end"
echo "P1_DONE"
