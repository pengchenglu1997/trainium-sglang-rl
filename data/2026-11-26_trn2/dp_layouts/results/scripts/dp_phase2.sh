#!/usr/bin/env bash
# dp_ phase 2: Qwen3-0.6B benches back to back on ONE chip (cores 0-3), each start warm from its own cache:
#   ctl_a (TP4 all-on, copy of q06_all) -> dp4 (--dp-size 4 of TP1) -> dp2 (--dp-size 2 of TP2) -> ctl_b (TP4 again)
set -u
D="$HOME/w/dp"
say() { echo "== P2 $(date -u +%H:%M:%S) $*"; }
say "phase 2 start"
bash "$D/dp_x.sh" bench ctl06_a ctl_q06_all 06 0-3 30410 ""
bash "$D/dp_x.sh" bench dp4_06 dp4_06 06 0-3 30411 "--tp-size 1 --dp-size 4 --max-total-tokens 65536"
if [ "${SKIP_DP2:-0}" != 1 ]; then
  bash "$D/dp_x.sh" bench dp2_06 dp2_06 06 0-3 30412 "--tp-size 2 --dp-size 2"
fi
bash "$D/dp_x.sh" bench ctl06_b ctl_q06_all 06 0-3 30413 ""
say "phase 2 end"
echo "P2_DONE"
