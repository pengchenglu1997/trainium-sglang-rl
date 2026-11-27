#!/usr/bin/env bash
# waits for the first cold start, stops its poller, then runs the two follow-up cold starts one after the other
W=$HOME/w/i29
until grep -q I29_COLD_DONE $W/i29_cold.log; do sleep 10; done
touch $W/poll.stop
echo "== $(date -u +%T) first run done"
bash $W/i29_cold2.sh i29_q06_temp_cold temperature
echo "== $(date -u +%T) temp done"
bash $W/i29_cold2.sh i29_q06_topk_cold2 topk_topp
echo "== $(date -u +%T) topk2 done"
echo CHAIN_DONE
