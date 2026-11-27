#!/usr/bin/env bash
# waits for part 1 (kv_run.log ends with KV_RUN_DONE), then runs part 2
until grep -q KV_RUN_DONE "$HOME/w/kv/results/kv_run.log"; do sleep 20; done
bash "$HOME/w/kv/kit/kv_run2.sh" c06b L06 c8b L8 > "$HOME/w/kv/results/kv_run2.log" 2>&1
