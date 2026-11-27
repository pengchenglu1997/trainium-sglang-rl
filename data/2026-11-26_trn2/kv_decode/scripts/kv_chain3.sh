#!/usr/bin/env bash
# waits for part 2 (kv_run2.log ends with KV_RUN2_DONE), then runs part 3
until grep -q KV_RUN2_DONE "$HOME/w/kv/results/kv_run2.log" 2>/dev/null; do sleep 20; done
bash "$HOME/w/kv/kit/kv_run3.sh" c06c C3k C16k > "$HOME/w/kv/results/kv_run3.log" 2>&1
