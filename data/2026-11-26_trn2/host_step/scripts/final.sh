#!/usr/bin/env bash
# Collect summaries into ~/w/host/results/summary, copy scripts, gzip traces, write SHA256SUMS.
set -u
H=$HOME/w/host; R=$H/results; S=$R/summary; mkdir -p "$S" "$R/scripts"
cd "$H" || exit 1
cp scripts/*.sh scripts/*.py "$R/scripts/"; cp p1.log p1b.log p2.log p3.log p4.log "$R/" 2>/dev/null
tags=$(ls "$R"/llmbench_*.csv | sed 's#.*/llmbench_##; s#[.]csv$##' | sort)
python3 scripts/tab.py "$R" $tags > "$S/llmbench_table.txt"
python3 scripts/ready_s.py "$R" > "$S/ready_s.txt"
python3 scripts/dl_sum.py "$R/server_logs/host_p1m.log" "$R/events.jsonl" p1m_t0_p1 p1m_t0_p2 > "$S/decode_log_p1m.txt"
for t in p1t:p1t_t0_p1:p1t_t0_p2 p8t:p8t_t0_p1:p8t_t0_p2 p3t:p3t_t0_p1; do
  d=${t%%:*}; rest=${t#*:}; set -- $(echo "$rest" | tr ':' ' ')
  python3 scripts/ht_sum.py "$R/htrace/$d/host_rank0.tsv" "$R/events.jsonl" "$@" > "$S/trace_sum_$d.txt"
  python3 scripts/ht_cell.py "$R/htrace/$d/host_rank0.tsv" "$R/events.jsonl" "$@" > "$S/trace_cell_$d.txt"
done
python3 scripts/ht_peek.py "$R/htrace/p1t/host_rank0.tsv" 40 30 40 > "$S/trace_peek_p1t_prefill40_conc1.txt"
python3 scripts/ht_peek.py "$R/htrace/p1t/host_rank0.tsv" 160 12 40 > "$S/trace_peek_p1t_prefill160_conc32.txt"
python3 scripts/rollout_steps.py "$HOME/w/results_s4/server_logs/server_prof_q06_all_001.log" \
  "$HOME/w/results_s4/server_logs/server_prof_q06_lmh_on_nd_001.log" "$HOME/w/results_s4/server_logs/server_prof_q06_main_nd_001.log" \
  "$HOME/w/results_s4/server_logs/server_prof_q06_rel_nd_001.log" "$HOME/w/results_s4/server_logs/server_prof_q8_main_nd_001.log" \
  "$HOME/w/results_s4/server_logs/server_prof_q8_all_001.log" "$R/server_logs/host_p1c.log" "$R/server_logs/host_p8t.log" > "$S/rollout_vs_llmbench_32row_steps.txt"
grep -h "Token-gen bucket length" "$R/server_logs/host_p3v7.log" | sort | uniq -c > "$S/p3v7_token_gen_buckets.txt"
grep -h "Token-gen bucket length" "$R/server_logs/host_p3c.log" | sort | uniq -c > "$S/p3c_token_gen_buckets.txt"
for d in p1t p8t p3t; do gzip -kf "$R/htrace/$d/host_rank0.tsv"; done
cd "$R" && find . -type f ! -name SHA256SUMS ! -name '.t0_*' ! -path './htrace/*/host_rank0.tsv' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
echo "FINAL_DONE files=$(wc -l < SHA256SUMS)"
