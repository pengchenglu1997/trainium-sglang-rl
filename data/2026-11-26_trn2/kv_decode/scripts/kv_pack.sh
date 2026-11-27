#!/usr/bin/env bash
# kv slot: collect scripts, summaries and the profile tgz into ~/w/kv/results, then SHA256SUMS
set -u
R="$HOME/w/kv/results"; K="$HOME/w/kv/kit"
mkdir -p "$R/scripts"
cp "$K"/kv_*.sh "$K"/kv_*.py "$K/s4_profile.py" "$R/scripts/"
cp "$HOME/w/kv/kv_profiles_q06_p2k_prof.tgz" "$R/"
python3 "$K/kv_summary.py" "$R" > "$R/kv_summary.txt"
{
  for f in "$R"/server_logs/*.log; do
    echo "== $(basename "$f")"
    echo "chunk_num_blocks warnings: $(grep -c 'not a multiple of 16' "$f")"
    grep -o 'chunk_num_blocks=[0-9]*' "$f" | sort | uniq -c
    grep 'TP0.*warm decode' "$f" | sed 's/.*warm decode/warm decode/' | tr '\n' ';'; echo
    grep 'TP0.*warm prefill' "$f" | sed 's/.*warm prefill/warm prefill/' | sort -u | tr '\n' ';'; echo
    grep -o 'Token-gen bucket length of [0-9]*: reducing block length by [0-9]*x, cache block length reduced from 128 to [0-9]*' "$f" | sort | uniq -c
  done
} > "$R/kv_server_log_facts.txt"
for a in ctl2 p2k ctx2k ctx3k ctx16k ctl3; do python3 "$K/kv_tokcmp.py" "$R/tokdump_b_q06_ctl.json" "$R/tokdump_b_q06_$a.json"; done > "$R/kv_tokcmp_q06.txt" 2>&1
for a in ctl2 p2k ctx2k ctx3k ctl3; do python3 "$K/kv_tokcmp.py" "$R/tokdump_b_q8_ctl.json" "$R/tokdump_b_q8_$a.json"; done > "$R/kv_tokcmp_q8.txt" 2>&1
cd "$R" && find . -type f ! -name SHA256SUMS | sort | xargs -d '\n' sha256sum > SHA256SUMS
wc -l SHA256SUMS; du -sh "$R"
