cd ~/w/results_s4
ls | head -60
for t in q8_main q06_main q8_all q06_lmh_on; do
  for f in $(ls | grep "$t" | grep -v bench | head -3); do
    echo "## $t $f"
    grep -o '"ready_s": [0-9.]*' "$f" | head -2
  done
done
nproc; uptime
