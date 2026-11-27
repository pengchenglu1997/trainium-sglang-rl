#!/bin/bash
# prof_all copy of devprof/scripts/run_tag.sh: run the earlier analysis scripts on one tag; outputs to ../out/<script>_<tag>.txt
D="<local-path>/results_s4_2026-11-26/prof_all"
p="$1"
cd "$D"
[ -d "parquet/$p" ] || tar xzf "parquet_$p.tgz" -C parquet
cd scripts
for s in g01_graph_walls verify_05_budget verify_06_detail verify_08_phase_engines g02_sampler_scope g04_transpose_where g05_labels g06_log_context; do
  python $s.py $p > ../out/${s}_$p.txt 2>&1 || echo "FAIL $s $p"
done
echo "done $p"
