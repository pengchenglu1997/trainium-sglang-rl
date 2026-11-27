cd ~/w/lmh/results/logs
for f in phaseB.log phaseC.log; do
  awk '/^-- docker ps/ {h=$0; next} /^-- load/ {print h " | " $0 " | others: " o; o=""; next} /^==/ {next} {if ($1 !~ /^lmh_/) o=o $1 " "}' $f
done
