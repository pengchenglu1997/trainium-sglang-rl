#!/usr/bin/env bash
# i34 extra starts after drive.sh: d = old launcher on a copy of the 8B-passed folder (control for c),
# e = repeat of b (new launcher on a fresh copy of the stale folder).
set -u
W=/home/ubuntu/w/i34; R=$W/results
until grep -q "all done" $W/drive.log; do sleep 10; done
echo "start2 $(date -u +%T)"
sudo cp -a /home/ubuntu/w/i12_cache_p4 $W/cache_d
sudo cp -a /home/ubuntu/w/i12_cache_p2 $W/cache_e
for c in d e; do echo "cache_$c files=$(sudo find $W/cache_$c -type f | wc -l)"; done >> $R/copies.txt
bash $W/run_one.sh i34_d_old $W/cache_d old 30323 1500; echo "d done $(date -u +%T)"
bash $W/run_one.sh i34_e_new $W/cache_e new 30324 1500; echo "e done $(date -u +%T)"
echo "all2 done $(date -u +%T)"
