#!/usr/bin/env bash
# i34 driver: copies, logic check of the helper, then hardware starts a, b, c one after another on cores 8-11.
set -u
W=/home/ubuntu/w/i34; R=$W/results; mkdir -p $R/logs
echo "start $(date -u +%T)"
# own copies of caches; originals are only read
sudo cp -a /home/ubuntu/w/i12_cache_p2 $W/cache_a
sudo cp -a /home/ubuntu/w/i12_cache_p2 $W/cache_b
sudo cp -a /home/ubuntu/w/i12_cache_p4 $W/cache_c
for c in a b c; do echo "cache_$c files=$(sudo find $W/cache_$c -type f | wc -l) du=$(sudo du -sh $W/cache_$c | cut -f1)"; done > $R/copies.txt
for s in i12_cache_p2 i12_cache_p4; do echo "$s files=$(sudo find /home/ubuntu/w/$s -type f | wc -l)"; done >> $R/copies.txt
(cd /home/ubuntu/w && sudo find i12_cache_p2 -type f -printf '%P %s\n' | sort) > $R/p2_listing.txt
(cd $W && sudo find cache_a -type f -printf '%P %s\n' | sort) > $R/cache_a_listing_before.txt
cat $R/copies.txt
# logic check of keep/remove branches, in the server image, no Neuron device
IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.1-ubuntu26.04
rm -rf $W/unit; e=$(sudo ls $W/cache_a/neuron/compile_cache/nki | head -1)
sudo cat $W/cache_a/neuron/compile_cache/nki/$e > $W/entry.json
python3 $W/make_unit.py $W/entry.json $W/unit
for case in keep remove; do
  docker run --rm -v $W/unit/$case:/cache -v $W/new/nki_index_check.py:/nki_index_check.py:ro --entrypoint python3 $IMAGE /nki_index_check.py /cache/neuron/compile_cache/nki
  echo "$case: index dir after: $(ls $W/unit/$case/neuron/compile_cache/nki 2>&1 | tr '\n' ' ')"
done > $R/unit_check.txt 2>&1
cat $R/unit_check.txt
bash $W/run_one.sh i34_a_old $W/cache_a old 30320 900;  echo "a done $(date -u +%T)"
bash $W/run_one.sh i34_b_new $W/cache_b new 30321 1500; echo "b done $(date -u +%T)"
bash $W/run_one.sh i34_c_new $W/cache_c new 30322 1500; echo "c done $(date -u +%T)"
echo "all done $(date -u +%T)"
