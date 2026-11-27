#!/usr/bin/env bash
cd /home/ubuntu/w/i12
R=results
sudo python3 -I nki_entries.py /home/ubuntu/w/i12_cache_r1b > $R/nki_entries_after_q06.txt
sudo python3 -I nki_entries.py /home/ubuntu/w/i12_cache_s1a > $R/nki_entries_s1a_8b.txt
head -1 $R/nki_entries_after_q06.txt; head -1 $R/nki_entries_s1a_8b.txt
echo "0.6B entries naming rotational_topk or gen_mask_tkg:"; grep -E "rotational_topk|gen_mask_tkg" $R/nki_entries_after_q06.txt | head -6
cut -d' ' -f1 $R/nki_entries_after_q06.txt | grep json | sort > $R/keys_q06.txt
cut -d' ' -f1 $R/nki_entries_s1a_8b.txt | grep json | sort > $R/keys_8b.txt
echo "entry keys present in both the 0.6B folder and the 8B folder: $(comm -12 $R/keys_q06.txt $R/keys_8b.txt | wc -l)"
comm -12 $R/keys_q06.txt $R/keys_8b.txt > $R/keys_shared.txt
for k in $(head -50 $R/keys_shared.txt); do grep "^$k " $R/nki_entries_after_q06.txt; done | cut -d' ' -f2 | sort | uniq -c
