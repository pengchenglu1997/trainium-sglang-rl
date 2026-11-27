#!/bin/bash
for t in q06_all q06_topk_all q8_all; do bash ~/w/r3prof/r3_conv.sh $t & done
wait
echo ALLDONE $(date -u +%FT%TZ)
