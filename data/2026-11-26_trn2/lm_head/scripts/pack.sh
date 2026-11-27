#!/usr/bin/env bash
# pack the lmh slot's small files: results (with llmbench raw per-request dirs), correctness, scripts, kit changes
set -u
W=$HOME/w/lmh
cd $W
mkdir -p pack_tmp/kit_changes
cp kit/s4_x_lmh.sh kit/s4_profile.py pack_tmp/kit_changes/
diff -u $HOME/w/sglang_rl/s4_x.sh kit/s4_x_lmh.sh > pack_tmp/kit_changes/s4_x_lmh.diff
diff -u $HOME/w/sglang_rl/s4_profile.py kit/s4_profile.py > pack_tmp/kit_changes/s4_profile.diff
find profiles -type f | sort > pack_tmp/profile_files.txt
du -sh profiles > pack_tmp/profile_size.txt
sudo -n chown -R ubuntu:ubuntu results cc/results 2>/dev/null
tar czf lmh_results.tgz results cc/lmh_cc_collect.py cc/lmh_cc_compare.py cc/results scripts patch pack_tmp
ls -la lmh_results.tgz
sha256sum lmh_results.tgz
echo PACK_DONE
