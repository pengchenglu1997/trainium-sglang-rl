#!/usr/bin/env bash
# lmh slot setup: private copy of the kit with own results dir, cache dirs and container names
set -eu
W=$HOME/w/lmh
mkdir -p $W/results/logs $W/results/server_logs $W/profiles $W/cc/results/logs
rm -rf $W/kit
cp -a $HOME/w/sglang_rl $W/kit
rm -rf $W/kit/__pycache__
cd $W/kit
# s4_x.sh: results, profiles and caches under the lmh slot; container name = tag (tags start with lmh_)
cp s4_x.sh s4_x_lmh.sh
sed -i 's#R="$DATA_ROOT/results_s4"; PROF="$DATA_ROOT/profiles_s4"#R="$HOME/w/lmh/results"; PROF="$HOME/w/lmh/profiles"#' s4_x_lmh.sh
sed -i 's#export SGL_CACHE="$DATA_ROOT/sgl_cache_s4_$tag"#export SGL_CACHE="$DATA_ROOT/sgl_cache_$tag"#' s4_x_lmh.sh
sed -i 's#--name "pair_$tag"#--name "$tag"#' s4_x_lmh.sh
sed -i 's#name=f"prof_{a.tag}"#name=f"{a.tag}_prof"#' s4_profile.py
diff s4_x.sh s4_x_lmh.sh || true
diff $HOME/w/sglang_rl/s4_profile.py s4_profile.py || true
ls $W
echo SETUP_OK
