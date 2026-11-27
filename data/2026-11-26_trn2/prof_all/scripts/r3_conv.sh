#!/bin/bash
# r3prof: convert one tag's raw device profile to parquet (same command as prof_work/conv_one.sh), writes only under ~/w/r3prof.
t="$1"
W=~/w/r3prof
mkdir -p $W/parquet $W/logs
src=~/w/profiles_s4/$t/serve_rank0
log=$W/logs/conv_$t.log
{
echo "START $t $(date -u +%FT%TZ)"
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
echo "neuron-explorer=$(which neuron-explorer) $(neuron-explorer --version 2>&1 | head -1)"
D=$(ls -d $src/i-* 2>/dev/null | head -1)
N=$(find $src/neffs -name '*.neff' ! -name 'barrier_*' | head -1)
NN=$(find $src/neffs -name '*.neff' ! -name 'barrier_*' | wc -l)
echo "session_dir=$D neff=$N model_neff_count=$NN"
if [ -z "$D" ] || [ -z "$N" ]; then echo "MISSING $t"; exit 1; fi
rm -rf $W/parquet/$t
nice -n 10 neuron-explorer view -d "$D" -n "$N" --output-format parquet --output-file $W/parquet/$t --ingest-only > $W/logs/nx_$t.out 2>&1
echo "nx_rc=$?"
tail -3 $W/logs/nx_$t.out
(cd $W/parquet && nice -n 10 tar czf $W/parquet_$t.tgz $t) ; echo "tar_parquet_rc=$?"
ls -l $W/parquet_$t.tgz
md5sum $W/parquet_$t.tgz
echo "DONE $t $(date -u +%FT%TZ)"
} > $log 2>&1
