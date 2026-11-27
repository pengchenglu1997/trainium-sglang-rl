set -u
cd /home/ubuntu/w/i34
echo "copy_start $(date -u +%H:%M:%S)"
sudo cp -a /home/ubuntu/w/i12_cache_p2 /home/ubuntu/w/i34/cache_f
echo "copy_done $(date -u +%H:%M:%S)"
bash /home/ubuntu/w/i34/run_one2.sh i34_f_new2 /home/ubuntu/w/i34/cache_f new2 30345 1500
echo "DRIVE_F_DONE $(date -u +%H:%M:%S)"
