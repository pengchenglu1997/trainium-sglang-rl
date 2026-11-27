cd ~/w/sglang_rl
du -sh . ; ls
grep -n "name" 76_serve_and_bench.py | head -30
grep -n "def __init__\|self.name\|container\|docker" server_ctl.py | head -30
grep -n "results\|DATA_ROOT\|Path.home\|w/" 22b_llmbench_grid.sh | head -20
grep -n "Path.home\|DATA_ROOT\|/w/" 82_sampler_kernel_hw.py | head
cat backends.json | head -40
