# Reproduce the Qwen3-0.6B number of the trn2 session of 2026-11-26

Target: the pre-transposed LM head plus the temperature sampler kernel, Qwen3-0.6B, TP4 on one chip, prompt 1,024,
256 output tokens, concurrency 32: 3,337.3 tok/s (pass 1) and 3,326.7 tok/s (pass 2), from
`data/2026-11-26_trn2/lm_head/results/llmbench_b_lmh_q06_s1k_t0_p1.csv` and `_p2.csv`.

## What was run, and what was not

- The session ran on a trn2.48xlarge with the Neuron DLAMI and Docker. Nothing below was rerun after the session
  ("not rerun" marks each step that needs the machine).
- The patch folder of that run was copied by hand from a fork commit; it is not in this repository. `make_patch_dir.py`
  is the scripted form. On a CPU machine it built from fork `main` (commit 831b54b) the same seven files, with the same
  sha256, as the as-run folder: `data/2026-11-26_trn2/lm_head/patch_sha256.txt`. A folder built by the script has not itself
  been started on a machine.
- The as-run line used a copy of `s4_x.sh` with three `sed` edits (results folder, cache folder name, container name),
  see `data/2026-11-26_trn2/lm_head/scripts/setup.sh`. The commands below use `s4_x.sh` unchanged, so results go to
  `$DATA_ROOT/results_s4`. Cores 48-51 and port 30550 are the as-run values; any four free cores of one chip work.
- As run, the `bench` was a warm start: `phaseA.sh` first ran `prof lmh_q06_s1k` (the cold compile), then `phaseB.sh` ran
  `bench lmh_q06_s1k` on that cache. Step 3 below does the cold compile inside the `bench`.

## Steps

```bash
git clone https://github.com/pengchenglu1997/trainium-sglang-rl.git && cd trainium-sglang-rl      # this repository
REPO=$PWD
export DATA_ROOT=$HOME/w                       # default; models, caches, results and llmbench live under it
export LLMBENCH_DIR=$DATA_ROOT/llmbench        # 22b_llmbench_grid.sh looks here, s4_prep.sh clones to here
# VENV overrides the venv path of s4_x.sh, s4_run.sh and s4_8b.sh (default /opt/aws_neuronx_venv_torch_neuronx_2_14_0);
# s4_prep.sh (step 2) has that path fixed

# 1. fork code and patch folder (CPU, run anywhere with git)
git clone https://github.com/pengchenglu1997/sglang-neuron.git $DATA_ROOT/fork                      # has origin/upstream-main
python make_patch_dir.py --fork $DATA_ROOT/fork --ref 831b54b --out $DATA_ROOT/patch_831b54b
(cd $DATA_ROOT/patch_831b54b && sha256sum -c $REPO/data/2026-11-26_trn2/lm_head/patch_sha256.txt)   # seven "OK" lines

# 2. image, models, GSM8K, llmbench at its pinned commit (not rerun)
LLMBENCH_URL=<the llmbench repository named in the header of 22b_llmbench_grid.sh> bash s4_prep.sh

# 3. cold compile of Qwen3-0.6B, then two temperature-0 passes and one temperature-0.8 pass (not rerun)
bash s4_x.sh bench lmh_q06_s1k 06 temperature 48-51 30550 $DATA_ROOT/patch_831b54b \
  "RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1" t

# 4. the table (not rerun)
python s4_table.py $DATA_ROOT/results_s4
```

Step 3 ends with `S_DONE bench lmh_q06_s1k`. Server flags and concurrency (1, 8, 32) are set inside `s4_x.sh`; the defaults of
`22b_llmbench_grid.sh` give 64 requests per cell (`LB_REQ`) after 4 warm-up requests (`LB_WARM`).

## Where the output lands and what to compare with

- Benchmark files: `$DATA_ROOT/results_s4/llmbench_b_lmh_q06_s1k_t0_p1.csv`, `_p2.csv` and `_t08.csv` (with `.yaml`).
  The server record is `76_serve_b_lmh_q06_s1k.jsonl` in the same folder.
- Compare the `c32` column of `s4_table.py` with the published files:
  `python s4_table.py data/2026-11-26_trn2/lm_head/results | grep lmh_q06_s1k_t0`.
  Published c32: 3,337.3 and 3,326.7 tok/s; c1 325.8 and 326.3; c8 1,639.9 and 1,654.6.
- Spread: the three control starts with both switches on, in `data/2026-11-26_trn2/kv_decode/` (tags `ctl`, `ctl2`,
  `ctl3`; patch folder `sglang_neuron_int_topk`, unpublished, same two switches), gave 3,341.1 to 3,357.5 at c32. The machine was shared.
- Both switches off gives 2,446.6 to 2,454.9 at c32 (`lm_head/results/llmbench_b_lmh_q06_s0n_*.csv`); a run that
  lands there did not get the switches.
