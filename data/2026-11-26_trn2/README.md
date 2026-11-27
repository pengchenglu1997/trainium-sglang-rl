# Result files of the trn2 session of 2026-11-26

One trn2.48xlarge (Trainium2, logical NeuronCore config 2), UTC 2026-11-27 00:34 to 05:13. Several jobs ran on the same
machine at the same time, each on its own chips. Every server in these files ran on the four logical cores of one chip,
TP4 unless the folder says otherwise (`dp_layouts/`). The write-up of the engine changes that were measured is in the fork
sglang-neuron (linked from the [top-level README](../../README.md); `docs/rl/RESULTS.md` there); this folder holds the files
behind it, and the scripts that wrote them are at the top level of this repository
([`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) section 10).

Every number in `tables.txt` is read from the files of this folder by `make_tables.py`. The text below states what
each sub-folder is; it repeats a number only where the file named next to it holds it.

## Conditions (as recorded by `s4_prep.sh`, `s4_run.sh`, `s4_x.sh` and the `76_serve_*.jsonl` records)

- Image: `public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04` (SGLang-Neuron
  v0.1.0). The vllm-neuron runs (`i12_vllm_start/`, the Qwen3-8B vllm rows in `serving/`) used a separate vllm-neuron image,
  named in `i12_vllm_start/step0_vllm_image.txt` and `i12_vllm_start/scripts/c15_capture.sh`.
- Models: Qwen3-0.6B and Qwen3-8B from the Hugging Face hub; GSM8K from the OpenAI grade-school-math repository
  (`serving/driver_logs/s4_prep.log`: 8,792 lines in the two files together).
- Server flags: the example `run_server.sh` of the image plus `--context-length 4096 --chunked-prefill-size 2048
  --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32`, TP4, cores of one chip.
  Each variant has its own compile cache folder; the `76_serve_*.jsonl` config line records the cache folder, the patch
  folder (`SGL_PATCH_DIR`) and the environment switches of every start.
- Benchmark client: llmbench at commit `08f416347db19bb45ff2489b8a69eacef62b6875` through `22b_llmbench_grid.sh`:
  prompt length 1,024 tokens, 256 output tokens with `ignore_eos`, concurrency 1, 8 and 32, 64 requests per cell after
  4 warm-up requests. The pass names in the file names: `t0` temperature 0 (two passes, `p1` and `p2`), `t08`
  temperature 0.8 with top-k, top-p and min-p switched off in the request, `k50` temperature 0.8 with top-k 50 and top-p 0.95.
  The request fields of each run are in the `.yaml` next to the `.csv`.
- Engine variants (patch folders are fork-branch states and are not in this repository; `s4_run.sh` and the
  `76_serve_*.jsonl` config lines name them): `rel` the released image; `main` fork main with the compile-cache stamp fix
  and the temperature sampler kernel (`RLKIT_SAMPLER_KERNEL=1`); `topk` the released `topk_topp` sampler; `kon` / `koff` the
  `topk_topp` kernel switched on / off (`RLKIT_SAMPLER_KERNEL_TOPK=1`); `lmh_on` / `lmh_off` the pre-transposed LM head
  switched on / off (`SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1` / `0`); `all` the integration branch with the temperature kernel
  and the LM head switch on, `topk_all` the same with the `topk_topp` kernel.
- Image tag, commit hashes and the environment of each start come from the scripts; nothing here was re-measured after
  the session.

## Folders

`serving/` The server records and benchmark summaries of the main script `s4_run.sh` and its generic form `s4_x.sh`
(`s4_8b.sh` for the Qwen3-8B rows). `llmbench_b_<tag>_<pass>.csv` and `.yaml` are the benchmark cells and their config;
`76_serve_b_<tag>.jsonl` records the server start (`ready_s`, the launch environment), a 64-token precheck request and the
exit status of each benchmark pass; `s4_profile_<tag>.jsonl` records the rollout passes of the profile runs (32 concurrent
requests of 256 tokens each, three passes, `ignore_eos`); `driver_logs/` are the marker lines of each driver call.
The first device-profile captures (`phaseA` and `phaseA2` of `s4_run.sh`: tags `q8_rel`, `q8_main`, `q06_topk`, `q06_main`,
`q06_rel`, `q06_lmh_off`, `q06_lmh_on`, without a suffix) ran with `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`: their rollout
passes ran at 43.2 to 203.7 tokens per second (for example `s4_profile_q06_rel.jsonl`: 70.4, 70.3 and 70.7, against 3,029.9,
1,642.7 and 3,039.2 in `s4_profile_q06_rel_nd.jsonl`, the same variant without the switch). The captures were repeated
without the switch (`phaseA3`, tags `_nd`; `phaseA4` for `q06_topk_kon` and `q06_topk_koff`). Pass 1 is slower than pass 0 and
pass 2 in every capture without the switch; the profile window was set to start at step 300 of rank 0, which lies in pass 1,
but whether that alone explains it was not isolated. `s4_profile_q06_lmh_on_nd.portfail.jsonl` and
`s4_profile_q8_rel_nd.portfail.jsonl` are two starts that did not become ready (the next start of the same tag worked).
The first attempt of the `lmh_off` and `lmh_on` benchmarks (`driver_logs/s4_b6.log`, `s4_b7.log`) ended with exit 3 (2 of 3
bench commands did not end in `LLMBENCH_DONE`); it left the four header-only `llmbench_*_k50_*.partial.csv` files. The second
attempt (`driver_logs/s4_x_lmh.log`, temperature-0 and `t08` passes) completed 3 of 3. The server records of the first attempt
(`*.prev.*`) are not published. A cache restored from a tar archive before one rerun is logged in `driver_logs/s4_restore.log`.
What this showed (`tables.txt` section 1, c32, temperature 0, passes 1 and 2): Qwen3-8B released 1,421.0 and 1,421.3 tokens per second,
`main` 1,496.5 and 1,499.1, `all` 1,492.1 and 1,500.8, vllm-neuron 1,126.9 and 1,112.4; Qwen3-0.6B `all` 3,324.2 and 3,313.3.

`correctness/` Token-level comparisons between servers (20 short prompts, 48 new tokens, one request at a time, `top_logprobs`
recorded, `out_*_t0.json`; sampled sets `t08`, `rl`, `minp` were also taken and are not published, `summary.jsonl` holds their
comparison lines), and a 200-question GSM8K pass (`21_gsm8k_cc_*.jsonl`, greedy, 512 new tokens). `events_*.jsonl` is each
collector's own record; `detail_*.jsonl` holds the per-prompt fork positions; `summary.jsonl` has one line per compared pair,
printed in section 4 of `tables.txt`. Each check has an A/A control (the same variant started twice) next to the switch
comparison. `gsm8k2/` is a second collection of two of the pairs. On Qwen3-0.6B the A/A controls are bit-equal in all sets;
on Qwen3-8B the extra A/A control `q8_rel` against `q8_rel_b` is not bit-equal and shows the same largest logprob
difference as the switch comparison `q8_rel` against `q8_all` (both lines of `summary.jsonl`, check4), so a difference of that size
there is not attributable to the switch.
What this showed (`tables.txt` section 4): Qwen3-0.6B, released against `all`, temperature 0: 17 of 20 prompts with the same tokens,
largest logprob difference before a fork 0.103; GSM8K 200 questions, accuracy 0.5550 against 0.5650, exact McNemar p 0.804.

`i12_vllm_start/` and `i34_nki_index/` Start tests of vllm-neuron on Qwen3-0.6B and Qwen3-8B. The image writes the kernel files
its NKI compile-result index names into the container's own `/var/tmp`; an index left in a cache folder by an earlier
container then points at files that no longer exist, and a later start can stop with `NCC_EVRF059`. `i12_vllm_start/`
is the first look: `starts.jsonl` has the config, the `parallel_trace_finished` time, the verdict and the cache file count of
each start, next to the cache listings and the index entries. The Qwen3-0.6B starts ended `ready` or `pass_stopped`; the
Qwen3-8B starts on a cache folder that a 0.6B container had left behind (`i12_r1_q8`, `i12_r1b_q8`) and an untouched copy of such a folder
(`i12_a1_ctl`) ended `fail`, and the copy with the index folder deleted first (`i12_a1_fix`) ended `pass_stopped`. Two more
Qwen3-8B starts are in `starts.jsonl`: `i12_a2_nopool`, on a stale cache with `VLLM_NEURON_DISABLE_PARALLEL_TRACE=1`, ended `fail`
(turning parallel trace off did not avoid the failure), and `i12_f34a` and `i12_f34b`, each on a cache folder that held 23 files
before the start (as `i12_a1_fix`) with no extra environment variable, ended `ready`. `i34_nki_index/`
holds the test of the launcher check that the repository now carries (`nki_index_check.py`, called by
`srv_vllm_neuron_docker.sh`): the old launcher on a stale cache stopped with 28 `NCC_EVRF059` lines (`results/starts.jsonl`,
`i34_a_old`); the new launcher removed the index folder and the same model reached ready (`i34_b_new`, `i34_c_new`); a start of
the old launcher on the cache the new launcher had left also reached ready (`i34_d_old`); `i34_e_new` repeated `i34_b_new` on a
second copy of the same stale cache (94 index entries): index folder removed, ready, 0 `NCC_EVRF059` lines (`results/starts.jsonl`). The launcher copies used in the test
are not repeated here. `retest/` is a rerun with a later version of the launcher (`i34_f_new2`, ready) with the CPU self-test
outputs on the machine and on a Windows git bash (`selftests_linux.txt`, `selftests_windows.txt`);
`selftests_linux_run1_fake_docker_not_executable.txt` is a first run that failed.

`i29_compiler_args/` The compiler command lines that the released image uses per graph, read from the processes of three
cold starts of Qwen3-0.6B, each on an empty cache (`cc_cmdlines.txt` is the raw list per start; `cc_table.txt` groups the compiles by
argument string and maps each to the server log's `warm` line; `neff_lists/` and `neff_compare_*.txt` compare the NEFF names of
the compile caches). With the `temperature` sampler (`i29_q06_temp_cold/cc_table.txt`) the 9 decode graphs share one argument
string with `-O2` and no modular-flow threshold, the 3 prefill graphs one with `-O1` and `--modular-flow-mac-threshold=10`.
With the `topk_topp` sampler (`i29_q06_topk_cold2/cc_table.txt`) all 12 graphs after the two small ones share the `-O1` string
with the threshold. The first `topk_topp` start's poller began late and recorded 11 compiles of 14. `inspect_*.json` is the
container description of one start. The image source excerpts that were read are not published.

`i37_topk_kernel/` Where the first-token logprob difference of the `topk_topp` kernel comes from, on Qwen3-0.6B TP4 with the
`topk_topp` sampler, 20 prompts, temperature 0, 48 tokens, `top_logprobs` 20, one server start per run. `r1`: kernel on and
off. `r2`: a second kernel-off server (A/A control); its kernel-on run with the vocabulary mask applied failed at collection
(`r2.log`) and was redone as `r3`. `r3`: kernel on and off with the engine's vocabulary mask always applied on the kernel path
(`patch_mask.diff`). `r4`: kernel off with the fp32 logits rounded to bf16 values (`patch_bf16.diff`) and kernel on with the winner
taken from the released code (`patch_dead.diff`). `extra/i37x/`: kernel off with the mask removed from the released path
(`patch_nomask.diff`), and three passes each of the kernel on and off with the mask applied, with their benchmark files.
`cmp_*.txt` are the pairwise comparisons; `grid_*.txt` count how many of the 380 top-20 logprob gaps lie on a 1/16 grid:
2 of 380 for `r1` with the kernel off, 380 of 380 for `r1` with the kernel on, 2 of 380 for the kernel on with the mask applied
(`grid_r3_kon_mask.txt`), 380 of 380 for both `r4` runs, 2 of 380 for the mask removed on the released path (`grid_r5.txt`).
In these runs the grid pattern appears with the kernel on when the mask is skipped (`r1` kon, `r4` kon_dead) and with the kernel off
when the fp32 logits are rounded to bf16 (`r4` koff_bf16, 380 of 380 in `grid_r4.txt`); it is absent with the mask applied
(`r3` kon and koff) and with the mask removed on the released path (`r5`). Why the compiler output changes with the mask line
was not determined. The answer dumps `out_*.json` (0.6 MB each) are not published.

`dp_layouts/` Replica layouts against TP4 on Qwen3-0.6B, on top of the `all` configuration (temperature kernel and LM head switch on): data-parallel 4 x TP1 and 2 x TP2 against the
TP4 server, with the TP4 control run twice (`ctl06_a` first, `ctl06_b` last) and a 20-prompt token comparison. The Qwen3-8B
data-parallel 4 x TP1 server did not become ready (`76_serve_d_c_dp4_8.jsonl`, `ready_s` null). Its server log (not published)
holds, at line 153, a `ValueError`: the loaded weights leave no GPU memory for the KV cache under `--mem-fraction-static=0.6`, and
it asks for a value above 0.637. No higher value was tried, so whether that layout can serve Qwen3-8B is not known.
`results/dp_table.txt` is the table made during the run.
What this showed (`results/dp_table.txt`, c32, temperature 0, passes 1 and 2): the four TP4 control passes 3,334.9 to 3,347.4 tokens per second;
4 x TP1 3,099.3 and 3,089.3; 2 x TP2 2,334.1 and 2,332.9.

`host_step/` The host side of one decode step on Qwen3-0.6B and Qwen3-8B, measured with two engine switches
(`--decode-log-interval 1` with `SGLANG_LOG_FORWARD_ITERS=1`, and `SGLANG_NEURON_HOST_TRACE`) and single flag changes
benchmarked against a control that stayed up (`events.jsonl`: starts and ready seconds; `summary/`: tables made from the server logs
and the host trace; `scripts/`: the summary scripts). The host trace TSV files (7 to 16 MB each), the process listing
and the server logs are not published.
What this showed (`tables.txt` section 1, c32, temperature 0, passes 1 and 2): `p2v1` to `p2v4` 3,313.1 to 3,350.0 tokens per second
against their controls 3,313.1 to 3,344.6; `p3v6` 3,339.4 and 3,338.9; `p3v5` 2,364.3 and 2,356.2; `p3v7` 2,462.1 and 2,496.9;
`p1m` 3,331.2 and 3,339.7, `p1t` 3,342.6 and 3,344.7 against `p1c` 3,337.5 and 3,339.2.

`kv_decode/` Decode and prefill settings that change the shape of the attention cache reads: `--cuda-graph-bs-prefill 2048`
(`p2k`), and `--context-length` 2048, 3072 and 16384 (`ctx2k`, `ctx3k`, `ctx16k`), each against a back-to-back control (`ctl`,
`ctl2`, `ctl3`), on Qwen3-0.6B and Qwen3-8B. `kv_summary.txt` is the table made during the run, `kv_tokcmp_*.txt` the token
comparison, `kv_server_log_facts.txt` the counts taken from the server logs (the prefill-kernel `chunk_num_blocks` warnings and the decode bucket lines of each start; the logs are not published). One device-profile
capture (`p2k_prof`) was taken and not analysed; the profile archive (49 MB) is not published.
What this showed (`kv_summary.txt`, c32, temperature 0, passes 1 and 2): Qwen3-0.6B controls 3,341.1 to 3,357.5 tokens per second (six passes);
`ctx2k` 3,271.0 and 3,335.2; `ctx3k` 3,059.9 and 3,063.2; `ctx16k` 2,591.3 and 2,594.9; `p2k` 2,874.2 and 2,868.8 with TTFT p50 0.461 and
0.463 s (controls 0.253 to 0.258 s). Qwen3-8B `p2k` 1,056.2 and 1,058.2 against controls 1,497.6 to 1,504.0.

`lm_head/` The pre-transposed LM head switch (`s0` / `s1`) with and without the temperature kernel (`k` / `n`), Qwen3-0.6B
(five server starts) and Qwen3-8B (two starts): benchmark files in `results/`, the 20-prompt and 200-question GSM8K comparison in
`cc/results/` (`summary.jsonl`; `cc/results/server_logs` are not published), the phase logs and the scripts. The benchmark
cells of the two Qwen3-8B starts are in `tables.txt` section 1 (`llmbench_b_lmh_q8_s0k_*` and `llmbench_b_lmh_q8_s1k_*`); the
switch settings differ by less than 1% in tokens per second at each concurrency there, one start each. The patch files and the
profile archive (0.5 GB) are not published; `patch_sha256.txt` lists the sha256 of the seven patch files (paths relative to the
patch folder).
What this showed (`tables.txt` section 1, Qwen3-0.6B c32, temperature 0, passes 1 and 2): `s0k` 2,738.6 and 2,738.5 tokens per second;
`s1k` 3,337.3 and 3,326.7; `s0n` 2,452.9 and 2,446.6; `s1n` 2,972.7 and 2,971.1. Qwen3-8B: `s0k` 1,501.1 and 1,500.2; `s1k` 1,497.5 and 1,504.1.

`devprof/` and `prof_all/` Analysis of device profiles of one decode step of TP rank 0 (four steps from step 300; neuron-explorer
tables converted to parquet on the machine). `devprof/` covers the `_nd` captures of the released and main variants,
the LM head switch, and the `topk_topp` kernel on/off; `prof_all/` the three all-on variants. `out/` holds the text and csv outputs
of the scripts in `scripts/` that tell the step wall, the exclusive split of the step into math, DMA issue, DMA,
collectives and idle, the engine split, the sampler scope and the place of the vocabulary-sized transposes; `records/` are the
profile runs' `s4_profile_*.jsonl`. The analysis scripts carry the paths of the analysis machine, which were replaced by
`<local-path>`. The raw profiles, the parquet tables (several GB) and the written summaries are not published; the
summary tables are the `out/` files, read with the script name in each file name (`g03_tables.txt` collects the splits).

`cpu_reference/` of the session (CPU fp32 reference logprobs) is not published: it was CPU work and its comparison with the
hardware runs was not part of the checks above.

## Tag glossary

Read from the scripts and the `76_serve_*.jsonl` config lines. `q06` / `q8`: Qwen3-0.6B / 8B. "All-on": `RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1`.
- `s0k`, `s1k`, `s0n`, `s1n` (`lm_head/scripts/phaseB.sh`, `phaseC.sh`): `s0` / `s1` is `SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0` / `1`; `k` adds
  `RLKIT_SAMPLER_KERNEL=1`, `n` does not. `s1n_r2` is a second start of `s1n` from a copy of its cache.
- `ctl`, `ctl2`, `ctl3` (`kv_decode/`): all-on with no extra flag, started three times. `ctl06_a`, `ctl06_b` (`dp_layouts/`): the same, before and after the layouts.
- `ctx2k`, `ctx3k`, `ctx16k`: all-on plus `--context-length 2048`, `3072`, `16384`. `p2k`: all-on plus `--cuda-graph-bs-prefill 2048`.
- `dp2`, `dp4`: all-on plus `--tp-size 2 --dp-size 2`, and `--tp-size 1 --dp-size 4 --max-total-tokens 65536`.
- `host_step/`, all-on Qwen3-0.6B (`scripts/p1.sh`, `p1b.sh`, `p2.sh`, `p3.sh`); `p1c`, `p2c`, `p3c` are the controls of each phase:
  - `p1m`: `--decode-log-interval 1` and `SGLANG_LOG_FORWARD_ITERS=1`. `p1t`: `SGLANG_NEURON_HOST_TRACE` (rank 0). `p8t`: the same for Qwen3-8B.
  - `p2v1` `--scheduler-recv-interval 8`; `p2v2` `--stream-interval 4`; `p2v3` `--tokenizer-worker-num 2`; `p2v4` `--detokenizer-worker-num 2`.
  - `p3v5` `--disable-overlap-schedule`; `p3v6` `SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=3`; `p3v7` `--context-length 2560`;
    `p3t` `--disable-overlap-schedule` with the host trace.
- `i12`, `i29`, `i34`, `i37`: prefixes of the start names in `i12_vllm_start/`, `i29_compiler_args/`, `i34_nki_index/`, `i37_topk_kernel/`;
  the number is a working-line id, the folder name says what it tests. A suffix `_prof` is a start with the device-profile capture window.

## Limits

- One session on a shared machine; other containers ran on the other chips during most benchmark passes (the load of the
  neighbours was recorded as `docker ps` snapshots before and after a pass in the runs' own logs, which are not published,
  and not in between). The effect of the neighbours on the numbers was not separated from the effect of the variants.
- One server start per variant in most rows; the `ctl`, `ctl2`, `ctl3` and `ctl06_a`, `ctl06_b` controls show how much
  two starts of the same variant differ. A pair of rows from different times of the session is not a controlled comparison.
- Benchmark cells: 64 requests per cell after 4 warm-up requests. All benchmark cells of the published csv files report
  zero request errors and full success counts (checked over every `llmbench_*.csv` of this folder, the header-only partial files
  excepted).
- The profile captures are of four steps of one rank in one pass; the captures with DGE notifications on are not
  comparable with the benchmark passes.
- The comparisons of `correctness/` use 20 prompts and 48 tokens, one server start each; a forked token is the first place
  the token lists differ, and logprob differences are taken before it.
- Server logs, per-request benchmark files, raw device profiles, parquet tables, model files and compile caches are not
  published (size, and the logs name machine details). The `*.log` files of the benchmark client are not published either; the
  `.csv` and `.yaml` are. This folder is about 2.7 MB in about 930 files.
- The written session summaries of each line are internal working notes and are not published; what they say that matters
  for reading these files is in this README.
- Model output text in the `.jsonl` and `.json` files is the answer to public GSM8K questions or short fixed prompts; the
  names in them are those of the public questions. Paths such as `/home/ubuntu/w/...` are folders on the test machine.
  Instance ids and the addresses of the container network (IP addresses, gateway and MAC address) were replaced by `<redacted>`.

## Scrub

`scrub_check.py` scans every file of this folder (see its header for the patterns), `scrub_allow.txt` lists the hits that were
accepted, `scrub_report.txt` is the report of the final run: zero hits and the accepted ones with their reason. The list of
internal words the scan also uses is kept outside the repository because it holds names. `make_tables.py` writes `tables.txt`.
