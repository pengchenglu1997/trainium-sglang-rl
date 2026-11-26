# Result files of the trn2 session of 2026-11-24

One trn2.48xlarge (Trainium2), Neuron SDK 2.33, the SGLang-Neuron v0.1.0 image. File times are UTC, so most files carry
the date 2026-11-25. The write-up of these results, with conditions for every row, is section 8 of
[RESULTS.md](https://github.com/pengchenglu1997/sglang-neuron/blob/main/docs/rl/RESULTS.md) in the fork; each row there
names its source file by the names used here.

| Files | What | RESULTS.md |
|---|---|---|
| `75_pr20_check_pr20_q06.jsonl`, `75_pr20_check_pr20_q8.jsonl` | upstream pull request 20 on its own branch: in-place update against a fresh start | 8.1 |
| `74_stamp_race_r1.jsonl` | compile-cache stamp race, two containers on one empty cache, 10 trials per arm | 8.2 |
| `76_pair_*.jsonl`, `76_serve_*.jsonl`, `77_pair_*.md`, `llmbench_q06*`, `llmbench_q8*` | baseline pair: vllm-neuron and SGLang-Neuron as released | 8.3 |
| `81_opt_sweep_*.jsonl`, `opt_sweep_*.md`, `llmbench_r9*`, `llmbench_r13*` to `llmbench_r16*` | serving changes on Qwen3-0.6B, one factor at a time | 8.4 |
| `82_sampler_kernel_r10.*`, `82_sampler_kernel_r12.*`, `83_sampler_check2_r11.jsonl`, `55_rollout_bench_roll_*.jsonl`, `llmbench_r10*`, `llmbench_r12*` | sampler kernel against the released samplers | 8.5 |
| `73_merge_kernel_bench.jsonl`, `78_kernel_h1.jsonl`, `78_kernel_h2.jsonl`, `50_adapter_h2_06b.jsonl`, `75_pr20_check_ulp1.jsonl` | LoRA merge kernel, first hardware run, and the one-ulp control | 8.6 |
| `80_moe30b_r8.jsonl`, `llmbench_moe30b_r8.csv` | Qwen3-30B-A3B start probe | 8.7 |
| `59_decode_profile_p*.jsonl` | the three device-profile captures (the run lines; the profiles themselves are not here) | 8.8 |
| `s3_*.jsonl` | the session driver's record of each step: status, seconds, time | - |

How the files read:

- `*.jsonl`: JSON lines; a finished run ends with `{"done": true}`.
- `llmbench_*.csv`: one row per benchmark cell (concurrency, prompt length): requests, errors, tokens per second, TTFT and
  end-to-end percentiles.
- `*.md`: tables made from the files above by the session's table scripts.
- Some `.jsonl` files hold model output text for public GSM8K questions and server log lines. Paths such as
  `/home/ubuntu/w/...` are folders on the test machine.

Not in this folder:

- the per-request files of the benchmark client, server logs and the device profiles (size);
- the probe scripts numbered 73 to 84 that wrote these files. The sampler-kernel scripts are in the fork under
  `tests/rlkit_sampler/hardware/`; the others are not public yet.
