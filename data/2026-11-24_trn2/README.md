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
| `81_opt_sweep_r9.jsonl`, `opt_sweep_r9.md`, `llmbench_r9_*` | serving changes on Qwen3-0.6B, one factor at a time | 8.4 |
| `81_opt_sweep_r9b.jsonl`, `opt_sweep_r9b.md`, `llmbench_r9b_*` | a second pass of one variant of that sweep; RESULTS.md does not name these files | - |
| `82_sampler_kernel_r10.*`, `82_sampler_kernel_r12.*`, `83_sampler_check2_r11.jsonl`, `55_rollout_bench_roll_*.jsonl`, `llmbench_r10*`, `llmbench_r12*` | sampler kernel against the released samplers | 8.5 |
| `81_opt_sweep_r13.jsonl`, `r15`, `r16`, `opt_sweep_r13.md`, `r15`, `r16`, `llmbench_r13*`, `llmbench_r15*`, `llmbench_r16*` | sampler kernel combined with serving changes at Qwen3-0.6B | 8.5 |
| `81_opt_sweep_r14.jsonl`, `opt_sweep_r14.md` | one variant that did not become ready; no benchmark file; RESULTS.md does not name these files | - |
| `73_merge_kernel_bench.jsonl`, `78_kernel_h1.jsonl`, `78_kernel_h2.jsonl`, `50_adapter_h2_06b.jsonl`, `75_pr20_check_ulp1.jsonl` | LoRA merge kernel, first hardware run, and the one-ulp control | 8.6 |
| `80_moe30b_r8.jsonl`, `llmbench_moe30b_r8.csv` | Qwen3-30B-A3B start probe | 8.7 |
| `59_decode_profile_p*.jsonl` | the run lines of three device-profile captures; the profiles and their `summary.json` are not here, and RESULTS.md 8.8 does not name these three files | 8.8 |
| `s3_*.jsonl` | the session driver's record of each step: status, seconds, time (the driver itself is not public) | - |

How the files read:

- `*.jsonl`: JSON lines; a finished run ends with `{"done": true}`.
- `llmbench_*.csv`: one row per benchmark cell (concurrency, prompt length): requests, errors, tokens per second, TTFT and
  end-to-end percentiles.
- `*.md`: tables made from the files above by the session's table scripts.
- Some `.jsonl` files hold model output text for public GSM8K questions and server log lines. Paths such as
  `/home/ubuntu/w/...` are folders on the test machine.

Not in this folder:

- the per-request files of the benchmark client, server logs, the `.out` files of the kernel runs (for example
  `78_kernel_h1_bench.out`) and the device profiles;
- the scripts themselves: numbers 73 to 81 are at the top level of this repository (index:
  [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) section 8); the sampler-kernel scripts 82 to 84 are in the fork
  under `tests/rlkit_sampler/hardware/`.
