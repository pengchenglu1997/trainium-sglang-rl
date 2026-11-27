# Experiment index

One row per script: the question it answers, what it needs, where its results are, and which record states them.
Every script prints its usage in its header. "In repo" means a result file is under `data/`. Where the results column
says "not in repo", the numbers are in the record named in the next column; the result files themselves stayed on the
laptop. `SESSION2_RECORD` is `docs/SESSION2_RECORD_2026-11-20.md` (the first report, 20 November), `NIGHT_RECORD` is
`docs/NIGHT_RECORD_2026-11-21.md`. Status uses three words: **measured** (a record in this repository shows the run),
**CPU-tested** (no Trainium), **not run** (written, no hardware run yet).

## 1. What the server supports (RL needs from a running server)

| Script | Question | Needs | Results | Record | Status |
|---|---|---|---|---|---|
| `20_rl_probes.py` | Which calls does an RL rollout need, and which does the server refuse? Weight-update and LoRA calls are sent with a path that does not exist, to read the refusal | a running server | not in repo | SESSION2_RECORD §3 | measured |
| `21_gsm8k_acc.py` | GSM8K accuracy of a running server, question by question, so two engines compare pairwise (`compare` mode) | a running server | not in repo | SESSION2_RECORD §3 | measured |
| `22_llmbench.sh` | Speed of a running server on the team's grid (concurrency 1-32, prompts of 128 / 1,024 / 3,072 tokens) | server, `LLMBENCH_DIR` | not in repo | SESSION2_RECORD §3 | measured |
| `23_restart_time.py` | How long does a restart on unchanged weights take (the cost of restart-based sync)? | trn2 or GPU | not in repo | SESSION2_RECORD §3; PR draft table | measured |
| `57_ttft_probe.py` | Time to first token, and do the K requests of one question share the prompt's prefill? | a running server | not in repo | SESSION2_RECORD §12.3 | measured |
| `58_sampler_probe.py` | Does `topk_topp` take top_p / top_k, where does its top-p cut, and what does it cost? | trn2, patched or plain server | `data/2026-11-21_night/58_sampler_*.jsonl` | NIGHT_RECORD §3 | measured |

## 2. In-place weight sync

| Script | Question | Needs | Results | Record | Status |
|---|---|---|---|---|---|
| `45_inplace_swap_probe.py` | Can the weights of a model running on a NeuronCore be replaced in place, with no new compile or warm-up? (no server) | one core | not in repo | SESSION2_RECORD §5; PR draft, "Why this is safe" | measured |
| `46_lora_swap_probe.py` | Adapter-only sync at the runtime level: adapter as a side branch against merged into the base (no server) | one core | not in repo | SESSION2_RECORD §5 | measured |
| `53_device_delta_probe.py` | Why did the adapter delta computed on the device differ from the host-merged checkpoint? | one core | not in repo | SESSION2_RECORD §11.3, §12.4; PR draft, last open item | measured |
| `54_vllm_backend_probe.py` | Does vLLM-Neuron's compile backend read weights as data? Run inside its container | vLLM-Neuron container | not in repo | SESSION2_RECORD §11.3 | measured |
| `48_inplace_server_test.py` | Does a patched server take new weights in place, and are they the right ones (equal to a fresh start, back to base)? | trn2, patched server | `data/2026-11-21_night/48_*.jsonl` | NIGHT_RECORD §2, §4 | measured |
| `49_sync_stress.py` | What breaks after many in-place updates, and while requests are in flight? | trn2, patched server | not in repo | SESSION2_RECORD §7; PR draft, measurements | measured |
| `50_adapter_server_test.py` | Is an adapter-only update equal to a fresh start on the merged checkpoint? | trn2, patched server | `data/2026-11-21_night/50_adapter_q8.jsonl` | NIGHT_RECORD §4 | measured |
| `51_logprob_parity.py` | Do trainer and server give the same log-probabilities for the same tokens, before and after an adapter sync? | trn2, patched server, one more core | not in repo | SESSION2_RECORD §8, §11.2; PR draft, measurements | measured |
| `61_adapter_delta_cost.py` | Host-side cost of an adapter update by operation, on Qwen3-8B shapes | host only | not in repo | experiment plan §3 | measured (laptop) |
| `62_lazy_delta_test.py` | Does the shard-only adapter delta (`_LowRankSlice`) equal the full-matrix path through the engine's own loaders? | CPU, sglang-neuron v0.1.0 source tree | none | experiment plan §3 | CPU-tested (24 of 24 cases) |
| `68_update_under_load.py` | What does an in-place update do to requests that are running? (plan row W1.5) | trn2, patched server | none | experiment plan W1.5 | not run |
| `mk_alt.py` | Helper: perturbed copy of a checkpoint that the update tests move to | host | none | NIGHT_RECORD | helper |

## 3. Trainer

| Script | Question | Needs | Results | Record | Status |
|---|---|---|---|---|---|
| `40_ddp_probe.py` | Does `torch.distributed` DDP train a LoRA on Neuron, and how does step time change with cores? | trn2, 1-N cores | not in repo | SESSION2_RECORD §3 | measured |
| `52_train_step_probe.py` | Where does a training micro-batch's time go, eager against compiled? | one core | not in repo | SESSION2_RECORD §11.2 | measured |
| `56_ptrainer_probe.py` | Does the multi-core trainer compute the same update as one core, and how much faster? | trn2, several cores | not in repo | SESSION2_RECORD §11.2, §12.3 | measured |
| `63_padding_share.py` | How much of the fixed 576-position training length is padding, from saved loop runs? (plan row W0.1) | saved loop files | `data/2026-11-21_night/63_padding_share.txt` | experiment plan W0.1 | measured on saved runs |
| `65_trainer8b_cpu_test.py` | Does `trainer8b.py` give the same gradients and update as `trainer.py`? | CPU | none | experiment plan W0.3 | CPU-tested (36 of 36 checks) |
| `70_mem_probe.py` | Device memory of Qwen3 training on one core, stage by stage (plan rows W2.1, W2.2) | trn2, one core | none | experiment plan W2 | not run |
| `71_handoff_probe.py` | Cost of handing an activation between two NeuronCore processes through host shared memory (W2.4) | trn2, two cores | none | experiment plan W2.4 | not run |
| `72_collective_backward_probe.py` | Do gradients through a collective on Neuron match a CPU reference (W2.5)? | trn2, 4 cores | none | experiment plan W2.5 | not run |

## 4. Rollout speed and compiled shapes

| Script | Question | Needs | Results | Record | Status |
|---|---|---|---|---|---|
| `55_rollout_bench.py` | Sampled tokens per second of a set of rollout servers, as the loop asks | trn2, server per entry | not in repo | SESSION2_RECORD §11.2, §12.3 | measured |
| `59_decode_profile.py` | Where does a decode step go? Neuron system profile around one rollout pass | trn2 | `data/2026-11-21_night/59_decode_profile_p*.jsonl` | NIGHT_RECORD §5 | measured |
| `59b_`, `59c_`, `59d_*.py` | Summarise a rank's system profile, peek at its table, summarise a device profile | profile folders | `data/2026-11-21_night/59b_`, `59c_`, `59d_*.txt` | NIGHT_RECORD §5 | measured |
| `60_shape_cost.py` | What do compiled shapes (decode buckets, context) cost when the load does not fill them? | trn2, one server per setting | `data/2026-11-21_night/60_shape_cost_{A,B,C,D}.jsonl`, `shape_cost_rows.txt` | NIGHT_RECORD §6 | measured |
| `67_long_context_needle.py` | Does the server still use text past token 8K, with and without sglang-neuron PR 4? (W1.3) | trn2, patched server, PR 4 variant | none | experiment plan W1.3 | not run |

## 5. The RL loop and its reports

| Script | Question | Needs | Results | Record | Status |
|---|---|---|---|---|---|
| `30_rl_loop.py` | How long is one RL step on each backend, where does the time go, and does accuracy rise? | trn2 or GPU; trainer; servers | `data/2026-11-20/**/30_rl_loop_*.jsonl` | SESSION2_RECORD §4, §8, §11-13; PR draft; experiment plan §3 | measured |
| `60_report_figures.py` | Figures of the first report from the loop result files | result files | none | SESSION2_RECORD | helper |

## 6. Framework integration (Miles)

| Script | Question | Needs | Results | Record | Status |
|---|---|---|---|---|---|
| `64_miles_replay.py` | Replay of the HTTP calls one Miles disk-delta update sends to a rollout engine (our replay, not Miles itself) | a server URL | none | `64_miles_sequence.md` | CPU-tested against the stand-in (51 of 51) |
| `66_pull_weights_shim.py` | A proxy in front of the server that answers `POST /pull_weights`, so Miles's sync can drive SGLang-Neuron | server, Miles delta files | none | `64_miles_sequence.md`, gap 1 | CPU-tested with Miles's publisher code |
| `66_pull_weights_shim_test.py` | Test of the shim with Miles's own delta publisher | CPU, Miles checkout | none | as above | CPU-tested |
| `selftest_fake_server.py` | Client-side scripts against a stand-in server that mimics SGLang-Neuron's documented behaviour | CPU | none | experiment plan W0.2 | CPU-tested |

## 7. Session scripts

Which probes ran together on one machine, and on which cores.

| Script | Session | Runs |
|---|---|---|
| `50_run_trn2_session.sh` | 19-20 Nov, trn2 | steps `prep sgl_up sgl_acc sgl_probes sgl_bench loop_sgl loop_hf loop_vllm ddp cache_record`; resumable, a finished step is skipped. `cache_record` (last, added later) stops the session's server and the loop servers (not a server started outside the script; with `KEEP_SGL=1` nothing is stopped and the counts may not be final) and runs `82_cache_record.py` on the cache folders under `DATA_ROOT`, `cache_archives` excluded. Only this script has the step: `80_moe30b_probe.sh` and the 2026-11-24 scripts write caches but record nothing; run the `82_cache_record.py` command by hand after them |
| `scripts/run_trn2_seeds.sh` | 20 Nov, trn2 | the 150-step loop per seed on the whole machine, not overlapped then overlapped |
| `scripts/run_cuda_baseline.sh` | 20 Nov, one GPU | the CUDA baseline loop and the swap probe |
| `run_night_1121.sh` | 21 Nov night, trn2 | checks `gen glm4\|dsv2`, `sampler`, `q8`, `profile` side by side on their own cores. Profile output now goes to `$PROFILE_ROOT` (default `$DATA_ROOT/profiles`), not into the cache folder; the result files of that night name `/cache/profile_*` because they were produced before this change |
| `run_shape_1122.sh` | 22 Nov, trn2 | four servers A-D for `60_shape_cost.py` |
| `run_w1w2.sh` | planned | rows W1.1-W1.7 and W2.1-W2.5 of the experiment plan; flags of w1.3 and the 8B example path unverified |

## 8. Probes of the trn2 session of 2026-11-24

Added 2026-11-25. The code is what was on the machine. Comments only were changed afterwards: a status note in
`73_merge_kernel_bench.py`, `78_kernel_h1.sh` and `78_kernel_h2.sh`, whose headers said "not yet run", and reworded
comment lines in `22b_llmbench_grid.sh`, `75_pr20_check.py`, `76_baseline_pair.sh` and `81_opt_sweep.py` that pointed to
private notes. Results are in [`data/2026-11-24_trn2/`](../data/2026-11-24_trn2/README.md); the write-up is section 8 of
the fork's [RESULTS.md](https://github.com/pengchenglu1997/sglang-neuron/blob/main/docs/rl/RESULTS.md).

| Script | Question | Needs | Results (in `data/2026-11-24_trn2/`) | RESULTS.md | Status |
|---|---|---|---|---|---|
| `74_stamp_race.py` | Do two containers started together on one empty compile cache stop at start-up, with the released code and with the fix? | trn2, two containers, the fix as a patch folder | `74_stamp_race_r1.jsonl` | 8.2 | measured |
| `75_pr20_check.py` | Does upstream pull request 20's in-place reload serve what a fresh start on the same checkpoint serves? | trn2, the three files of the pull request as a patch folder | `75_pr20_check_pr20_q06.jsonl`, `75_pr20_check_pr20_q8.jsonl`, `75_pr20_check_ulp1.jsonl` | 8.1, 8.6 | measured |
| `76_baseline_pair.sh`, `76_serve_and_bench.py`, `77_pair_table.py` | vllm-neuron and SGLang-Neuron as released, one model, one benchmark client: tokens per second and TTFT | trn2, both images | `76_*.jsonl`, `77_pair_*.md`, `llmbench_q06*`, `llmbench_q8*` | 8.3 | measured |
| `22b_llmbench_grid.sh` | Helper: one grid of the benchmark client (llmbench at commit `08f4163`), which it clones. Called by `76_baseline_pair.sh`, `80_moe30b_probe.sh` and `81_opt_sweep.py` | network for the clone | the `llmbench_*.csv` files | 8 (run conditions) | helper |
| `untie_ckpt.py` | Helper: a copy of a tied-embedding checkpoint with its own `lm_head.weight`, which vllm-neuron's Qwen3 loader needs | host | none | 8.3 | helper |
| `73_merge_kernel_bench.py`, `78_kernel_h1.sh`, `78_nki_version.py`, `lora_merge.py` | Can the LoRA merge kernel be called on device tensors outside a serving graph, and what does one merge cost? One NeuronCore, no server | trn2, one core | `73_merge_kernel_bench.jsonl`, `78_kernel_h1.jsonl` | 8.6 | measured, first run |
| `78_kernel_h2.sh`, `79_kernel_ran_check.py` | Adapter updates on a Qwen3-0.6B server through the kernel, every read-back check on; and did the updates go through the kernel? | trn2, server with the kernel patch | `78_kernel_h2.jsonl`, `50_adapter_h2_06b.jsonl` | 8.6 | measured, first run |
| `mk_ulp.py` | Helper: a checkpoint in which a few elements of every matrix are moved by one unit in the last place (the control of 8.6) | host | none | 8.6 | helper |
| `80_moe30b_probe.sh` | Does the `integration` build start Qwen3-30B-A3B TP4 EP4 on trn2 and answer a known-answer prompt? One small benchmark cell | trn2, the `integration` build as a patch folder | `80_moe30b_r8.jsonl`, `llmbench_moe30b_r8.csv` | 8.7 | measured, one probe |
| `81_opt_sweep.py`, `81_opt_table.py`, `r9_run.sh` | Serving settings of SGLang-Neuron on Qwen3-0.6B, one factor at a time, then combinations | trn2 | `81_opt_sweep_*.jsonl`, `opt_sweep_*.md`, `llmbench_r9*`, `llmbench_r13*` to `llmbench_r16*` | 8.4, 8.5 | measured |
| `patch_manifest.py` | Helper: which files a patch folder would mount into the engine, with md5 | host | none | - | helper |
| `selftest_74.py`, `selftest_75.py`, `selftest_81.py` | CPU self-tests of the log classifier of 74, of 75 and of the sweep logic of 81 | CPU, no device | none | - | CPU only; run again 2026-11-25: 74 passes 4 of 4, 75 passes 32 of 32, 81 passed 108 of 110 (two cases still expected the default variant order from before it was changed during the session; they now expect the changed order, and 81 passes 125 of 125) |
| `82_cache_record.py` | End-of-session step: size in bytes and file count of each compile-cache folder, and any profile folder still inside one. Run it after the servers have stopped | host, no device | `82_cache_record_<tag>.jsonl` | - | CPU-tested (`selftest_cache_record.py`), not yet run in a session |
| `selftest_cache_record.py`, `selftest_launcher_env.py` | CPU self-tests of `82_cache_record.py` and of the profile-folder rule of `59_decode_profile.py`; of the environment handling of `srv_vllm_neuron_docker.sh` (a stand-in `docker`: the two compile-timeout names, `VLLM_ENV`, and the NKI index check branch, i.e. the check container's arguments, `VLLM_NKI_INDEX_CHECK=0`, a failing check container, the in-use skip for a cache path with a space, a parent and a mount source of `/`, and a launcher copied without the helper) | CPU, no device | none | - | CPU only |
| `selftest_nki_index_check.py` | CPU self-test of `nki_index_check.py`: keep, remove, unreadable-only entries, null, integer or empty kernel path, an empty or missing index folder, an error during removal | CPU, no device | none | - | CPU only |

Changed with these probes: `50_adapter_server_test.py` also writes the unrounded largest logprob difference;
`srv_sglang_neuron.sh` takes extra mounts (`SGL_MOUNTS`); `srv_vllm_neuron_docker.sh` takes the compile timeout
(`VLLM_COMPILE_TIMEOUT`), which it now sets under two names (`VLLM_NEURON_COMPILATION_TIMEOUT` and `NEURON_LIBTORCH_COMPILATION_TIMEOUT`; which one the compiler reads is not verified; the docs describe the second as a per-NEFF compile timeout in one place and as a wait for another compiling process in another), and extra container environment (`VLLM_ENV`, `NAME=value` items separated by spaces). `59_decode_profile.py` writes its profile under `/profiles` in the container (the host folder `--profile-root`, `$PROFILE_ROOT` or `$DATA_ROOT/profiles`), never under `/cache`. In the same change, comment lines that pointed to private notes were reworded in
`21_gsm8k_acc.py`, `22_llmbench.sh`, `srv_other.sh`, `srv_vllm_neuron_docker.sh` and `trainer.py`; no code changed.

Not in this repository:

- The patch folders these scripts mount (`patches/sglang_neuron_v0.1.0_pr20`, `_stampfix`, `_sampler_kernel`,
  `sglang_neuron_integration`). They are the changed files of the fork's branches: upstream pull request 20's head,
  `fix/compile-cache-stamp-race`, `feat/sampler-kernel` and `integration`. Their `PATCH_INFO.md` and `BLOBS.txt` lists
  are not here either.
- The kernel files `78_kernel_h2.sh` expects in `patches/sglang_neuron_v0.1.0_inplace/models/`
  (`rlkit_kernel_path.py`, `rlkit_lora_merge.py`): on the fork's branch `feat/lora-merge-kernel`. Without them the
  script stops at its setup check.
- The session driver that started the steps and wrote the `s3_*.jsonl` records.
- The sampler-kernel probes `82_` to `84_`: in the fork under `tests/rlkit_sampler/hardware/`.
- Planning notes that some headers refer to (task ids such as R1 to R9, U2, a ROADMAP, a kernel plan, a run list, a factor-layout note).

## 9. How a result file reads

Result files are JSON lines written by `rl_common.py`: one line per step or probe, a finished run ends with
`{"done": true}`, a failed step leaves a line with `"status": "fail"` before it. Counts and timings only; no question or
answer text. (The files of `data/2026-11-24_trn2/` differ: several were written by probes that are not indexed in this
file, and some hold model output text and server log lines; see that folder's README.) A file without the last line was stopped (for example `48_inplace_dsv2_full_stopped_after_1h.jsonl`, kept
on purpose; `48_inplace_dsv2_try1_cache_race.jsonl` is the first attempt that died on a compile-cache race).
