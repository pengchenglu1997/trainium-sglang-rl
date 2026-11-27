# Session 2 record, 2026-11-20: SGLang / vLLM on Trainium2, weight sync, CUDA baseline

> Copy of the record of the first report. Parts that do not concern this repository are left out (sections 2 and 10 and
> one machine row), and disk ids are not given. Everything else is as written. Files and folders named here
> (`results_*`, `DESIGN_NOTE_*`) are on the author's laptop, not in this repository; the loop result files of this
> session are under `data/2026-11-20/`.

Times are UTC unless marked. Numbers are read from the result files named in each section; nothing here is
estimated unless it says so. Result folders (laptop):

- `results_sgl_2026-11-20/final/` (trn2; `part6.tgz` sha256 `e0a16e1e…`, 30 result files, checked for their
  `{"done": true}` line; `30_rl_loop_hf*.jsonl` were stopped by hand, `48_inplace_inplace1.jsonl` is a recorded failure)
- `results_gpu_2026-11-20/` (failed SGLang-on-GPU attempts in `results_sgl/`)
- kit: `sglang_rl/` (scripts 20–51, `backends.json`, `patches/sglang_neuron_v0.1.0_inplace/`)

## 1. Machines

| what | id | type | region | from – to | notes |
|---|---|---|---|---|---|
| trn2 | `<instance id redacted>` (spot `<spot request id redacted>`) | trn2.48xlarge | ap-south-2 | 08:57 – guard 21:58 | guard moved twice by the user's word (18:59 → 20:58 → 21:58) |
| GPU 2 | `<instance id redacted>` (spot `<spot request id redacted>`) | g7e.2xlarge | us-east-1 | 19:51 – about 20:00 | started for the SGLang CUDA reference, stopped on the user's word before it produced anything; disk deleted with it |

Idle time: the first five trn2 steps ended 09:54; the session that drove the machine stopped, and nothing ran until
17:38. The loops, probes and patches below all ran after 17:38.

## 2. (omitted from this copy)

## 3. trn2: engines as released (Qwen3-0.6B, 4 logical cores, TP4)

| | SGLang-Neuron v0.1.0 | vLLM-Neuron 0.29.0.1.2.0 (container) |
|---|---|---|
| starts on trn2 | yes (docs name Trn3 only) | yes |
| first start / restart, warm cache | 1224 s / 455–460 s | 350 s / 350 s |
| GSM8K greedy, 1,319 questions | 0.5641 | 0.5648 (paired: both right 692, only SGLang 52, only vLLM 53) |
| llmbench, 1024-token prompt, concurrency 32 | 2,469 tok/s | not run here (team round 3: 2,326 tok/s, another run) |
| sampling with top_p / top_k | refused under the `temperature` sampler (HTTP 400); full-vocabulary sampling works | works |
| n > 1, seed | not reached in the first probe run (all sampled requests were refused) | n = 8 works; same seed, same text |
| logprobs | yes | yes |
| weight update endpoints | HTTP 400, "weights are loaded into its compiled graphs once" | HTTP 404 (no such route) |
| LoRA endpoints | "LoRA is not enabled" | 404 |

Launch notes that cost time: the SGLang client must send `top_p 1, top_k -1, min_p 0` (the model's generation config
is otherwise applied); vLLM-Neuron refuses `NEURON_RT_VISIBLE_CORES` and wants `NEURON_VISIBLE_DEVICES` as a list of
logical cores (`16,17,18,19`; `4` resolved to one core); the Neuron PyTorch 2.14 DLAMI has no vLLM environment.

Restart time is warm-up, not weights: weight load 0.45 s; 21 decode shapes (7 batch buckets × 3 contexts) 340.6 s,
about 16 s a shape. Context 1024 / 32 requests: 18 shapes, 375 s. One batch bucket (`--cuda-graph-bs-decode 32`): 3
shapes, 139 s.

DDP (LoRA step, global batch 8, one process per core): 0.99 s single, 1.03 / 0.76 / 0.69 / 0.67 s at 1 / 2 / 4 / 8
ranks.

## 4. RL loop with restart-based sync (the released engines)

Same loop (`30_rl_loop.py`, 4 questions × 8 answers a step, lr 2e-4), per step:

| seconds | SGLang restart (4 steps) | vLLM restart (10 steps) |
|---|---|---|
| rollout | 2.6 | 1.0 |
| train | 4.8 | 3.6 |
| export merged checkpoint | 0.9 | 0.9 |
| sync (restart) | 457.5 | 349.5 |

Three jobs ran side by side on separate cores when these were taken. The vLLM run degraded (greedy 0.59 → 0.02);
with 4 questions a step these runs say nothing about learning. The transformers-generate loop did not finish its
first evaluation in 25 minutes and was stopped.

## 5. Weight sync at the runtime level (one NeuronCore, no server)

`45_inplace_swap_probe.py`, `46_lora_swap_probe.py`:

| | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| weights on the device | 1.19 GB | 16.38 GB |
| overwrite all parameters in place | 0.10 s | 1.19 s |
| forward after the swap vs warm forward | same (0.02 s compiled) | same (0.17 s eager) |
| device bytes before / after | equal | equal |
| logits after copying the old weights back | identical | identical |

Adapter only (0.6B, rank 16, 20 MB): adapter kept as a branch in the graph, overwrite A and B: 0.026 s, forward
0.028 s against 0.019–0.021 s for the bare base; delta computed on the device and added to the base weights: 0.13 s
(subtract old, add new). Two different compiled graphs of the same weights differ by about 2 logit units in bf16
(side branch zeroed vs bare base: 1.94), which is the size of the differences seen for merge and un-merge; these
could not be told apart. The first run of this probe (tag `lora`) counted adapter 1 twice in its merged part; its
timings stand, its logit columns do not; tag `lora2` is the corrected run.

## 6. Source reading (three readers, each checked by a second reader against the code)

- SGLang-Neuron, Qwen3 dense: weights are device parameters passed to the compiled graphs as inputs. The refusal is
  two guards (`general_plugin.py` `_refuse_weight_update`; `models/host.py` second `load_weights`). No technical
  reason is stated for Qwen3; direct-submit models (Qwen3.5, GPT-OSS giga, GLM-5.2) keep a template that assumes no
  update. Each rank holds a transformed shard (fused QKV, transposes, slices); an executor cache is keyed by tensor
  identity, so an update must write into the existing tensors.
- vLLM-Neuron: no update hook; the only reload is a rebuild. It compiles through another backend
  (libtorch-neuronx-lite), not in the copied sources: the runtime-level result above is not shown for it.
- vllm-neuron PR #98: `param.data.copy_` in place; Qwen3.5 only and TP 1; the same tensor is broadcast to every
  rank; the worker does not check shapes or dtypes; no drain or cache reset; no timing in the PR.

## 7. Experimental patch: in-place sync inside SGLang-Neuron (`patches/sglang_neuron_v0.1.0_inplace/`)

Not upstream, not mergeable as is. `update_weights_from_disk` is let through to SGLang's own path; a second
`load_weights` (a) for a checkpoint: rebuilds this rank's shards on the host with the model's own loaders and
`copy_`s them into the live parameters; (b) for a PEFT adapter folder: pushes `scale · B·A` through each
parameter's own loader, adds it to a host copy of the base shard, writes in place. The new path is read from the
caller's frame. The engine ties its compile cache to its own source hash, so each patch version needed its own
cache (`sgl_cache_v0.1.0_patched`, `_patched2`).

| check (`48`, `49`, `50`) | result |
|---|---|
| full update, HTTP round trip, 4 ranks | 0.41 s and 0.34 s (per rank: stage 0.14–0.27 s, copy 0.03–0.07 s) |
| back to the base | token logprobs identical to the start |
| in place vs fresh restart on the same checkpoint | same text, logprob difference 0.0 |
| 100 updates in a row | 100 ok, every answer equal to its reference, 0.26–0.38 s, no drift |
| 40 updates with 8 threads asking | 688 answers: 320 old weights, 368 new, 0 neither, 0 HTTP errors; 0.46–0.58 s |
| adapter update | first 8.5 s (one-time staging), then 0.67 s (delta 0.55 s per rank, copy 0.03 s) |
| adapter 1, then 2, then 1 | third equals first exactly |
| zero adapter | base exactly |
| adapter sync vs restart on the merged checkpoint | same text, logprob difference 0.0 |

Not covered: long requests or more than 8 concurrent ones during an update; any model family other than Qwen3
dense; the ordering of a host-to-device copy against launches already queued (the test saw no mixed answer; SGLang's
update lock is the likely reason, read in its tokenizer manager).

## 8. RL loop with in-place sync

Same loop as section 4 on the patched server (3-shape configuration):

| | 4 questions a step, 20 steps | 32 questions a step, lr 1e-4, 25 steps | 32 questions, lr 2e-4, 25 steps |
|---|---|---|---|
| rollout s | 2.3 | 19.9 | 20.7 |
| train s | 4.4 | 33.1 | 40.6 |
| export s | 0.9 | 0.9 | 0.9 |
| sync s | 0.38 | 0.37 | 0.37 |
| greedy before → after | 0.58 → 0.55 (100 questions) | 0.574 → 0.680 (500 questions) | 0.574 → 0.622 (500 questions) |
| reward, first 5 → last 5 steps | 0.43 → 0.59 | 0.56 → 0.75 | 0.57 → 0.68 |

One run and one seed per column. In the lr 2e-4 run every answer reached the 256-token limit in 6 of the 25
steps, the first at step 18; in the lr 1e-4 run in none.
Rollout errors: 0 in all three.

Trainer vs server logprobs on the same sampled tokens (`51`, 24 answers, 1,536 tokens, temperature 1): base/base
median 0.001, p95 0.078, mean 0.015; after an adapter sync median 0.002, p95 0.094, mean 0.049 with one token at
28.96 (not examined; without it the mean is about 0.03); control with mismatched weights median 0.006, p95 0.486,
mean 0.088.

## 9. Not obtained

- SGLang on CUDA with its own in-place update (the reference number): pip install of sglang 0.5.23 on the GPU AMI
  failed four ways (no CUDA home; no ninja; nvcc 13.5 against 13.0 headers; then cicc 13.5 emitting PTX 9.5 for a
  13.0 ptxas); the official image did not fit in 52 GB free. GPU 2 was started for it and stopped before use.
- vLLM-Neuron: restart with fewer buckets; any in-place test; PR #98 itself.

## 10. (omitted from this copy)

## 11. Afternoon additions (after 21:00 UTC; guard moved to 01:57 UTC on the user's word)

Final copy: `results_sgl_2026-11-20/final/` from `part7.tgz` (sha256 `214e7e70…`) and `part7b.tgz` (py-spy raw and summary),
45 result files; not finished by design: `30_rl_loop_hf*.jsonl`; recorded failures: `48_inplace_inplace1` (compile cache of
another source hash), `50_adapter_adapter_dev2` (the engine refuses unknown `SGLANG_NEURON_*` environment names). Final
adapter of the 150-step run: `final/adapters_long1/`.

### 11.1 Same loop, four layouts (Qwen3-0.6B, 32 questions x 8 answers a step, lr 1e-4, 25 steps, 500 eval questions)

| seconds per step | in-place sync (`big_lr1e4`) | + 4 replicas, compiled training, adapter sync (`all_opt`) | whole machine: 8 TP4 replicas, 16 training processes (`whole1`) |
|---|---|---|---|
| rollout | 19.9 | 5.3 | 2.7 |
| train | 33.1 | 14.1 | 2.9 |
| export | 0.90 | 0.07 | 0.08 |
| sync | 0.37 | 1.20 (4 replicas, adapter, host path) | 1.22 (8 replicas) |
| step | about 55 | 20.6 | 6.9 |
| greedy before -> after | 0.574 -> 0.680 | 0.580 -> 0.662 | 0.574 -> 0.636 |

One run each. The restart-based loop of section 4 stood at about 466 s a step with 4 questions.
150 steps on the whole-machine layout (`long1`, 4,800 questions, 17 minutes): greedy accuracy on 500 questions
0.574 (before), 0.672 (step 25), 0.656 (50), 0.712 (75), 0.696 (100), 0.710 (125), 0.748 (after); step 6.83 s mean; largest
share of answers at the token limit in a step 0.30; no rollout error.

### 11.2 What each addition measured

- 8B inside the patched server (`48_inplace_inplace_8b`): full in-place update 6.09 s and 5.65 s (per rank: staging its 4.1 GB
  shard from disk 4.7-4.9 s, copy 0.9-1.1 s); back to the base and against a fresh restart: token logprobs identical;
  restart 169 s.
- Compiled training step (`52_train_step_train`): 0.39 s against 0.87 s per micro-batch of 4 x 576, same loss (0.58056 /
  0.58054), first call 158 s, peak 14.3 / 14.6 GB.
- Multi-core LoRA trainer (`ptrainer.py`, `56_ptrainer_*`): 128 sequences, 8 processes 2.6-2.7 s against 13.8 s on one;
  first update's loss and LoRA checksums equal to the single-core run; later updates differ in the fourth digit of the
  loss. The first version (lists of tensors through pipes) took 15.6 s for the same update; flat arrays fixed it.
- Server layouts (`55_rollout_bench_*`, 512 answers, 32 requests in flight per server): TP1 976-999 tokens/s on one
  core; TP2 1,079-1,088 on two; TP4 about 2,300 on four (from the loop, not this script). A 256-token answer takes
  about 8.4 s on TP1 and about 2.7 s on TP4 at batch bucket 32.
- Logprob parity with token alignment fixed (`51_parity_parity2`): base/base mean 0.015; after an adapter sync mean
  0.017, median 0.002, p95 0.088, max 0.44 (22 answers); mismatched weights mean 0.088. The 28.96 outlier of the first
  run was a retokenisation misalignment (review finding), not a weight difference.
- Warm-cache start, py-spy on one server process for 110 s (`logs/pyspy_rl_all_opt_1.*`): of the samples with
  `_warmup_all` on the stack (39%), nearly all are inside torch._dynamo convert_frame (39%), AOT (24%) and StableHLO
  conversion (16%): the model is re-traced and lowered on every start before the NEFF cache is used. One process, one
  start: a lead, not a full account.
- Review of the day's code (3 reviewers, each finding checked): 4 confirmed and fixed (final evaluation on stale weights
  when sync-every > 1 or on an early stop; a dead replica not detected; adapter base not reset by a full update; parity
  token alignment).

### 11.3 Open

- Adapter delta computed on the device (`50_adapter_adapter_dev`, `_dev3`): 0.13-0.15 s an update, replace and zero checks
  exact, but NOT equal to a fresh start on the merged checkpoint (logprobs off by up to 1.14). The in-server diagnostic
  shows the device delta and final values equal to the host's (max difference 1e-9 / under 0.001% of entries) for the
  first parameter of every rank, and the same operations agree outside the server (`53_device_delta_delta`). Not
  located. The loops use the host path (0.67 s a replica).
- vLLM-Neuron: a plain Hugging Face model does not run on its compile backend (`54_vllm_backend_probe.out`: eager fails
  on a dtype check, compiled fails with "Could not load the model status=2"); testing in-place updates there needs a
  patch of its worker, not done.
- Training Qwen3-8B: 16.4 GB of weights on a 24 GB core and 14 GB peak for 0.6B training; no single-core path; not designed.
- Not done from DESIGN_NOTE_sglang_rl_2026-11-20.md: length buckets for training, one decode context instead of three,
  sync from tmpfs for 8B, trace caching for cold start.
- SGLang on CUDA with its own in-place update: still not obtained (section 9).

## 12. Engine-level round (22:40 UTC to 00:05 UTC on 2026-11-21); trn2 terminated at about 00:06 UTC

Final copy: `results_sgl_2026-11-20/final/` from `part8.tgz` (sha256 `245aa4ec…`), 67 result files, all with their done
line; recorded failures as in section 11 (`48_inplace_inplace1`, `50_adapter_adapter_dev2`); `30_rl_loop_long2.jsonl.attempt1`
is a start that failed because the previous loop's trainer still held its core. This section replaces the open items of
11.3 where it says so.

### 12.1 Patch v6 (what the PR draft describes)

Route changed from a second `load_weights` with frame inspection to an AROUND hook on
`WeightUpdater.update_weights_from_disk` that drains the device queue and calls `ModelHost.update_weights_in_place(path)`;
the second `load_weights` raises again as in v0.1.0. `50_adapter_v6_host`: all five adapter checks exact, including equality
with a fresh start on the merged checkpoint; drain 0.001 s, write 0.67-0.70 s. Upstream function md5 on 43881a7:
`76cf178e5ecde7f739c01439a1c64f65`. Patch files now mirror the package tree (`general_plugin.py`, `models/host.py`,
`models/qwen3/sglang_model.py`; originals under `orig/`).

### 12.2 The loop, final layouts (Qwen3-0.6B, 32 x 8 a step, lr 1e-4, 8 TP4 replicas, 16 training processes, adapter sync)

| run | steps | rollout s | train s | sync s | step wall s | greedy before -> after (500) |
|---|---|---|---|---|---|---|
| `whole2`: shared-memory trainer | 25 | 2.72 | 1.37 | 1.23 | 5.57 | 0.572 -> 0.666 |
| `whole3`: + rollout overlapped with training | 25 | 2.72 | 1.50 | 2.03 (waits for the rollout) | 3.86 | 0.578 -> 0.604 |
| `whole4`: + length buckets 320/448/576 | 25 | 2.71 | median 1.51, mean 8.3 | 1.95 | median 3.96, mean 10.5 | 0.572 -> 0.620 |
| `long2`: overlapped, 150 steps | 150 | 2.71 | 1.30 | 2.27 | 3.89 | 0.574, 0.642, 0.716, 0.706, 0.710, 0.680, then 0.582 at the end |

`long1` (section 11.1, not overlapped, 150 steps) ended at 0.748. So the overlapped loop is 1.4x faster per step, and in
its one long run the accuracy fell back to the start between step 125 and 150 (largest share of answers at the token
limit in a step: 0.49); in its two 25-step runs the gain was 2.6 and 4.8 points against 9.4 and 6.2 without overlap.
One run each: the overlap is not shown to be safe, and not shown to be the cause. The length buckets stalled the loop
whenever a worker met a shape for the first time (steps of 43 s), with no gain under overlap.

### 12.3 Single measurements

- Decode batch bucket on one TP4 server (`55_rollout_bench_tp4b*`): 32 -> 2,315 tokens/s, 16 -> 1,606, 8 -> 1,034; per
  stream 72, 100 and 129 tokens/s (13.8, 10.0, 7.7 ms a decode step).
- Shared-memory trainer (`56_ptrainer_pt16_shm`): 128 sequences on 16 processes 1.11-1.34 s; send 0.001 s.
- Answer-window logits (`pt1_win`): 21.5 s per 128 sequences against 13.8 s, peak 12.0 GB against 15.9 GB; with buckets
  (`pt1_winb`) 17.6-18.0 s. Length buckets alone (`pt1_b`): 9.3 s. First call with three buckets: 442-451 s.
- Time to first token (`57_ttft_v6`, one TP4 server): single request 20 ms; 8 requests of one question at once median
  65 ms, max 103 ms, the same when repeated; 32 at once median 169 ms, max 327 ms. The server reports no cached prompt
  tokens, so the prefix cache could not be observed.
- One decode context (`RLKIT_DECODE_LADDER=0`, `23_restart_v6_noladder`): decode warm-up 18.4 s for 1 shape against
  48.7 s for 3; whole start 114, 144 and 109 s (other jobs running), so most of a start is no longer the warm-up.
- 8B update with both checkpoints in tmpfs (`48_inplace_inplace_8b_tmpfs`): 6.01 s and 5.60 s, the same as from disk.
  The 4.7-4.9 s "stage" is the host-side rebuild of the rank's shard, not disk reads. Section 11.2's wording
  ("staging its 4.1 GB shard from disk") and the PR draft are corrected by this.

### 12.4 Device-side adapter delta: where it stands (replaces the first item of 11.3)

- Written three ways (`v6_dev_host`, `_add`, `_copy`): the same difference from the merged-checkpoint start each time
  (largest logprob difference 1.138437).
- Per-parameter diagnostic (`v7_diag`): on all 4 ranks, 0 of 112 parameters differ from the host result in more than 0.1%
  of entries; largest single difference 2.4e-4.
- Host path plus an unrelated eager device operation per update (`v8_dummy`): still exactly equal to the fresh start. So
  eager device operations in the server process are not the cause.
- Fork report (`v8_dev_fork`): on two of three prompts the 48 greedy tokens are the same as the fresh start's, with
  token logprobs differing by 0.005 and 0.011 on average (largest 0.03 and 0.11); on the third the answers fork at token
  3, where the two logprobs are -0.775 and -0.743; before the fork the mean difference is 0.016. The 1.14 is a
  difference between different texts after that fork.
- Reading: the device path serves weights that are close to the host path's (the size of the trainer/server
  difference of section 11.2), not identical; what makes them differ when the written values agree to single ulps in
  under 0.005% of entries is not found. The host path stays the default.

## 13. CUDA baseline and repeat runs (2026-11-21 00:20 to 02:35 UTC)

Machines (both launched on the user's instruction):

- GPU 3: `<instance id redacted>`, g7e.2xlarge spot (`<spot request id redacted>`), us-east-1, one NVIDIA RTX PRO 6000 Blackwell
  Server Edition (97,887 MiB), fresh 120 GB disk (deleted with the machine). Terminate call sent 01:15 UTC, state
  `shutting-down` at the last look; final state and spot request not re-checked (AWS login expired at 02:32 UTC).
- trn2 #2: `<instance id redacted>`, trn2.48xlarge spot (`<spot request id redacted>`), ap-south-2. Powered off from inside at about
  02:35 UTC (`shutdown -h`, instance set to terminate on shutdown); ssh refused afterwards. Final state and spot request
  not re-checked for the same reason.

Re-checked on 2026-11-21 after a new login: neither instance and neither spot request is listed any more; no instance
with our key and no open spot request in us-east-1 or ap-south-2; no volume attached to either instance id. The three
leftover volumes from section 10 are unchanged (all `available`).

### 13.1 CUDA baseline (`results_cuda_2026-11-20/`, archive sha256 7f5a4e15...)

Standard SGLang v0.5.23 (`lmsysorg/sglang:v0.5.23-cu130`, `--mem-fraction-static 0.35`), server and trainer on the one
GPU; torch 2.12.1+cu130, transformers 5.18.0, peft 0.21.2. The same loop script and settings as trn2 (32 questions x 8
answers, lr 1e-4, 500 evaluation questions). Sync: merged checkpoint through SGLang's own `update_weights_from_disk`.
Seconds, mean per step:

| run | steps | rollout | train | export | sync | step wall | greedy accuracy |
|---|---|---|---|---|---|---|---|
| `cuda_b4` (micro-batch 4) | 25 | 1.66 | 3.51 | 1.58 | 0.16 | 7.22 | 0.584 -> 0.646 |
| `cuda_b8` | 25 | 1.52 | 3.30 | 1.60 | 0.16 | 6.91 | 0.592 -> 0.714 |
| `cuda_b16` | 25 | 1.46 | 3.82 | 1.52 | 0.16 | 7.35 | 0.582 -> 0.660 |
| `cuda_long_b4` | 150 | 1.50 | 3.16 | 1.57 | 0.16 | 6.69 | 0.574, 0.664, 0.698, 0.696, 0.728, 0.704, 0.714 |

- `cuda_b32` and `cuda_long_b32`: failed, CUDA out of memory in the trainer (server holds 35.1 GiB). No numbers.
- `cuda_long_b4`: largest share of answers at the token limit in one step 0.90; no rollout errors.
- The "before" accuracy differs between runs (0.574 to 0.598) on the same weights and the same 500 questions; the
  evaluation is batched greedy decoding on the server, not checked further.
- In-place swap probe on CUDA (`45_swap_swap_cuda`): 0.126 s for all 310 tensors (1.19 GB), eager and compiled, device
  bytes unchanged, exact restore.
- The export of 1.5-1.6 s is the merged 1.2 GB checkpoint. trn2 exports only the adapter (0.08 s). A CUDA run with
  adapter-only sync was not made; "CUDA step minus export" (about 5.1-5.8 s) is arithmetic, not a measurement.

### 13.2 trn2 repeat runs (`results_seeds_2026-11-20/`, archive sha256 755379e1...)

Layout of `long1` / `long2`: 8 TP4 replicas (cores 0-31), trainer on core 48 plus 15 workers, compiled training,
adapter-only in-place sync, 150 steps, evaluation of 500 every 25 steps. All six files end with the done line, no
failure rows, no rollout errors.

Not overlapped:

| run | accuracy at step 0, 25, ..., 150 | rollout | train | export | sync | step wall |
|---|---|---|---|---|---|---|
| `long1` (seed 1, section 12; trainer before shared memory) | 0.574, 0.672, 0.656, 0.712, 0.696, 0.710, 0.748 | 2.70 | 2.83 | 0.08 | 1.22 | 6.83 (step_s; no wall figure) |
| `plain_s2` | 0.572, 0.642, 0.700, 0.696, 0.728, 0.690, 0.712 | 2.71 | 1.21 | 0.09 | 1.22 | 5.40 |
| `plain_s3` | 0.574, 0.662, 0.680, 0.730, 0.732, 0.718, 0.728 | 2.71 | 1.27 | 0.08 | 1.21 | 5.43 |
| `plain_s4` | 0.572, 0.648, 0.678, 0.686, 0.690, 0.732, 0.702 | 2.71 | 1.26 | 0.08 | 1.21 | 5.44 |

Final accuracy over the four seeds: 0.702, 0.712, 0.728, 0.748 (mean 0.7225). Gain over the start: 12.8 to 17.4 points.

Overlapped (`--overlap`; step wall 3.89-3.97 s):

| run | steps run | accuracy | what happened |
|---|---|---|---|
| `long2` (seed 1, section 12) | 150 | 0.574, 0.642, 0.716, 0.706, 0.710, 0.680, 0.582 | fell back by the end |
| `overlap_s2` | 150 | 0.574, 0.636, 0.606, 0.704, 0.676, 0.698, 0.732 | completed |
| `overlap_s3` | 60 | 0.574, 0.662, 0.672, then 0.470 at the stop | reward 0.71 at step 56, 0.04 at step 58; length guard stopped it (share at limit 0.82, 0.81) |
| `overlap_s4` | 9 | 0.572, then 0.618 at the stop | share at limit rose from step 4 (0.36) to 0.82; guard stopped it |

Reading: not overlapped, four of four seeds improve and stay improved. Overlapped, one of four finishes well, one falls
back and two run into the token limit and are stopped. The overlapped layout trains on answers sampled one update
behind; with these settings it is not stable. It is not offered as a result. What would make it stable (importance
weights for the stale policy, a smaller lr, a length penalty) was not tried.

In every 150-step run the reward of step 150 is 0.57-0.60 after 0.79-0.83 at step 149: the question order does not
depend on the seed, so this is most likely one hard batch (not checked).

### 13.3 trn2 against CUDA, as far as these runs go

One whole trn2.48xlarge (64 cores) against one RTX PRO 6000: step wall 5.4 s (seeds 2-4; seed 1 ran the earlier trainer, 6.8 s) against 6.7-7.3 s; accuracy after 150
steps 0.702-0.748 (four seeds) against 0.714 (one seed). By part: trn2 trains faster (1.2-1.4 s on 16 processes against
3.2-3.8 s), samples slower (2.7 s against 1.5-1.7 s) and syncs slower (1.2 s against 0.16 s). Whole machine against
one card; the sync payloads differ (13.1). Not a claim that trn2 is faster than CUDA.
