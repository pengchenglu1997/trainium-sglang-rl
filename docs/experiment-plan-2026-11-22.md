# Experiment plan: RL post-training of Qwen3 on Trainium2 with SGLang-Neuron

*Draft for review, 22 November 2026.*

## 1. Goal

Bring RL post-training of the dense Qwen3 models (0.6B and 8B) on Trainium2 to what the CUDA stack gives: weights
reach the rollout server without a restart, sampling takes the usual parameters, answers of any length are served, an
8B model trains, and an existing RL framework can drive the server.

The plan ends with one table and one figure per question below, for 0.6B and 8B, against one RTX PRO 6000:

1. How long does a weight update take, full checkpoint and adapter only?
2. How long is one RL step, and where does it go?
3. What does a step cost when answers are long or of mixed length?
4. Does an 8B LoRA step fit and run on trn2, and at what speed?
5. Does accuracy move the same way as on CUDA?
6. Can Miles (RadixArk's RL framework, which uses SGLang as its rollout engine) drive the patched server?

## 2. Rules

- An experiment runs only if it has one thing to check and a stated expected result.
  Every row below gives the check, the expected result with its basis, and the decision either way.
- "Estimate" marks a number that is not measured. Measured numbers come from `NIGHT_RECORD_2026-11-21.md` and
  `SESSION2_RECORD_2026-11-20.md`.
- Correctness before speed: an update path counts only when its token log-probabilities equal a fresh start on the
  same weights.
- Every reported number has its script and result file in the repo; one run is reported as one run.
- Cost is reported as machine-hours and run counts.

## 3. Where we stand (measured)

| | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| Full update in place | 0.34-0.41 s | 4.7-6.2 s (loaders 3.6-4.7 s, copy 0.3-1.1 s) |
| Adapter update in place | 0.67 s | 13.9-14.0 s (13.5 s forming full matrices on the host) |
| In place equals a fresh start | yes | yes; also GLM-4-9B and DeepSeek-V2-Lite, no patch change |
| RL step, 32 questions x 8 answers | 5.4 s (CUDA 6.7 s) | not run |
| Sampler | `topk_topp` takes top_p / top_k, 2,713-2,730 tok/s; keeps the token that crosses p | not run |
| Decode step, 32 rows | 10.4 / 14.8 / 26.5 ms at compiled context 1024 / 4096 / 8192 | not run |
| Decode step, 1 row | 9.3 ms with one bucket, 4.6 ms with the default buckets | not run |
| Where a decode step goes | the device runs back to back (gaps 6-9 µs); half of the matmul flops are transposes | not run |
| Training | LoRA, 16 processes, 1.2 s a step | not run; weights on one core 16.38 GB of 25.8 GB |
| Accuracy, GSM8K-500 | 0.57 -> 0.70-0.75 (4 seeds); CUDA 0.71 (1 seed) | not run |

Ready on the laptop, not yet on hardware: the adapter update that forms only this rank's shard (CPU-tested:
`python 62_lazy_delta_test.py <sglang-neuron v0.1.0 source root>`; host timing on the laptop: `python 61_adapter_delta_cost.py`); sglang-neuron PR 4 applied on
our patch (one file, no overlap).

## 4. Work packages

### W0. Laptop, before any machine (about half a day)

| # | Check | Expected | If not |
|---|---|---|---|
| W0.1 | Training-side padding: from the saved rollouts, the share of the 576-token training length that is padding, and the same for 2x length buckets | a share to put next to the serving row fill (0.58-0.63); no expectation on the value | - |
| W0.2 | Miles full-weight path on a fake server: Miles's `disk-delta` publication followed by `update_weights_from_disk`, `begin_weight_update` / `end_weight_update`, `update_weight_version`, `pause_generation` / `continue_generation`, against `selftest_fake_server.py` extended with those routes | the calls Miles makes are all routes the SGLang under SGLang-Neuron v0.1.0 has (read 22 November: present; `pull_weights`, `get_weight_version`, `check_weights` absent) | list the missing calls; write a thin shim for them or fall back to SkyRL's adapter-by-path route |
| W0.3 | Write the 8B trainer variant: one model copy per core, per-layer gradient checkpointing, micro-batch 1, optimizer on the host | runs on CPU with a 2-layer stand-in; gradients equal the unsplit trainer | fix before W2 |
| W0.4 | Scripts for W1 and W2 dry-run against the fake server | pass | fix |

**W0 results, 22 November** (each piece built, then rerun by a reviewer who also tried to break it):

- W0.1: the saved loop files hold only per-step sums. Four 150-step trn2 runs, 38,400 answers each: mean answer
  149.0 / 158.1 / 160.6 / 166.9 tokens; 11.5-14.4 % of answers end at the 256-token limit; 43-46 % of answers are
  trained (the rest have zero advantage). Prompt lengths exist only in the server logs of the first run (41,900
  requests, mean 115.6, p95 157): with them the padding share at 576 is about 54 % for that run. Bucket shares cannot
  be computed from what was saved; the loop now logs sorted token counts per step (compile-checked, not yet run).
  `63_padding_share.py`.
- W0.2: one Miles `disk-delta` update sends the engine, in order, `POST /pull_weights` (twice at the first update),
  `POST /pause_generation`, `GET /flush_cache`, `POST /update_weights_from_disk {model_path, weight_version}`,
  `POST /continue_generation`, and reads `GET /model_info` for the version. In the SGLang under SGLang-Neuron v0.1.0
  every route exists with a compatible payload and the plugin lets it through, except `/pull_weights` (404): the
  engine side is where Miles expects the checkpoint to be rebuilt from the published deltas, and that code is not in
  our SGLang and was not read. So W1.7 needs a shim that does that rebuild; Miles would stop at its first call
  without it. This is a replay of Miles's sequence (read at main 9011e81), not Miles itself.
  `64_miles_sequence.md`, `64_miles_replay.py`; CPU-tested against the stand-in: `python selftest_fake_server.py`.
- W0.3: `trainer8b.py` (per-layer checkpointing, micro-batch 1, chunked answer logits, optimizer on the host,
  memory report). CPU-tested against a random small Qwen3 model: the gradients of the LoRA tensors and the values after
  one update are compared with the existing trainer, and planted bugs make the test fail
  (`python 65_trainer8b_cpu_test.py`). Not tested: anything on Neuron, length buckets, the multi-core trainer.
  `65_trainer8b_cpu_test.py`. The loop takes `--trainer8b`.
- W0.4: not done. The session scripts wait for the new AWS setup (below).

**Access, 22 November:** the AWS setup for Trainium is changing. No session starts until the new one is in place; the machine details of sections 4 and 5 (region, image, key,
security group) may change with them.

### W1. trn2 session 1, serving and sync (rows run side by side; about 3 h)

| # | Check | Expected, and why | If met | If not | Min |
|---|---|---|---|---|---|
| W1.1 | 8B adapter update with the shard-only delta | 13.9 s -> 2.5-3.5 s (estimate: host arithmetic was 13.5 of 14 s, scaled by the laptop timing of `61_adapter_delta_cost.py`); equal to a fresh start on the merged checkpoint | adapter sync is the 8B path | the per-rank log shows what is left (delta, add, copy); fix that one | 15 |
| W1.2 | Decode context ladder on, compiled context 8192, same short requests | step 10.4-14.8 ms instead of 26.5 ms (estimate: short rows run a quarter-context graph) | serve with the ladder on | per-operation device profile at 8192 | 30 |
| W1.3 | PR 4 at context 16384: a question whose answer needs text past token 8K, with and without the PR; GSM8K-500 at context 1024 both ways | without: wrong past 8K; with: right; GSM8K unchanged (basis: the PR's runs on Qwen3-MoE; no dense run exists) | carry PR 4 for long answers | if dense is right without it, leave it out | 45 |
| W1.4 | One group of 32 answers, one decode bucket against the default set | default set 5-20 % faster (estimate from 9.3 ms at 32 rows, 5.1 ms at 4) | default buckets for evaluation, one bucket for saturated rollouts | keep one bucket | 15 |
| W1.5 | Update while 8 requests of 2,000+ tokens are running | the update waits or pauses them, no HTTP error, every answer on one set of weights | report the wait | correctness bug: fix before anything else | 20 |
| W1.6 | 8B rollout speed, 256 requests of 256 tokens, TP4 and TP8 | no expectation; this is the missing row of the step table | - | - | 20 |
| W1.7 | Miles's full-weight calls (W0.2) against the real server at 0.6B, while requests run | update 0.34-0.41 s plus Miles's publication time; no failed request | plan the Miles run of W4 | fix the shim | 30 |

### W2. trn2 session 1, 8B training probes (other cores; about 1.5 h)

Design tried first: one whole 8B copy per core with per-layer checkpointing. It keeps the trainer we have, and one
six-minute probe can rule it out. Fallbacks: a two-stage layer pipeline (W2.4), tensor-parallel training (W2.5).

| # | Check | Expected, and why | If met | If not | Min |
|---|---|---|---|---|---|
| W2.1 | Free device memory after loading 8B and one eager forward at 1 x 576 | about 9.4 GB (25.77 - 16.38) | W2.2 | under 8 GB: go to W2.4 and W2.5 | 6 |
| W2.2 | Gradient checkpointing under compile, 0.6B at 1 x 576 | about 5 GB without, 3 GB with (estimate) | W2.3 | try eager checkpoint around a compiled block | 4 |
| W2.3 | 8B LoRA rank 16, seven projections, micro-batch 1: eager, then each layer compiled | peak 17.3-18.5 GB; about 4 s a micro-batch eager, under 2 s compiled (estimates) | W3 | W2.4, W2.5 | 35 |
| W2.4 | Hand-off of a [4, 576, 4096] bf16 tensor between two cores through host shared memory | under 10 ms a hop (estimate) | pipeline design possible | over 200 ms: dead | 10 |
| W2.5 | Backward through a collective on 4 cores against a CPU gradient | cosine at least 0.99 in one variant | tensor-parallel training possible | not on this stack | 15 |
| W2.6 | TorchTitan Neuron (listed in SDK 2.33.1, beta): does its Qwen3 or Llama example take one LoRA step on trn2 | unknown; it decides whether Miles's torchtitan backend is a route for the trainer | read its sharding for our trainer | drop this route | 30 |

Gate after session 1: W1.1, W1.5 and one of W2.3 / W2.4 / W2.5 must be met to go on to W3. If no 8B training design
survives, the plan continues at 0.6B (W4, W5) and reports the 8B training gap with the probe results.

### W3. Session 2, the 8B loop (trn2 about 6 h; one g7e about 3 h)

| # | Check | Expected | Min |
|---|---|---|---|
| W3.1 | Trainer and server log-probabilities on the same 8B answers | mean difference under 0.01 as at 0.6B | 20 |
| W3.2 | 8B RL step, 32 questions x 8 answers, adapter sync, layout chosen from W1.6 and W2.3: 25 steps for the step table | step time is the sum of the parts measured in session 1, within 20 % | 60 |
| W3.3 | 8B, 150 steps, 3 seeds on trn2; evaluation of 500 every 25 steps | accuracy rises from the 8B baseline; no expectation on the size | 3 x 40 (estimate) |
| W3.4 | The same loop on one RTX PRO 6000, 3 seeds, adapter-only sync (the fair row the first report lacks) | step time and curve for the comparison | 180 |
| W3.5 | 0.6B CUDA seeds 2-4 (a teammate's run) joined to the four trn2 seeds | equivalence test on the 150-step accuracy; step-0 spread 0.572-0.598 sets the noise floor | - |

### W4. Session 3, length and framework (trn2 about 4 h)

| # | Check | Expected | Min |
|---|---|---|---|
| W4.1 | Step time against answer length: max new tokens 256, 1,024, 4,096 with the ladder on (and PR 4 if W1.3 kept it) | tokens per second per replica falls with the compiled context as in W1.2; the table gives the cost of long answers | 60 |
| W4.2 | Mixed-length rollout (a task whose answers span 50 to 2,000 tokens): row fill and tokens per second, one bucket against the default set, requests sorted by expected length against unsorted | sorted and default buckets recover part of the empty rows; size unknown | 45 |
| W4.3 | Training with length buckets compiled at start (256 / 512 / 1,024) against one fixed length | step time falls by about the padding share of W0.1, with no mid-run compile | 45 |
| W4.4 | Miles end to end at 0.6B: Miles trainer on the GPU machine or on CPU, trn2 rollout server, full-weight `disk-delta` sync, 20 steps | the loop runs; update time as W1.7 | 90 |

Miles ships LoRA to SGLang as live named adapters over CUDA IPC or NCCL and rejects LoRA in `disk-delta`; a Neuron
process can take neither. So W4.4 is full-weight sync, and LoRA through Miles stays open (section 7).

### W5. Upstream

After a reply on sglang-project/sglang-neuron issue 15: a PR with a per-model switch in `catalog.py`, the replaced
function md5-pinned, a CPU-mode test (update in place, compare with a fresh load), covering Qwen3, GLM-4 and
DeepSeek-V2 as measured. The shard-only adapter delta goes in a second PR.

## 5. Machine time

| session | machine | hours (estimate) | what |
|---|---|---|---|
| 1 | trn2.48xlarge spot | 4 | W1, W2 |
| 2 | trn2.48xlarge spot | 6 | W3.1-W3.3 |
| 2 | g7e.2xlarge spot | 3 | W3.4 |
| 3 | trn2.48xlarge spot | 4 | W4 |

Each machine starts with a shutdown guard one hour past its estimate. Sessions 2 and 3 start only after the gate of
the session before.

## 6. Report

One document after session 2, updated after session 3, in the form of the 20 November report: summary, setup, one
table and one figure per question of section 1, issues found, limitations, code link at a commit. Figures: update
time by path and model; step time by part, 0.6B and 8B, trn2 and CUDA; step time against compiled context and against
live rows; accuracy against step with all seeds. The repo gets the result files this time.

## 7. Not covered, and open

- Qwen3.5 / 3.8 (needs a training kernel for the Gated DeltaNet layers; forward kernels exist, no backward found).
- Kernel work on the decode step (a teammate's line; the device profile of 21 November is shared with them).
- Serving an unmerged LoRA (the decode attention is one fused kernel), and with it LoRA through Miles.
- Overlapping rollout with training (two of four runs drifted; cause not shown).
- Whether any framework's trainer runs on Trainium; W2.6 is the only probe of that.
- Whether the 7B/8B target is the team's target: to confirm with the team before session 2.
