# trainium-sglang-rl

RL post-training on AWS Trainium2 with SGLang-Neuron: a measured loop, and an experimental patch that lets
SGLang-Neuron v0.1.0 take new weights without a server restart.

> **Where the work continues.** This page describes the work up to 2026-11-22. Since then the engine changes and
> kernels are developed in the fork [pengchenglu1997/sglang-neuron](https://github.com/pengchenglu1997/sglang-neuron)
> (its README has the branch layout), and the results of later sessions, including the trn2 session of 2026-11-24, are
> written up in that fork's
> [docs/rl/RESULTS.md](https://github.com/pengchenglu1997/sglang-neuron/blob/main/docs/rl/RESULTS.md). The result files
> of sessions after 2026-11-21 are not in this repository yet. The sections "Results so far" and "Plan and open work"
> below are as of 2026-11-22.

Three parts, each usable on its own:

| Part | What it is | Where |
|---|---|---|
| **Weight-sync patch** | `update_weights_from_disk` for SGLang-Neuron v0.1.0: a full checkpoint or a LoRA adapter is written into the tensors the compiled graphs already read | `patches/` |
| **RL loop** | GRPO-style policy gradient on GSM8K, LoRA trainer, rollout from one or more servers; the same loop runs on Trainium2 (SGLang-Neuron, vLLM-Neuron) and on CUDA (standard SGLang) | `30_rl_loop.py` and the modules it imports |
| **Probes and records** | one small script per timing or correctness claim, the result files they wrote, and the written records of each session | `NN_*.py`, `data/`, `docs/` |

Status: experimental, one run per claim unless a record says otherwise. [Limits](#limits) lists what was not tried.

## Contents

1. [Status of each piece](#status-of-each-piece)
2. [Architecture](#architecture)
3. [Repository map](#repository-map)
4. [Running it](#running-it)
5. [Results so far](#results-so-far)
6. [The patch](#the-patch)
7. [Data and records](#data-and-records)
8. [Limits](#limits)
9. [Plan and open work](#plan-and-open-work)
10. [License](#license)

## Status of each piece

| Piece | State | Evidence |
|---|---|---|
| In-place full-checkpoint update, Qwen3 (0.6B, 8B), GLM-4-9B-0414, DeepSeek-V2-Lite | measured on trn2, equal to a fresh start | `docs/NIGHT_RECORD_2026-11-21.md` §2, `patches/PR_DRAFT_inplace_weight_update.md` |
| In-place adapter update, Qwen3 | measured on trn2, equal to a fresh start; slower than the full update at 8B | NIGHT_RECORD §4 |
| RL loop on Trainium2 and CUDA, Qwen3-0.6B | measured, 4 seeds on trn2, 1 on CUDA | `data/2026-11-20/` |
| `topk_topp` sampler, decode profile, shape cost | measured on trn2 | NIGHT_RECORD §3, §5, §6 |
| Shard-only adapter delta (`_LowRankSlice` in the patch) | CPU test only, not run on hardware | `62_lazy_delta_test.py` |
| 8B LoRA trainer (`trainer8b.py`) | CPU gradient check only | `65_trainer8b_cpu_test.py` |
| `/pull_weights` shim for Miles disk-delta sync | tested on CPU with Miles's publisher code, not against a real server | `66_pull_weights_shim*.py`, `64_miles_sequence.md` |
| Probes `67_`, `68_`, `70_`, `71_`, `72_`, `run_w1w2.sh` | written, not run on hardware | each file says so in its header |
| Overlapped rollout and training (`--overlap`) | unstable with the default settings | [Limits](#limits) |

## Architecture

```
                      30_rl_loop.py   (one step: rollout -> reward -> advantages -> train -> sync)
                      /      |         \
        rollout_client.py    |          trainer.py  --> ptrainer.py (one process per NeuronCore)
        (OpenAI-style HTTP)  |                      --> trainer8b.py (8B: checkpointing, host optimizer)
                 |           |
                 |      server_ctl.py  reads backends.json, runs one srv_*.sh per engine
                 |           |
                 v           v
        rollout server(s): SGLang-Neuron (patched or not), vLLM-Neuron, SGLang CUDA, vLLM CUDA
                 ^
                 |  weight sync, chosen by --sync-method / --sync-payload
                 |    restart      stop the server, start it on the merged checkpoint (v0.1.0 default path)
                 |    sglang_disk  POST /update_weights_from_disk, merged checkpoint or adapter (needs the patch on Neuron)
                 |
        patches/sglang_neuron_v0.1.0_inplace/   copied over the installed sglang_neuron package at server start
```

One step of the loop: sample K answers for each of Q questions, reward 1 for the right final number, group
advantages, one LoRA policy-gradient update, then every `--sync-every` steps the new weights reach the servers and
every `--eval-every` steps a greedy evaluation runs. Each run writes one JSON-lines file ending with `{"done": true}`;
a failed step writes a line with `"status": "fail"` before it (`rl_common.py`).

## Repository map

Files sit at the top level because the records, `backends.json`, the session scripts and every script's `sys.path`
refer to them by name. Numbers in file names are the order in which the probes were written, not a pipeline.

| Group | Files | Role |
|---|---|---|
| **Core library** | `rl_common.py`, `rollout_client.py`, `server_ctl.py`, `backends.json`, `mk_alt.py` | results format and prompts, HTTP client, server start/stop from `backends.json`, perturbed checkpoint maker |
| **Trainers** | `trainer.py`, `ptrainer.py`, `trainer8b.py` | LoRA trainer (one core); data-parallel version (LoRA values and gradients in shared memory); 8B variant |
| **The loop** | `30_rl_loop.py` | the loop; main flags `--backend`, `--replicas`, `--sync-payload merged\|adapter`, `--train-compile`, `--train-cores`, `--overlap`, `--trainer8b`, `--len-buckets`, `--seed` |
| **Server launchers** | `srv_sglang_neuron.sh`, `srv_vllm_neuron_docker.sh`, `srv_sglang_cuda_docker.sh`, `srv_other.sh` | one script per engine; backends registered in `backends.json`: `sglang_neuron`, `sglang_neuron_inplace`, `vllm_neuron`, `vllm_neuron_docker`, `vllm_cuda`, `sglang_cuda`, `sglang_cuda_docker` |
| **Probes** | `20_` to `72_` | see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): question, hardware, result files and record for each |
| **Session scripts** | `50_run_trn2_session.sh`, `run_night_1121.sh`, `run_shape_1122.sh`, `run_w1w2.sh`, `scripts/run_trn2_seeds.sh`, `scripts/run_cuda_baseline.sh` | the order in which probes ran on one machine; each prints marker lines and a last line `*_DONE` |
| **Tests without hardware** | `selftest_fake_server.py`, `62_lazy_delta_test.py`, `65_trainer8b_cpu_test.py`, `66_pull_weights_shim_test.py` | CPU only |
| **Report helpers** | `60_report_figures.py`, `59b_profile_sum.py`, `59c_profile_peek.py`, `59d_devprof_sum.py`, `63_padding_share.py` | figures and summaries read from result files |
| **Patch** | `patches/sglang_neuron_v0.1.0_inplace/` (patched files, `orig/` holds the v0.1.0 originals), `patches/diff/`, `patches/PR_DRAFT_inplace_weight_update.md` | see [The patch](#the-patch) |
| **Data** | `data/2026-11-20/`, `data/2026-11-21_night/` | result files, see [Data and records](#data-and-records) |
| **Docs** | `docs/SESSION2_RECORD_2026-11-20.md`, `docs/NIGHT_RECORD_2026-11-21.md`, `docs/experiment-plan-2026-11-22.md`, `docs/EXPERIMENTS.md`, `64_miles_sequence.md` | records, plan, experiment index |

## Running it

**Needs.** Trainium2: a trn2.48xlarge with the Neuron DLAMI (torch-neuronx, transformers, peft) and Docker with the
SGLang-Neuron v0.1.0 image. CUDA: one GPU and Docker. The probes and loop read these from the environment:

| Variable | Meaning |
|---|---|
| `MODELS_ROOT` | host folder mounted read-only at `/models` in the server container |
| `SGL_CACHE` | compile cache folder, the same on every start (a patched server needs its own folder, see [The patch](#the-patch)) |
| `SGL_PATCH_DIR` | folder of patched plugin files to copy over the installed package |
| `SGL_IMAGE`, `SGL_CORES`, `SGL_SAMPLER`, `SGL_EXTRA` | image, `NEURON_RT_VISIBLE_CORES` of the server, sampler mode, extra launch flags |
| `DATA_ROOT` | disk with models and GSM8K for the session scripts |

**RL loop.**

```bash
# Trainium2: 8 TP4 replicas + 16-process trainer, seeds 2 3 4
bash scripts/run_trn2_seeds.sh "2 3 4"
# one GPU, standard SGLang in its container
bash scripts/run_cuda_baseline.sh
# figures from the result files
python 60_report_figures.py <dir with the result folders> <out dir>
```

**Single checks on a patched server.** Each probe prints its usage in its header, for example
`48_inplace_server_test.py` (update equals a fresh start), `49_sync_stress.py` (many updates, updates under load),
`50_adapter_server_test.py` (adapter update). The full list is in `docs/EXPERIMENTS.md`.

**CPU tests, no Trainium.**

```
python selftest_fake_server.py
python 62_lazy_delta_test.py <sglang-neuron v0.1.0 source root>
python 65_trainer8b_cpu_test.py
MILES_DIR=<radixark/miles checkout> python 66_pull_weights_shim_test.py
```

## Results so far

Trn2 numbers are one run each unless the record says more. Sources are the records named in the last column.

| Question | Qwen3-0.6B | Qwen3-8B | Source |
|---|---|---|---|
| Restart-based sync (the only path in v0.1.0) | 455-460 s (21 decode shapes), 139 s (one bucket) | not run | `patches/PR_DRAFT_inplace_weight_update.md` |
| Full update in place | 0.34-0.41 s | 4.7-6.2 s (loaders 3.6-4.7 s, copy 0.3-1.1 s) | PR draft; NIGHT_RECORD §4 |
| Adapter update in place | 0.67 s | 13.9-14.0 s (13.5 s forming deltas on the host) | PR draft; NIGHT_RECORD §4 |
| In place equals a fresh start | yes, same text and logprobs | yes; also GLM-4-9B-0414 and DeepSeek-V2-Lite, patch unchanged | NIGHT_RECORD §2 |
| Many updates, requests in flight | 100 updates, no drift; 40 updates with 8 threads: no HTTP error, no answer on neither weight set | not run | PR draft |
| RL step, 32 questions x 8 answers | 5.4 s on trn2 (CUDA 6.7 s) | not run | `docs/experiment-plan-2026-11-22.md` §3 |
| GSM8K-500 greedy accuracy after 150 steps | 0.57 to 0.70-0.75 (4 trn2 seeds); CUDA 0.71 (1 seed) | not run | experiment plan §3 |
| `topk_topp` sampler | takes top_p / top_k; 2,713-2,730 tok/s against 2,183-2,218 for `temperature` | not run | NIGHT_RECORD §3 |
| Decode step, 32 rows, compiled context 1024 / 4096 / 8192 | 10.4 / 14.8 / 26.5 ms | not run | NIGHT_RECORD §6 |
| Decode step, 1 row | 9.3 ms (one bucket), 4.6 ms (default buckets) | not run | NIGHT_RECORD §6 |
| Where a decode step goes | device busy back to back, gaps 6-9 µs | not run | NIGHT_RECORD §5 |
| Training | LoRA, 16 processes, 1.2 s a step | weights alone take 16.38 GB of a 25.8 GB core; not run | experiment plan §3 |

## The patch

`srv_sglang_neuron.sh` copies the files under `SGL_PATCH_DIR` over the installed `sglang_neuron` package inside the
v0.1.0 container at start. The patch:

1. removes `update_weights_from_disk` from the refused routes and hooks SGLang's weight updater, which drains running
   launches first;
2. checkpoint case: builds the first-phase graph on the CPU, runs the model's own loaders, and writes each tensor with
   `copy_` into the live device parameter of the same name (never a new tensor object, so the executor's binding cache
   stays valid);
3. adapter case (a PEFT `adapter_config.json` in the path): forms `scale * (B @ A)` through each parameter's own
   loader, adds it to a host copy of the base shard, writes the result in place; adapters replace each other and a full
   update clears the base copy;
4. the shard-only form of step 3 (`_LowRankSlice` in `models/host.py`) is CPU-tested only.

Scope: Qwen3 dense, GLM-4 and DeepSeek-V2 as measured; Qwen3-MoE, GPT-OSS, GLM-5.2, Qwen3.5 and Kimi-K3 were not
tried. The engine ties its compile cache to a hash of its own sources, so a patched server needs its own cache
directory and a cold first start. What a real upstream PR still needs is in
`patches/PR_DRAFT_inplace_weight_update.md`. The attention change of sglang-neuron PR 4 is not carried here; apply
that PR's diff to v0.1.0 to try it.

## Data and records

| Where | What |
|---|---|
| `data/2026-11-20/` | the 34 `30_rl_loop_*.jsonl` files of the first report: `trn2_sgl_final`, `trn2_seeds` (Trainium2) and `cuda`, `gpu_sgl` (GPU, with the three earlier attempts kept) |
| `data/2026-11-21_night/` | result files of the night session: `48_` (in-place update on other model families), `50_` (adapter), `58_` (sampler), `59*` (decode profile summaries), `60_` (shape cost), `63_` (padding share) |
| `docs/SESSION2_RECORD_2026-11-20.md` | the first report's record: engines as released, restart-based sync, the patch, the loop layouts, CUDA baseline, seeds (parts that do not concern this repository are left out) |
| `docs/NIGHT_RECORD_2026-11-21.md` | machine, setup and every number of the night session, with checksums of the archives |
| `docs/experiment-plan-2026-11-22.md` | the plan for the next sessions: work packages W0 to W5, expected results, gates |
| `64_miles_sequence.md` | the calls Miles's disk-delta weight sync sends to a rollout engine, and the gaps against SGLang-Neuron |

Not in the repository (size): server logs and raw Neuron profiles of 21 November (the record gives their checksums),
and the result files of the probes of 19-20 November other than the loop files (their numbers are in `docs/SESSION2_RECORD_2026-11-20.md`).

## Limits

- The RL loop was run on Qwen3-0.6B only. The weight sync was also tested on Qwen3-8B, GLM-4-9B-0414 and
  DeepSeek-V2-Lite, one run each.
- Models with packed or quantized weights (kernel models) were not tried.
- The overlapped mode (`--overlap`) is not stable with the default settings: two of four runs drifted, the cause is not
  shown.
- The AWS setup for Trainium changes after 22 November; machine details in the records (region, image, security group)
  may no longer hold.

## Plan and open work

`docs/experiment-plan-2026-11-22.md`: W0 (laptop work: padding share, Miles call replay, 8B trainer variant) is done
except W0.4; W1 (serving and sync probes) and W2 (8B training probes) are written and wait for a machine; W3 (8B
loop), W4 (answer length and Miles end to end) and W5 (upstream PRs) follow the gates in the plan. Open items outside
the plan: serving an unmerged LoRA, LoRA through Miles, Qwen3.5-class models (no training kernel for the Gated
DeltaNet layers found).

## License

Apache-2.0. The files under `patches/sglang_neuron_v0.1.0_inplace/` are modified copies of files from
[sgl-project/sglang-neuron](https://github.com/sgl-project/sglang-neuron) at tag v0.1.0 (Apache-2.0); `orig/` holds
the unmodified v0.1.0 files.
