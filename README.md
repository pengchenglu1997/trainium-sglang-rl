# trainium-sglang-rl

RL post-training on AWS Trainium2 with SGLang-Neuron: a measured loop, and an experimental patch that lets
SGLang-Neuron v0.1.0 take new weights without a server restart.

**Current state**

- Last hardware session: the trn2 session of 2026-11-26 (UTC 2026-11-27 00:34 to 05:13).
- Its result files: [`data/2026-11-26_trn2/`](data/2026-11-26_trn2/README.md); the scripts that wrote them are `s4_*` at the top level.
- Reproduce its fastest Qwen3-0.6B number: [`REPRODUCE.md`](REPRODUCE.md).
- Write-up and results table of the engine changes: [`docs/rl/RESULTS.md`](https://github.com/pengchenglu1997/sglang-neuron/blob/main/docs/rl/RESULTS.md) in the fork [pengchenglu1997/sglang-neuron](https://github.com/pengchenglu1997/sglang-neuron).
- The results table and the plan of 2026-11-22 were moved out of this page: [`docs/archive/README_front_page_2026-11-22.md`](docs/archive/README_front_page_2026-11-22.md).

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
5. [The patch](#the-patch)
6. [Data and records](#data-and-records)
7. [Limits](#limits)
8. [License](#license)

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
| **Server launchers** | `srv_sglang_neuron.sh`, `srv_vllm_neuron_docker.sh` (with `nki_index_check.py`; CPU self-test `selftest_nki_index_check.py`), `srv_sglang_cuda_docker.sh`, `srv_other.sh` | one script per engine; backends registered in `backends.json`: `sglang_neuron`, `sglang_neuron_inplace`, `vllm_neuron`, `vllm_neuron_docker`, `vllm_cuda`, `sglang_cuda`, `sglang_cuda_docker` |
| **Probes of 2026-11-24** | `73_` to `81_`, `82_cache_record.py` (added after the session), `22b_llmbench_grid.sh`, `lora_merge.py`, `mk_ulp.py`, `untie_ckpt.py`, `patch_manifest.py`, `selftest_74.py`, `selftest_75.py`, `selftest_81.py`, `selftest_cache_record.py`, `selftest_launcher_env.py`, `r9_run.sh` | the trn2 session of 2026-11-24, as run; see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 8. The patch folders they mount are not here |
| **Session scripts of 2026-11-26** | `s4_prep.sh`, `s4_run.sh`, `s4_x.sh`, `s4_8b.sh`, `s4_profile.py`, `s4_table.py`; `22b_llmbench_grid.sh` (two new request variables) | the trn2 session of 2026-11-26, as run (comment lines reworded, one clone address made a variable; `s4_x.sh`, `s4_run.sh` and `s4_8b.sh` since read `DATA_ROOT` and `VENV` from the environment); see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 10. The as-run patch folders are not here; `make_patch_dir.py` builds one from a fork commit ([`REPRODUCE.md`](REPRODUCE.md)) |
| **Probes** | `20_` to `72_` | see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): question, hardware, result files and record for each |
| **Session scripts** | `50_run_trn2_session.sh`, `run_night_1121.sh`, `run_shape_1122.sh`, `run_w1w2.sh`, `scripts/run_trn2_seeds.sh`, `scripts/run_cuda_baseline.sh` | the order in which probes ran on one machine; each prints marker lines and a last line `*_DONE` |
| **Tests without hardware** | `selftest_fake_server.py`, `selftest_cache_record.py`, `selftest_launcher_env.py`, `selftest_nki_index_check.py`, `selftest_make_patch_dir.py`, `62_lazy_delta_test.py`, `65_trainer8b_cpu_test.py`, `66_pull_weights_shim_test.py` | CPU only |
| **Report helpers** | `60_report_figures.py`, `59b_profile_sum.py`, `59c_profile_peek.py`, `59d_devprof_sum.py`, `63_padding_share.py` | figures and summaries read from result files |
| **Patch** | `patches/sglang_neuron_v0.1.0_inplace/` (patched files, `orig/` holds the v0.1.0 originals), `patches/diff/`, `patches/PR_DRAFT_inplace_weight_update.md` | see [The patch](#the-patch) |
| **Data** | `data/2026-11-20/`, `data/2026-11-21_night/`, `data/2026-11-24_trn2/`, `data/2026-11-26_trn2/` | result files, see [Data and records](#data-and-records) |
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
| `PROFILE_ROOT` | host folder for Neuron profile output (`59_decode_profile.py`, `run_night_1121.sh`), mounted at `/profiles` in the server container; default `$DATA_ROOT/profiles`; never the compile cache folder |
| `LLMBENCH_URL`; `LB_TEMP`, `LB_EXTRA_BODY` | `s4_prep.sh`: the address `git clone` uses for llmbench (required, no default; the repository named in the header of `22b_llmbench_grid.sh`). `22b_llmbench_grid.sh`: the request temperature (default 0) and more request fields (`name=value` items separated by spaces, default none) |
| `VLLM_ENV`, `VLLM_COMPILE_TIMEOUT` | `srv_vllm_neuron_docker.sh`: extra container environment (`NAME=value` items separated by spaces); the compile timeout in seconds, set under both `VLLM_NEURON_COMPILATION_TIMEOUT` and `NEURON_LIBTORCH_COMPILATION_TIMEOUT` (which one the compiler reads is not verified) |
| `VLLM_NKI_INDEX_CHECK` | `srv_vllm_neuron_docker.sh`, default 1: before each start, `nki_index_check.py` reads `$VLLM_CACHE/neuron/compile_cache/nki/*.json` in a throwaway container of the server image with the same mounts. vllm-neuron 0.29.0.1.2.0 writes the kernel files those entries name to the container's own `/var/tmp`, which is gone with the container; a later start can fail with `NCC_EVRF059`. If any entry names a file that does not exist there, only that index folder is removed. With this image every entry seen names such a container-local file, so in practice any non-empty index is removed, including one with which a start reached ready; the keep branch (every readable entry names an existing file) applies only to entries naming files under `/cache` or `/models`, which this image does not write, and is covered by the CPU self-test only. Entries with an unreadable or empty kernel path are counted and on their own cause no removal. When the check is on, the launcher prints exactly one line starting `nki index check:` (also for a missing index folder, a missing helper file, or a failed check container, with its exit code). Skipped (and said so) while a running container mounts the cache folder, a folder inside it or a parent of it. A helper path that contains a comma makes the check container fail to start (the `--mount` option splits on commas); the launcher then prints the "check container failed" line and the start proceeds. Exactly `0` turns the check off |

**RL loop.**

```bash
# Trainium2: 8 TP4 replicas + 16-process trainer, seeds 2 3 4
bash scripts/run_trn2_seeds.sh "2 3 4"
# one GPU, standard SGLang in its container
bash scripts/run_cuda_baseline.sh
# figures from the result files
python 60_report_figures.py <dir with the result folders> <out dir>
```

**End of a session.** After the servers have stopped: `python 82_cache_record.py --results <dir> --tag <tag> --root $DATA_ROOT` writes the size in bytes and the file count of each compile-cache folder (every sub-folder of `$DATA_ROOT` with `cache` in its name, except `cache_archives`, where the saved tar archives of the caches are) to `82_cache_record_<tag>.jsonl`; `50_run_trn2_session.sh` (the 19-20 Nov session script) has it as its last step, `cache_record`, which stops the servers that script starts first (with `KEEP_SGL=1` it stops none and the counts may not be final). No other script runs it: after `80_moe30b_probe.sh` or the scripts of the 2026-11-24 session, run the command above by hand once their servers have stopped.

**Single checks on a patched server.** Each probe prints its usage in its header, for example
`48_inplace_server_test.py` (update equals a fresh start), `49_sync_stress.py` (many updates, updates under load),
`50_adapter_server_test.py` (adapter update). The full list is in `docs/EXPERIMENTS.md`.

**CPU tests, no Trainium.**

```
python selftest_fake_server.py
python 62_lazy_delta_test.py <sglang-neuron v0.1.0 source root>
python 65_trainer8b_cpu_test.py
python selftest_81.py
python selftest_cache_record.py
python selftest_launcher_env.py     # needs bash
python selftest_nki_index_check.py
python selftest_make_patch_dir.py   # needs git
MILES_DIR=<radixark/miles checkout> python 66_pull_weights_shim_test.py
```

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
| `data/2026-11-24_trn2/` | result files of the trn2 session of 2026-11-24: upstream PR 20 check, stamp race, baseline pair, serving changes, sampler kernel, LoRA merge kernel, 30B start probe. Its own [README](data/2026-11-24_trn2/README.md) maps each file to a section of the fork's RESULTS.md |
| `data/2026-11-26_trn2/` | result files of the trn2 session of 2026-11-26: serving benchmarks of the sampler kernels and the LM head switch on Qwen3-0.6B and Qwen3-8B, device-profile captures and their analysis, token-level correctness checks with A/A controls, replica layouts, host-side step time, attention-cache settings, vllm-neuron start tests, compiler arguments. Its own [README](data/2026-11-26_trn2/README.md) says what each sub-folder is, the conditions and the limits; `scrub_check.py` and `scrub_report.txt` are the check that no machine or account detail is in the folder |
| `docs/SESSION2_RECORD_2026-11-20.md` | the first report's record: engines as released, restart-based sync, the patch, the loop layouts, CUDA baseline, seeds (parts that do not concern this repository are left out) |
| `docs/NIGHT_RECORD_2026-11-21.md` | machine, setup and every number of the night session, with checksums of the archives |
| `docs/experiment-plan-2026-11-22.md` | the plan for the next sessions: work packages W0 to W5, expected results, gates |
| `64_miles_sequence.md` | the calls Miles's disk-delta weight sync sends to a rollout engine, and the gaps against SGLang-Neuron |

Not in the repository (size) from the session of 2026-11-26: server logs, per-request benchmark files, raw device profiles and
parquet tables, answer dumps of 0.6 MB and more, the written session summaries; `data/2026-11-26_trn2/README.md` lists them.

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

## License

Apache-2.0. The files under `patches/sglang_neuron_v0.1.0_inplace/` are modified copies of files from
[sgl-project/sglang-neuron](https://github.com/sgl-project/sglang-neuron) at tag v0.1.0 (Apache-2.0); `orig/` holds
the unmodified v0.1.0 files.
