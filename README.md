# trainium-sglang-rl

Measures an RL post-training loop on AWS Trainium2 with SGLang-Neuron, and carries an experimental patch that lets
SGLang-Neuron v0.1.0 take new weights without a server restart.

- **Weight sync:** a patch to the SGLang-Neuron plugin so `update_weights_from_disk` writes a full checkpoint, or a
  LoRA adapter, into the tensors the compiled graphs already read (`patches/`).
- **RL loop:** GRPO-style policy gradient on GSM8K with a LoRA trainer; rollout from one or more servers; the same
  script runs on Trainium2 (SGLang-Neuron, vLLM-Neuron) and on CUDA (standard SGLang) (`30_rl_loop.py`).
- **Probes:** each timing or correctness claim has its own small script (`4x_`, `5x_`).

## How a run works

`30_rl_loop.py` starts the rollout server(s) through `server_ctl.py` (one backend entry in `backends.json` and one
`srv_*.sh` per engine), then repeats: sample answers, score them, one trainer update, export, sync the servers, and
every N steps a greedy evaluation. Every run writes one JSON-lines file that ends with `{"done": true}`.

```bash
# Trainium2 (trn2.48xlarge, Neuron DLAMI): 8 TP4 replicas + 16-process trainer, seeds 2 3 4
bash scripts/run_trn2_seeds.sh "2 3 4"
# one GPU, standard SGLang in its container
bash scripts/run_cuda_baseline.sh
# figures from the result files
python 60_report_figures.py <dir with the result folders> <out dir>
```

## Files

| File | What it does |
|---|---|
| `30_rl_loop.py` | the loop; `--replicas`, `--sync-payload merged\|adapter`, `--train-compile`, `--train-cores`, `--overlap`, `--seed` |
| `trainer.py`, `ptrainer.py` | LoRA trainer; data-parallel version, one process per NeuronCore, LoRA values and gradients in shared memory |
| `rollout_client.py`, `server_ctl.py`, `backends.json`, `srv_*.sh` | sampling client and server start/stop per engine |
| `23_restart_time.py` | time of a restart-based weight sync |
| `45_`, `46_`, `53_` | in-place weight and adapter writes at the runtime level, no server |
| `48_`, `49_`, `50_` | in-place update on a running server: timing, equality with a fresh restart, stress |
| `51_logprob_parity.py` | trainer against server log-probabilities for the same answers |
| `52_`, `56_` | trainer step time, single process and parallel |
| `55_rollout_bench.py`, `57_ttft_probe.py` | rollout throughput and time to first token |
| `patches/sglang_neuron_v0.1.0_inplace/` | the patched plugin files; `orig/` holds the v0.1.0 originals; `patches/diff/` the same change as diffs |
| `patches/PR_DRAFT_inplace_weight_update.md` | what the patch does and what a real pull request still needs |

## Update of 22 November 2026

**Measured on trn2 on 21 November** (one run each; record in `docs/NIGHT_RECORD_2026-11-21.md`, result files in
`data/2026-11-21_night/`):

- The in-place update equals a fresh start on GLM-4-9B-0414 and DeepSeek-V2-Lite with the patch unchanged
  (`48_inplace_server_test.py`).
- The `topk_topp` sampler accepts top_p / top_k and keeps the token that crosses p (`58_sampler_probe.py`).
- Qwen3-8B update timing per rank, system and device profile of a decode step (`59*_`), and step time against decode
  buckets and compiled context (`60_shape_cost.py`).

**Local only, not yet run on hardware** (each file says so in its header): the shard-only adapter delta in the patch
(`_LowRankSlice` in `models/host.py`), `trainer8b.py`, `66_pull_weights_shim.py` (the `/pull_weights` route that
Miles's disk-delta sync calls), the probes `67_`, `68_`, `70_`, `71_`, `72_`, and `run_w1w2.sh`. The plan they belong
to is `docs/experiment-plan-2026-11-22.md`.

CPU tests (PyTorch on CPU, no Trainium):

```
python selftest_fake_server.py
python 62_lazy_delta_test.py <sglang-neuron v0.1.0 source root>
python 65_trainer8b_cpu_test.py
MILES_DIR=<radixark/miles checkout> python 66_pull_weights_shim_test.py
```

`data/2026-11-20/` holds the 34 `30_rl_loop_*.jsonl` files of the first report (`trn2_sgl_final`, `trn2_seeds`: Trainium2; `cuda`, `gpu_sgl`: GPU, with the three earlier
attempts kept). The server logs and raw profiles of
21 November are not in the repository (size); the record gives their checksums. The attention change of
sglang-neuron PR 4 is not carried here: apply that PR's diff to v0.1.0 to try it.

## The patch

`srv_sglang_neuron.sh` copies the files under `SGL_PATCH_DIR` over the installed `sglang_neuron` package inside the
v0.1.0 container at start. The patch removes `update_weights_from_disk` from the refused routes, hooks SGLang's weight
updater, drains running launches and copies the new values in place (`copy_`, never a new tensor object). It covers the
Qwen3 dense family only. The engine ties its compile cache to a hash of its own sources, so a patched server needs
its own cache directory and a cold first start.

## Limits

Experimental. The RL loop was run on Qwen3-0.6B only; the weight sync was also tested on Qwen3-8B, GLM-4-9B-0414 and
DeepSeek-V2-Lite (one run each). The kernel models with packed or quantized weights were not tried. The overlapped mode (`--overlap`)
is not stable with the default settings.

## License

Apache-2.0. The files under `patches/sglang_neuron_v0.1.0_inplace/` are modified copies of files from
[sgl-project/sglang-neuron](https://github.com/sgl-project/sglang-neuron) at tag v0.1.0 (Apache-2.0); `orig/` holds
the unmodified v0.1.0 files.
