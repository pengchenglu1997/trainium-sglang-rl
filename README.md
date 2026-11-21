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

## The patch

`srv_sglang_neuron.sh` copies the files under `SGL_PATCH_DIR` over the installed `sglang_neuron` package inside the
v0.1.0 container at start. The patch removes `update_weights_from_disk` from the refused routes, hooks SGLang's weight
updater, drains running launches and copies the new values in place (`copy_`, never a new tensor object). It covers the
Qwen3 dense family only. The engine ties its compile cache to a hash of its own sources, so a patched server needs
its own cache directory and a cold first start.

## Limits

Experimental. One model family, tested on Qwen3-0.6B and (sync only) Qwen3-8B. The overlapped mode (`--overlap`)
is not stable with the default settings.

## License

Apache-2.0. The files under `patches/sglang_neuron_v0.1.0_inplace/` are modified copies of files from
[sgl-project/sglang-neuron](https://github.com/sgl-project/sglang-neuron) at tag v0.1.0 (Apache-2.0); `orig/` holds
the unmodified v0.1.0 files.
