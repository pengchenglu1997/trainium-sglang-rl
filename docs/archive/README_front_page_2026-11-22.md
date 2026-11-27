# Front page of 2026-11-22, moved out of the README

These parts of the top-level README describe the state of 2026-11-22. They are moved here without changes
(the links in them are relative to the repository root).

> **Where the work continues.** This page describes the work up to 2026-11-22. Since then the engine changes and
> kernels are developed in the fork [pengchenglu1997/sglang-neuron](https://github.com/pengchenglu1997/sglang-neuron)
> (its README has the branch layout), and the results of later sessions, including the trn2 session of 2026-11-24, are
> written up in that fork's
> [docs/rl/RESULTS.md](https://github.com/pengchenglu1997/sglang-neuron/blob/main/docs/rl/RESULTS.md). The result files
> of that session are in [`data/2026-11-24_trn2/`](data/2026-11-24_trn2/README.md); the probe scripts that wrote them
> (numbers 73 to 81) are at the top level, indexed in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 8. The trn2 session of
> 2026-11-26 (UTC 2026-11-27 00:34 to 05:13) is in [`data/2026-11-26_trn2/`](data/2026-11-26_trn2/README.md); its scripts
> (`s4_*`) are at the top level, indexed in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) section 10. The sections "Results so far" and "Plan and open work"
> below are as of 2026-11-22.

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

## Plan and open work

`docs/experiment-plan-2026-11-22.md`: W0 (laptop work: padding share, Miles call replay, 8B trainer variant) is done
except W0.4; W1 (serving and sync probes) and W2 (8B training probes) are written and wait for a machine; W3 (8B
loop), W4 (answer length and Miles end to end) and W5 (upstream PRs) follow the gates in the plan. Open items outside
the plan: serving an unmerged LoRA, LoRA through Miles, Qwen3.5-class models (no training kernel for the Gated
DeltaNet layers found).
