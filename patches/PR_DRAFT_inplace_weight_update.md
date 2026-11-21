# Draft: in-place weight updates for the dense transformer hosts (SGLang-Neuron)

Status: draft for discussion, written 2026-11-20 against the v0.1.0 release image (plugin sources as installed there,
upstream SGLang 0.5.24.dev413+g43881a7). Not opened anywhere. The patch in `sglang_neuron_v0.1.0_inplace/` is the
experiment the numbers below come from; it is not in the shape of a pull request yet (section "What a real PR still
needs").

## Why

An RL post-training loop sends new weights to its rollout server after every update. v0.1.0 refuses every
`update_weights_*` call ("the model's weights are loaded into its compiled graphs once"), so the only way to new
weights is a restart, and a restart re-warms every compiled shape:

| Qwen3-0.6B, TP4, trn2.48xlarge | seconds |
|---|---|
| restart, example configuration (21 decode shapes) | 455-460 |
| restart, one batch bucket (3 decode shapes) | 139 |
| of which loading the weights | 0.45 |

In a loop of 4 questions x 8 answers a step that is 457 s of sync next to 2.6 s of rollout and 4.8 s of training.

## What the patch does

1. `update_weights_from_disk` is no longer refused at the tokenizer. An AROUND hook on
   `WeightUpdater.update_weights_from_disk` (the model runner's side) drains the device queue
   (`drain_launches`, as `_flush_cache_drained` does) and calls `ModelHost.update_weights_in_place(model_path)`.
   Upstream's own body is not run (it would hand the model a weight iterator and call `load_weights` again).
2. `ModelHost.update_weights_in_place(path)`, checkpoint case: a fresh CPU graph of the first phase is built, its
   `load_weights(path, cpu)` runs the model's own loaders (this rank's shards, fused QKV, transposes), and each tensor
   is written with `copy_` into the live device parameter of the same name. No parameter object is replaced, so the
   executor binding cache stays valid; an aggregated server's second graph and a tied LM head share the same objects.
   Nothing is compiled or warmed.
3. Adapter case (`path` holds a PEFT `adapter_config.json`): only the adapter is read. For every live parameter built
   from adapted checkpoint matrices, `scale * (B @ A)` goes through that parameter's own loader (the loaders are
   slicing, concatenation and transposes, so they commute with the sum), is added to a host copy of the base shard
   taken at the first adapter update, and is written in place. Each update starts from the base copy, so adapters
   replace each other. A full update clears the base copy.
4. `second load_weights` still raises, as in v0.1.0.

Why this is safe to do at all: in a plain TorchNeuron process (eager and `torch.compile(backend="neuron")`), overwriting
all 596 M parameters of Qwen3-0.6B in place takes 0.10 s (16.4 GB of Qwen3-8B: 1.19 s), the next forward costs what a
warm forward costs, the logits change, and copying the old weights back restores them exactly. The compiled graphs read
the weights as inputs.

## Measurements (trn2.48xlarge, SDK 2.33, one run each unless said)

| check | result |
|---|---|
| full update over HTTP, Qwen3-0.6B TP4 | 0.34-0.41 s |
| full update, Qwen3-8B TP4 | 5.6-6.1 s (4.7-4.9 s of it staging the rank's 4.1 GB shard on the host; the same with the checkpoint in tmpfs) |
| in place vs a fresh start on the same checkpoint (0.6B and 8B) | same text, token logprobs identical |
| back to the first checkpoint | token logprobs identical to the start |
| 100 updates in a row | all ok, every answer equal to its reference, 0.26-0.38 s each, no drift |
| 40 updates with 8 threads sending requests | 688 answers: 320 on the old weights, 368 on the new, none on neither, no HTTP error; 0.46-0.58 s |
| adapter update (rank 16, 7 projections, 40 MB as float32) | 0.67-0.70 s after a first of 7-9 s |
| adapter 1, then 2, then 1 again; then a zero adapter | third equals first exactly; zero adapter gives the base exactly |
| adapter update vs a fresh start on the host-merged checkpoint | same text, token logprobs identical |
| trainer vs server logprobs on the same sampled tokens, before / after an adapter update | mean abs difference 0.015 / 0.017 (0.088 with mismatched weights) |

The same RL loop (32 questions x 8 answers a step) with in-place sync: 0.37 s a sync, 55 s a step; with 8 replicas,
adapter sync and a 16-process LoRA trainer: 5.6 s a step; 150 steps took Qwen3-0.6B from 0.574 to 0.748 greedy
accuracy on 500 GSM8K test questions in 17 minutes (one run).

## What a real PR still needs

- The route: an explicit capability instead of an AROUND hook that skips upstream's body, and the replaced upstream
  function pinned in `_RUNTIME_SOURCES_MD5` (its md5 on 43881a7 is `76cf178e5ecde7f739c01439a1c64f65`, logged by the
  patch at each update).
- `update_weights_from_tensor` with a custom loader, so that a trainer on the same host does not go through files.
- Staging without building a whole CPU graph each time (the loaders and mappings can be kept from the first load):
  for 8B this is most of the 6 s.
- Models with a direct-submit decode (Qwen3.5, GPT-OSS giga, GLM-5.2): their template assumes no update; in-place
  `copy_` should keep it valid, not tested. Quantized (MX) weights need requantization through their own loaders.
- Pipelined (PP) and disaggregated servers: not looked at.
- Tests: none written; the checks above are scripts of an RL kit (`48_`, `49_`, `50_`, `51_`).
- `weight_version`, and a flush of the prefix cache after an update: left to upstream's scheduler code; that it runs
  after this hook was not verified beyond "answers after an update match a fresh start".
- The adapter delta computed on the device (0.13 s instead of 0.67 s) is close to the host result but not identical:
  same greedy tokens on two of three test prompts with token logprobs 0.005-0.011 apart on average, a fork at a near
  tie on the third. The written values agree with the host's to single ulps in under 0.005% of entries on every rank,
  the way of writing does not matter, and an unrelated eager device operation next to the host path changes nothing.
  Cause not found; the host path is the default.
- The engine ties its compile cache to a hash of its own sources: every edit of the plugin forces a full recompile.
  That made each iteration of this patch cost 4-5 minutes; a note for whoever continues.

## Separate small item

`Qwen3ForCausalLM.decode_context_ladder = True` compiles three decode contexts per batch bucket. A rollout server with
a short context uses one. The patch reads `RLKIT_DECODE_LADDER=0` to turn the ladder off; a launch option would be the
proper form. Measured: decode warm-up 18.4 s for 1 shape against 48.7 s for 3; the whole start stayed at 110-145 s
with other jobs on the machine, so the warm-up is no longer most of it.
