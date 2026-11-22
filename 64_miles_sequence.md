# Miles disk-delta weight sync: the calls a rollout engine receives, against our SGLang-Neuron

Plan item W0.2 (trainium-qwen3-experiment-plan-2026-11-22.md, section 4). Written 22 November 2026.

## What is Miles's own code and what is ours

* Miles's own code path: the files fetched from github.com/radixark/miles with the gh CLI on 22 November, branch
  `main`. The fetch did not pin a commit; `main` was at 9011e812d091a7bbc57d508898b93a081f2497d0 (committed 20 November
  21:06 UTC) when I asked afterwards, so line numbers below are for that state unless `main` moved in between.
  Files read in full: `miles/utils/disk_delta.py` (D), `miles/backends/training_utils/weight_update/protocols/delta.py`
  (P), `.../weight_update/updater.py` (U), `.../weight_update/session.py` (S), `.../weight_update/protocol.py` (PR),
  `miles/backends/sglang_utils/sglang_api_client.py` (CL), `tests/fast/backends/training_utils/weight_update/
  test_disk_delta_engine_calls.py` (T), `docs/advanced/disaggregated-rollout.md` (DOC); `megatron_utils/actor.py`
  lines 1035-1120 (A). Local copies: `miles_dl/` in the session scratchpad.
* Our replay of it: `64_miles_replay.py` sends the routes, methods and JSON fields below in the same order and applies
  Miles's own success test (`_make_request` raises on HTTP >= 400, CL:145-152; `check_weight_sync_results` raises on a
  JSON object with `"success": false`, S:98-119). We are not installing or running Miles; no Miles code is executed
  anywhere in the kit. The stand-in server in `selftest_fake_server.py` is ours.
* The engine side of `/pull_weights` is NOT in the files I read. D:15-18 says it lives in the engine
  (`sglang.srt.weight_sync.local_checkpoint`), and DOC:141-146 describes it in prose. I did not read that code.
  Everything below about what the engine does at `/pull_weights` is from those comments and the prose, marked "per
  Miles text", not from the implementation.

## One update, in order (what one rollout engine receives)

Rank 0 of the trainer is the only process that sends HTTP (P:375, P:176). Each stage goes to every engine at once and
waits for all before the next stage (`async_utils.submit` + `wait_futures`, P:376-409). For one engine the order is:

| # | Call | JSON fields (as sent) | Miles accepts when | Miles source |
|---|------|-----------------------|--------------------|--------------|
| 0 | POST /pull_weights, first update only | `local_checkpoint_dir`, `source_dir`, `target_version` = 0 | HTTP < 400 and no `success: false` | P:182-191, P:244; client CL:381-393 |
| 1 | POST /pull_weights | `local_checkpoint_dir`, `source_dir`, `target_version` = N (N counts from 1) | same | P:378-388; T:60-64 |
| 2 | POST /pause_generation | `{"mode": <pause_generation_mode>}`, "retract" in the test | HTTP < 400 (body not read) | P:389-392; CL:466-473; T:33, T:59 |
| 3 | GET /flush_cache, skipped when mode is `in_place` | none | HTTP 200; any other answer or error is retried up to 60 times, 1 s apart, then `TimeoutError` | P:393-394; CL:307-323 |
| 4 | POST /update_weights_from_disk | `model_path` = local checkpoint dir, `weight_version` = str(version); `load_format` is left out | HTTP < 400 and no `success: false` | P:395-406; CL:395-409; T:65 |
| 5 | POST /continue_generation | `{}` | HTTP < 400 (body not read) | P:407-409; CL:475-480 |
| 6 | GET /model_info, only under `--ci-test` and not LoRA | none | `weight_version` equals the str of the version, else `RuntimeError` | A:1100-1101; U:127-133; CL:325-333 |

Notes on the table:

* Step 0 is the baseline: `_capture_baseline` (P:166-250) runs once, at the first `begin_sync`, clears the shared dir
  (`shutil.rmtree` then `makedirs`, P:177-178), sends `pull_weights(target_version=0)` and overlaps it with the
  trainer's snapshot gather. "No delta version is published" for it (DOC:132-134). The same update then goes on to
  publish version 1 and runs steps 1-5 (T:195-204 shows the six calls `pull_weights, pull_weights, pause_generation,
  flush_cache, update_weights_from_disk, continue_generation` with `target_version` 0 then 1).
* The version in step 1 is the file version `_snapshot_file_version`, incremented once per update (P:132); the version
  in step 4 is the trainer's `weight_version` from `ModelCompanionWeightVersionUtils.from_params(...)` (A:1035-1038,
  A:1080), a different number. I did not read how that number is built.
* The pull happens before the pause and overlaps generation; the pause-to-continue window is the activation boundary
  (P:367-370, DOC:147-149).
* Not called by disk-delta: `begin_weight_update`, `end_weight_update`, `update_weight_version`. The class sets
  `use_weight_update_session = False` (P:87), and the updater only calls them when that flag is true (U:151-155,
  U:189-192). The plan text for W0.2 lists them; they belong to the other transfer modes. Also not called per update:
  `check_weights`, `get_weight_version` (the client's `get_weight_version` tries `/model_info` first, CL:327).
* If any step raises, Miles stops: no `continue_generation` is sent after a failed `update_weights_from_disk` (the
  calls sit in one function without a `finally`, P:375-410). A server left paused stays paused. Our replay always sends
  the continue.
* Connection-time calls, outside one update: `GET /health_generate` and `GET /flush_cache` until 200 (CL:48-72).

## What is written to the shared directory, and who builds the checkpoint

Written by the trainer ranks (Miles code), under `--update-weight-disk-dir` (`delta_dir`, P:92):

```
<update_weight_disk_dir>/
  weight_v000001/                                   one directory per published version (P:258; zero-padded to 6)
    model-00000-of-0000K.safetensors                one file per trainer rank that has changed tensors (P:343-349)
    model-00001-of-0000K.safetensors
    model.safetensors.index.json                    written by rank 0 last (P:353-364)
  weight_v000002/ ...
```

* `weight_v000000` is never written (see above). Files are written as `<name>.tmp`, fsynced, then `os.replace`d
  (P:435-441), so a reader sees a whole file or none.
* In each `model-*.safetensors`, the tensors are the changed HF tensors under their HF names, each a 1-D `uint8`
  array holding the zstd (level 1) compression of the delta bytes; the safetensors metadata maps tensor name to the
  checksum of the tensor's complete new bytes (P:298-299, P:347). Encoding `xor`: delta = new XOR old
  (P:287-289; T:191-192 decodes it that way). Encoding `overwrite`: u4 count, u4 positions, then the new values
  (D:21-25). Checksum algorithms: `xxh3-128` (default per DOC:158), `blake3`, `adler32` (D:41-52).
* The index: `{"metadata": {"version", "base_version", "delta_encoding", "compression_format": "zstd",
  "checksum_format"}, "weight_map": {tensor name: file name}}`; `base_version` is the previous version, "000000"
  for the first (P:354-363; T:188). An update with no changed byte writes no model file and an empty `weight_map`
  (T:193-194).
* Duplicate names across pipeline stages are dropped to the lowest rank, and a checksum mismatch between replicas
  raises (P:309-326).
* The base the deltas apply to: each tensor's bytes in `--hf-checkpoint` (a local directory, DOC:174-176); the trainer
  seeds its snapshot from it and checks dtype, shape and byte count against what the trainer emits (P:198-224). Deltas
  work on raw bytes, so the layout must match exactly.

Who builds the checkpoint the engine reloads: not Miles. Miles writes only the delta directory above. Per Miles text
(D:15-18, P:76-81, P:166-173, DOC:141-143), the engine's own `/pull_weights` handler, on every host the engine spans,
(a) at `target_version` 0 materialises the base checkpoint in `local_checkpoint_dir` (`--update-weight-local-
checkpoint-dir`, a host-local path), and (b) at `target_version` N reads `source_dir/weight_v00000N`, verifies lineage
and checksums, and patches the files in `local_checkpoint_dir` in place under a per-host flock. Then
`update_weights_from_disk(model_path=local_checkpoint_dir)` reloads that directory (P:399; the same string is the pull
target and the reload path, T:60-65). Unverified, because that code is not in our SGLang and I did not read it: how the
engine learns where the base checkpoint is (the pull payload carries no base path), and the response fields of
`/pull_weights` (Miles reads only `success`, `error_message` or `error`, S:101-119; an empty body also passes, CL:153).

## Each route against our SGLang and the plugin

Our SGLang: `engine_src/sglang_up/sglang/srt/` (22 November copy under SGLang-Neuron v0.1.0). Plugin:
`patches/sglang_neuron_v0.1.0_inplace/general_plugin.py`. A route "lets through" when no hook of the plugin refuses it
for Miles's payload.

| # | Route | Exists in our SGLang | Payload compatible | Plugin lets it through |
|---|-------|----------------------|--------------------|------------------------|
| 0, 1 | POST /pull_weights | **no.** No such route in `entrypoints/http_server.py`; `srt/weight_sync/` holds only `tensor_bucket.py` and `utils.py`; no `pull_weights` string anywhere under `srt/`. FastAPI answers 404 `{"detail": "Not Found"}` | not applicable. Unknown body fields are ignored in general (`utils/msgspec_utils.py:98`, `extra_behavior="ignore"`), but there is no handler to read them | no hook (nothing to hook). Miles stops here on its first call, the baseline pull |
| 2 | POST /pause_generation | yes, `http_server.py:1821-1831`; handler `tokenizer_manager.py:2400-2413` | yes. `PauseGenerationReqInput.mode: Literal["abort","retract","in_place"] = "abort"` (`io_struct.py:1984-2003`); Miles sends one of them. Answer `{"message": "Generation paused successfully.", "status": "ok"}` | yes at `--pp-size 1`. With `pp_size > 1` the plugin answers HTTP 400 to `in_place` and `retract` and lets only `abort` through (`general_plugin.py:196-211`, registered at :314). Miles's default mode I did not read; the test uses retract (T:33, T:59) |
| 3 | GET /flush_cache | yes, route takes GET and POST, optional query `timeout` (`http_server.py:1015-1031`). 200 with a text body when flushed, 400 when it could not (running requests) | yes, Miles sends no body and no timeout | yes. The plugin refuses only a flush with `timeout > 0` under PP (`general_plugin.py:214-232`, registered :315); `_flush_cache_drained` (:928) drains device launches first. A 400 here makes Miles retry (see table) |
| 4 | POST /update_weights_from_disk | yes, `http_server.py:1290-1316`; manager `tokenizer_manager.py:2421-2455`. Answer `{"success", "message", "num_paused_requests"}`, HTTP 200, or 400 with `success: false` | yes. `UpdateWeightFromDiskReqInput` requires `model_path`; `weight_version` optional; `load_format` omitted is filled from the config (`io_struct.py:2027-2049`, `tokenizer_manager.py:2429-2430`). After success `weight_version` is stored and `model_path` and `served_model_name` become the posted path (`tokenizer_manager.py:2451-2453, :2502-2507`) | **yes, by the experiment of 20 November.** `update_weights_from_disk` is NOT in `_WEIGHT_UPDATES` (`general_plugin.py:67-74`, comment at :68-69); the runner side is replaced by `_update_weights_in_place` (:895-925, registered :438), which calls `ModelHost.update_weights_in_place(path)` (`models/host.py:120`). That reads a full checkpoint of the served model (rebuild on the host with the first-load loaders, `copy_` into the live parameters), or a LoRA folder if `adapter_config.json` is present (`host.py:133`). Per the code read; I did not run it |
| 5 | POST /continue_generation | yes, `http_server.py:1834-1843` | yes. `ContinueGenerationReqInput.torch_empty_cache: bool = True`; Miles sends `{}` | yes, no hook |
| 6 | GET /model_info | yes, `http_server.py:811-` returns `weight_version` among others (also `/get_model_info`, deprecated) | yes, Miles reads `weight_version` only | yes, no hook |

Other routes of the same client that Miles's other transfer modes use, checked only for existence and field names
(not part of disk-delta): `update_weight_version` exists (`http_server.py:1517`, fields `new_version`,
`abort_all_requests` match CL:495-499); `begin_weight_update` / `end_weight_update` exist (`:1448`, `:1463`) but the
client sends `sync_base` and `expected_lora_checksums`, fields our `BeginWeightUpdateReqInput` /
`EndWeightUpdateReqInput` do not have (ignored on receipt); `weights_checker` exists (`:1589`, `skip_tensor_list`
matches CL:376). None of the three is in `_WEIGHT_UPDATES`, so the plugin does not refuse them; what the Neuron
scheduler does with them I did not read. `update_weights_from_tensor`, `update_weights_from_distributed`,
`update_weights_from_ipc`, `init_weights_update_group`, `init_weights_send_group_for_remote_instance` are refused with
HTTP 400 by the plugin (`general_plugin.py:67-74, :189-193`).

## Gaps

1. **`/pull_weights` does not exist in our SGLang.** Miles's first call of its first update (the baseline pull)
   answers 404 and Miles raises. This is the one missing route of the sequence; the plan's "present" expectation holds
   for steps 2-6 and fails for the pull. The plan's list of absent calls ("`pull_weights`, `get_weight_version`,
   `check_weights`") is half right: `get_weight_version` is not needed (the client falls back to `/model_info`, which
   exists) and `check_weights` is the route `/weights_checker`, which exists, and is not on this path.
2. The shim has to do the engine's job, not only answer the route: materialise the base in `local_checkpoint_dir` for
   `target_version` 0 and patch it per version (read index, per-tensor zstd decompress, XOR or overwrite onto the raw
   bytes, compare checksum, write). It has to run on each host the engine spans (one host on a trn2 single node). It
   needs the base checkpoint path, which the Miles payload does not carry (so the shim takes it as its own setting).
   Alternative named in the plan: skip delta and write the full checkpoint to the local dir ourselves, giving Miles an
   endpoint that answers `{"success": true}`.
3. The reload target must be a full HF checkpoint directory that the Neuron loaders accept, byte-identical in layout to
   what the shim patched; our in-place update re-runs the model's loaders on the whole directory every time (cost per
   update is the plan's 0.34-0.41 s figure, which I did not re-measure). Anything else in the directory (an
   `adapter_config.json`) switches the hook to its LoRA branch.
4. After a successful reload our server reports `model_path` and `served_model_name` as the local checkpoint path
   (`tokenizer_manager.py:2502-2507`). Clients that send `"model": <old name>` should be checked; Miles's sequence
   does not read it back.
5. Failure leaves the server paused: Miles has no `finally` around steps 2-5 (P:375-410). Any shim or checkpoint error at
   step 4 gives HTTP 400, Miles raises, and `continue_generation` is not sent.
6. Under `--pp-size > 1` the pause must be `abort` (plugin). Not our configuration at 0.6B, but the Miles default mode
   should be read before a PP run.
7. `/flush_cache` returns 400 until the server is idle; Miles retries every second for up to 60 s. After a `retract`
   pause the running requests are moved back to the waiting queue (`scheduler.py:6329-6345`), and a paused scheduler's
   flush ignores the waiting queue (`scheduler.py:5920`, `is_fully_idle(ignore_waiting=self._engine_paused)`), so it
   should pass at once; not measured here, and the Neuron plugin's `_flush_cache_drained` drain was not timed.
8. Not checked, outside the call sequence: the Miles argument defaults (`pause_generation_mode`, delta encoding),
   `GeneralHttpClientProvider` timeouts, and what `weight_version` value Miles computes.

## The stand-in server and the replay

`selftest_fake_server.py` answers these routes the way the code above says our SGLang does: pause (200, 422 for a mode
outside the three), flush (GET and POST, 200 text), `update_weights_from_disk` (200 `{"success": true, ...}` when the
path has a `config.json` and a `.safetensors` file, else 400 `{"success": false, ...}`, 422 without `model_path`),
continue, `/model_info` (a subset of the real fields), `/ready` (503 while paused), `update_weight_version`, 404 for
`/pull_weights`, and a chat request waits while paused (as `tokenizer_manager.py:978-979` does). It does not load any
weights, has no running requests to retract, and its `/model_info` has far fewer fields than the real one.

`64_miles_replay.py --base-url URL --ckpt PATH_AS_THE_SERVER_SEES_IT` sends steps 1-6 (add `--baseline` for step 0,
`--pause-mode in_place` to skip the flush, `--no-verify` to skip step 6) and writes
`results/64_miles_replay_<tag>.jsonl` with per-call status and seconds, whether Miles would accept the answer, and a
summary row (`miles_would_stop_at`, `pause_to_continue_seconds`). Against a real SGLang-Neuron today it should show
`pull_weights` 404 and, if `--ckpt` is a valid checkpoint, the other calls accepted. Against the stand-in it is part of
`selftest_fake_server.py` (the "64 replay" checks).
