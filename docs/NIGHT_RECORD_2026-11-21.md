# Night session record, 2026-11-21 (Seattle) / 2026-11-22 UTC: generic in-place update, sampler, 8B sync, decode profile, shape cost

Times are UTC. Every number is read from a result file in `results_night_2026-11-21/results_night/` (archive
`results_night.tgz`, sha256 `30f95a6ddb164643…`, equal on the machine and the laptop; raw profile sessions in
`night_profiles.tgz`, sha256 `20fe096fea2a9012…`). One run each unless a row says otherwise. Kit: `sglang_rl/`
(`run_night_1121.sh`, `run_shape_1122.sh`, scripts 48, 50, 58, 59, 59b-d, 60, `mk_alt.py`).

## 1. Machine

| id | type | region | from – to | notes |
|---|---|---|---|---|
| `<instance id redacted>` (spot `<spot request id redacted>`) | trn2.48xlarge | ap-south-2 | 07:58 – 09:43 | powered off from inside (`shutdown -P now`; shutdown behaviour terminate). The AWS login had expired at that moment, so the terminated state and the absence of a left-over volume were NOT read back from AWS that night; read back on 22 November: no trn2 instance, no left-over volume. Guard was 14:01. |

Image `pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04`, patch
`patches/sglang_neuron_v0.1.0_inplace` with two additions of this session in `models/host.py`: a coverage log line
(live parameters and buffers, staged against live) and a timing split of the full update (host graph, loaders, copy);
`RLKIT_KEEP_STAGED` was added and is of no use (see 4). All servers TP4 on four logical cores, context 1024, one decode
bucket of 32 (`--cuda-graph-bs-decode 32`) unless a row says otherwise.

## 2. In-place full-checkpoint update on other model families (`48_inplace_*.jsonl`)

| model | graph | live tensors | update | back to base | in place vs fresh start |
|---|---|---|---|---|---|
| Qwen3-8B | shared dense transformer | 291 parameters, 1 buffer (`rotary_emb.inv_freq`) | 4.7-6.2 s (12 updates, two servers) | same text, logprob diff 0.0 | same text, logprob diff 0.0 |
| GLM-4-9B-0414 | shared dense transformer | 363 parameters, 1 buffer | 4.9 s (2 updates) | same text, 0.0 | same text, 0.0 |
| DeepSeek-V2-Lite | its own graph (MLA, MoE) | 351 parameters, 1 buffer | 10.3-10.6 s (2 updates) | same text, 0.0 | same text, 0.0 |

- No change to the patch was made for GLM-4 or DeepSeek-V2. In all three the only buffer is `inv_freq`, equal in the
  staged and the live graph, so walking `named_parameters()` covers every weight of these families.
- GLM-4: the perturbed checkpoint changed the logprobs (max diff 0.087) but not the 48-token texts.
- DeepSeek-V2-Lite, first attempt: died at start on a compile-cache race (`/cache/neff/.KERNEL_SOURCES.194` renamed by
  another server starting on the same empty cache), not a refusal. A start with the example's own flags (context 4096)
  was still warming after one hour and was stopped by hand; the reduced shapes were ready in 953 s.
- Not covered: Qwen3-MoE, GPT-OSS, GLM-5.2, Qwen3.5, Kimi-K3 (kernel models with packed or quantized weights).

## 3. Sampler (`58_sampler_s1.jsonl`, `58_sampler_b1.jsonl`), Qwen3-0.6B

- `SGLANG_NEURON_SAMPLER=topk_topp`: five (top_p, top_k) settings all HTTP 200. `temperature` mode: 200 for the
  full vocabulary, 400 for the four filtered settings (message: use a topk_topp server for filtered sampling).
- Speed, 256 requests of up to 256 tokens, 32 in flight, three passes: `topk_topp` full vocabulary 2,713-2,730 tok/s,
  `topk_topp` with top_p 0.95 / top_k 20 2,700-2,764 tok/s, `temperature` 2,183-2,218 tok/s. One server each.
- Where top-p cuts: 8 tests on 4 prompts, 400 one-token samples each at temperature 1, top_p set inside the mass of
  the token of rank j. In all 8 the number of distinct tokens returned was j (the token that crosses p is kept), as in
  vLLM and standard SGLang; the vllm-neuron behaviour reported by a teammate (crossing token dropped) was not seen here.

## 4. Qwen3-8B weight update, where the time goes (`48_inplace_q8_*.jsonl`, server logs)

- Full checkpoint: per rank, host graph built in 0.00-0.02 s, loaders 3.6-4.7 s, copy to the device 0.3-1.1 s. The
  assumption that rebuilding the host graph is the cost was wrong; keeping it between updates changes nothing.
- Adapter only (`50_adapter_q8.jsonl`, rank 16 on seven projections): first update 35.8 s, then 13.9-14.0 s, of which
  13.5-13.6 s is forming the deltas on the host and 0.3 s the copy. Correct: equal to a fresh start on the merged
  checkpoint (same text, logprob diff 0.0), adapters replace each other, the zero adapter gives the base back.
  So at 8B the adapter path is slower than the full checkpoint (5-6 s).

## 5. Decode profile, Qwen3-0.6B, 32 requests of up to 256 tokens

System profile (`59_decode_profile_p1.jsonl`, `59b_profile_sum.txt`, two of four ranks read):

- Pass wall time 2.67 s with and without the profile window. 287 program executions per physical core in a 2.66 s
  window; execution median 9.13 ms; executions sum to 2,653 ms of a 2,658 ms span per core. Gap between consecutive
  executions: median 6-9 µs, total about 5 ms. Collective barrier waits 9.8 ms in total.
- Reading: the device runs back to back; host, scheduler and collectives are not where the time is.

Device profile (launch-time window, TP rank 0, 4 decode steps; `59d_devprof_sum.txt`, `devprof_pq/summary.json`):

- 4 executions of 9.2-9.3 ms. Summary fields: `transpose_flops` 45.2e9, `hardware_flops` 92.3e9, `model_flops`
  48.4e9; `mfu_estimated_percent` 0.0031, `mbu_estimated_percent` 0.091; active time over the 4 executions (39.5 ms):
  DMA 21.7 ms, GpSimd engine 14.0 ms, scalar 13.2 ms, sync 11.7 ms, vector 10.4 ms, tensor (matmul) engine 8.9 ms.
- The profiler warns that dynamic DMA values may be inaccurate without DGE notifications and that the NEFF has no
  compiler metrics. Per-operation attribution was not done.

## 6. What an unfilled compiled shape costs (`60_shape_cost_{A,B,C,D}.jsonl`), Qwen3-0.6B, context ladder off

Step = wall time / longest answer. Same GSM8K questions at every level; answers up to 256 tokens.

| concurrent requests | A: one bucket (32), ctx 1024 | B: default buckets, ctx 1024 | C: one bucket, ctx 4096 | D: one bucket, ctx 8192 |
|---|---|---|---|---|
| 32 | 10.4 ms | 9.3 ms | 14.8 ms | 26.5 ms |
| 16 | 9.8 | 7.1 | 14.2 | 25.9 |
| 8 | 9.5 | 5.8 | 13.9 | 25.5 |
| 4 | 9.3 | 5.1 | 13.7 | 25.4 |
| 2 | 9.3 | 5.0 | 13.6 | 25.3 |
| 1 | 9.3 | 4.6 | 13.7 | 25.2-25.4 |

- With one bucket a step costs the same whether 1 or 32 rows are live. With the default buckets one request runs at
  4.6 ms a step (215 tok/s) against 9.3 ms (108 tok/s).
- The step grows with the compiled context although the requests are the same short ones: 10.4 / 14.8 / 26.5 ms at
  1024 / 4096 / 8192.
- Row fill of one group of 32 answers: 0.58-0.63 (tokens / (32 x longest answer)); 1,790-2,030 tok/s for a single
  group against 2,713-2,730 with 256 requests queued (section 3).
- Server ready time: A 109 s, B 350 s, C 215 s, D 230 s (the cache already held some shapes).

## 7. Not done

Upstream PR #10 together with the patch; training-side padding share; 8B training memory; teacher scoring speed;
per-operation reading of the device profile (a teammate's kernel on/off runs cover that line).
