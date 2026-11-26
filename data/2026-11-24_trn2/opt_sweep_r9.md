# SGLang-Neuron optimisation sweep, Qwen3-0.6B (r9)

Numbers only. Each cell shows output tokens per second (llmbench output_token_throughput_tps) and the median time to first token in seconds (ttft_p50_s), as "tok/s / TTFT p50". The released defaults are the image's example script qwen3_0p6b/run_server.sh with its own flags (--tp-size 4 --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240 --mem-fraction-static 0.4), the kit's sampler temperature, cores 0-3, compile cache folder sgl_cache_v0.1.0_release.

- result file: 81_opt_sweep_r9.jsonl; baseline files: llmbench_q06_vllm.csv (found), SGLang-Neuron baseline row from llmbench_q06_sgl.csv; repeat file llmbench_q06_sgl_rep.csv (found)

## Prompt 1024 tokens, 256 output tokens

| row | what differs from the released defaults | ready s | c=1 | c=4 | c=8 | c=32 | same text as V0 (of 5) | marks |
|---|---|---|---|---|---|---|---|---|
| vllm-neuron 0.29.0.1.2.0 (baseline) | TP4, max-model-len 4096, max-num-seqs 32 (76_baseline_pair.sh) | 461 | 209.9 / 0.021 | 766.7 / 0.047 | 1,230.3 / 0.076 | 2,375.3 / 0.276 | - |  |
| SGLang-Neuron v0.1.0 released defaults (V0) | none | 1,219 | 211.1 / 0.025 | 679.0 / 0.047 | 1,168.0 / 0.075 | 2,460.6 / 0.260 | reference |  |
| V1: sampler greedy | SGL_SAMPLER=greedy (kit default: temperature) | 1,054 | 311.8 / 0.024 | - | 1,608.8 / 0.073 | 2,781.6 / 0.259 | 5 |  |
| V2: sampler topk_topp (decode compiles at -O1) | SGL_SAMPLER=topk_topp (kit default: temperature) | 747 | 218.1 / 0.026 | - | 1,266.0 / 0.084 | 2,607.0 / 0.271 | 5 |  |
| V7: 4 x TP1 on the same four cores (four containers; a different serving layout, not a kernel) | 4 x TP1 on the same four cores; 4 containers on cores 0,1,2,3; SGL_EXTRA='--tp-size 1 --max-total-tokens 65536 --max-running-requests 8' in each | 1,642 (replicas: 722, 300, 310, 310) | 192.4 / 0.038 | 768.6 / 0.040 | 1,188.6 / 0.057 | 2,755.7 / 0.140 | 5 |  |
| V8: 2 x TP2 on cores 0-1 and 2-3 (two containers; a different serving layout, not a kernel) | 2 x TP2 on cores 0-1 and 2-3; 2 containers on cores 0-1,2-3; SGL_EXTRA='--tp-size 2 --max-total-tokens 131072 --max-running-requests 16' in each | 1,298 (replicas: 948, 350) | 129.1 / 0.033 | 244.6 / 0.044 | 460.5 / 0.068 | 1,513.1 / 0.203 | 4 | text differs from V0 on 1 of 5 guard prompts |
| V11: combination V1, V7 | 4 x TP1 on the same four cores; SGL_SAMPLER=greedy (kit default: temperature); 4 containers on cores 0,1,2,3; SGL_EXTRA='--tp-size 1 --max-total-tokens 65536 --max-running-requests 8' in each | 2,440 (replicas: 1,336, 361, 372, 371) | 1.5 / 0.041 | - | - | - | 5 | bench_fail; failed requests 57 of 64 (prompt 1024, concurrency 1) |
| V13: one container, --dp-size 4 of TP1 on cores 0-3, 8 running requests per replica (SGLang's data-parallel controller) | one container, --dp-size 4 of TP1 on cores 0-3; SGL_EXTRA='--tp-size 1 --dp-size 4 --max-total-tokens 65536 --max-running-requests 8' | 330 | 192.2 / 0.040 | - | 1,172.4 / 0.052 | 2,701.1 / 0.166 | 5 |  |
| V14: chunked prefill size 1024 | SGL_EXTRA='--chunked-prefill-size 1024' | 451 | 211.0 / 0.025 | - | 1,162.1 / 0.075 | 2,458.7 / 0.267 | 5 |  |
| V16: NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=1 (platform default 0) | none | 456 | 210.7 / 0.025 | - | 1,159.5 / 0.075 | 2,455.9 / 0.261 | 5 |  |
| V18: sampler greedy and --dp-size 4 of TP1 on cores 0-3 (V1 + V13 in one server) | one container, --dp-size 4 of TP1 on cores 0-3; SGL_SAMPLER=greedy (kit default: temperature); SGL_EXTRA='--tp-size 1 --dp-size 4 --max-total-tokens 65536 --max-running-requests 8' | - | - | - | - | - | - | not_ready |

## Prompt 3072 tokens, 256 output tokens (V9 and the baseline)

| row | what differs from the released defaults | ready s | c=1 | c=32 | same text as V0 (of 5) | marks |
|---|---|---|---|---|---|---|
| vllm-neuron 0.29.0.1.2.0 (baseline) | TP4, max-model-len 4096, max-num-seqs 32 (76_baseline_pair.sh) | 461 | 204.7 / 0.051 | 1,909.8 / 0.719 | - |  |
| SGLang-Neuron v0.1.0 released defaults (V0) | none | 1,219 | 203.6 / 0.055 | 1,974.8 / 0.357 | reference |  |
| V14: chunked prefill size 1024 | SGL_EXTRA='--chunked-prefill-size 1024' | 451 | 202.3 / 0.063 | 1,972.9 / 0.356 | 5 |  |

## Replica rows

Rows V7, V8, V11 are a different serving layout on the same cores, not a kernel. A cell of such a row sums the tokens per second of one llm-bench process per replica (each at concurrency c / number of replicas, started together; concurrency 1 is one replica alone) and takes the median of their TTFT p50 values (the mean of the two middle ones for an even count). Each process ran 64 requests, so a cell holds 64 x replicas requests. V7 is "4 x TP1 on the same four cores". The processes are not time-aligned beyond starting together:

Each replica process runs 64 requests at concurrency c / replicas, so it has 64 / (c / replicas) waves of requests (the ramp and the tail of a wave weigh more with more waves; the baseline at c=32 has 2 waves). Replica k uses the llm-bench seed 42 + k (the baseline run and replica 0 use 42), so the replicas do not send one prompt sequence.

| row | concurrency | replicas | waves per replica | launch spread (s) | end spread (s) | tok/s of each replica |
|---|---|---|---|---|---|---|
| V7 | 1 | 1 | 64 | 0.0 | 0.0 | 192.4 |
| V7 | 4 | 4 | 64 | 0.0 | 0.45 | 192.1, 192.2, 192.2, 192.1 |
| V7 | 8 | 4 | 32 | 0.0 | 0.3 | 297.6, 297.3, 295.9, 297.6 |
| V7 | 32 | 4 | 8 | 0.0 | 0.05 | 688.8, 689.5, 688.5, 688.9 |
| V8 | 1 | 1 | 64 | 0.0 | 0.0 | 129.1 |
| V8 | 4 | 2 | 32 | 0.0 | 0.11 | 122.3, 122.3 |
| V8 | 8 | 2 | 16 | 0.0 | 0.05 | 230.2, 230.3 |
| V8 | 32 | 2 | 4 | 0.0 | 0.0 | 756.5, 756.5 |
| V11 | 1 | 1 | 64 | 0.0 | 0.0 | 1.5 |
| V11 | 4 | 4 | 64 | None | None | - |
| V11 | 8 | 4 | 32 | None | None | - |
| V11 | 32 | 4 | 8 | None | None | - |

## Guard answers

Five fixed prompts, chat route, temperature 0, 48 new tokens, on the first replica; "same" is the text equal to V0's for that prompt. A tensor-parallel layout may change a late token: it is recorded, it does not fail a variant.

| variant | status | same text per prompt | guard ok on every replica |
|---|---|---|---|
| V0 | ok | reference | yes |
| V1 | ok | same same same same same | yes |
| V2 | ok | same same same same same | yes |
| V7 | ok | same same same same same | yes |
| V8 | ok | same same differs same same | yes |
| V11 | bench_fail | same same same same same | yes |
| V13 | ok | same same same same same | yes |
| V14 | ok | same same same same same | yes |
| V16 | ok | same same same same same | yes |
| V18 | not_ready | - | - |

## Cache and warm-up records

Files in the shared compile-cache folder before and after the variant, and the engine's warm-up line of each server ("warmup done in <s>s (<n> shapes)"): a variant that adds no files started warm.

| variant | cache files before | after | warm-up lines |
|---|---|---|---|
| V0 | 223 | 223 | [neuron] warmup done in 342.7s (21 shapes); [neuron] warmup done in 342.7s (21 shapes); [neuron] warmup done in 342.7s (21 shapes); [neuron] warmup done in 342.7s (21 shapes) |
| V1 | 296 | 320 | [neuron] warmup done in 924.8s (21 shapes); [neuron] warmup done in 924.8s (21 shapes); [neuron] warmup done in 924.8s (21 shapes); [neuron] warmup done in 924.8s (21 shapes) |
| V2 | 413 | 497 | [neuron] warmup done in 592.2s (21 shapes); [neuron] warmup done in 592.2s (21 shapes); [neuron] warmup done in 592.2s (21 shapes); [neuron] warmup done in 592.2s (21 shapes) |
| V7 | 223 | 296 | [neuron] warmup done in 555.4s (12 shapes); [neuron] warmup done in 127.8s (3 shapes) / [neuron] warmup done in 176.1s (12 shapes); [neuron] warmup done in 83.1s (3 shapes) / [neuron] warmup done in 177.0s (12 shapes); [neuron] warmup done in 83.1s (3 shapes) / [neuron] warmup done in 178.8s (12 shapes); [neuron] warmup done in 84.1s (3 shapes) |
| V8 | 328 | 413 | [neuron] warmup done in 781.0s (15 shapes); [neuron] warmup done in 781.0s (15 shapes); [neuron] warmup done in 113.6s (3 shapes); [neuron] warmup done in 113.6s (3 shapes) / [neuron] warmup done in 242.0s (15 shapes); [neuron] warmup done in 242.0s (15 shapes); [neuron] warmup done in 70.4s (3 shapes); [neuron] warmup done in 70.4s (3 shapes) |
| V11 | 500 | 512 | - / - / - / - |
| V13 | 296 | 296 | [neuron] warmup done in 177.1s (12 shapes); [neuron] warmup done in 177.6s (12 shapes); [neuron] warmup done in 178.2s (12 shapes); [neuron] warmup done in 178.9s (12 shapes) |
| V14 | 320 | 328 | [neuron] warmup done in 340.1s (21 shapes); [neuron] warmup done in 340.1s (21 shapes); [neuron] warmup done in 340.1s (21 shapes); [neuron] warmup done in 340.1s (21 shapes) |
| V16 | 328 | 328 | [neuron] warmup done in 342.9s (21 shapes); [neuron] warmup done in 342.9s (21 shapes); [neuron] warmup done in 342.9s (21 shapes); [neuron] warmup done in 342.9s (21 shapes) |
| V18 | 497 | 500 | - |

## V11 (combination)

V11 rule: per group (sampler V1/V2, bucket setting V5/V3/V4, layout V6/V8/V7) take the finished variant (status ok, no mark for failed requests or output length) with the highest tokens per second at concurrency 32, prompt 1024, that is strictly above V0's; equal values go to the first listed in the group; a group with no such variant keeps the default; V11 runs only when at least two groups have a variant, else it would repeat one variant or the default. "Strictly above V0" means above V0 by more than a margin: the larger of 3 % and the baseline repeat spread at concurrency 32, prompt 1024 (llmbench_<baseline-tag>_sgl_rep.csv against llmbench_<baseline-tag>_sgl.csv: |a - b| / mean of the two); a smaller gain is run-to-run noise, since V3, V4 and V5 run the same 32-row decode graph as V0 at concurrency 32 and V0's number is from another server start.

- chosen: {'sampler': 'V1', 'layout': 'V7'}
- margin 3.0 % = larger of 3 % and the baseline repeat spread at concurrency 32, prompt 1024 (0.2 %)
- V1: 2,781.6 tok/s, above V0 2,460.6 by more than 3.0 %
- V2: 2,607.0 tok/s, above V0 2,460.6 by more than 3.0 %
- V5: no result
- V3: no result
- V4: no result
- V6: no result
- V8: 1,513.1 tok/s, not above V0 2,460.6 by more than 3.0 % (needs more than 2,534.4)
- V7: 2,755.7 tok/s, above V0 2,460.6 by more than 3.0 %

## Notes

- Each cell is one run of 64 requests (after 4 warm-up requests) on one server start; a replica row holds one such run per replica.
- Baseline repeat spread, prompt 1024 (|pass 1 - pass 2| / mean of the two, SGLang-Neuron released defaults, llmbench_q06_sgl_rep.csv): concurrency 1: tok/s 0.0 %, TTFT p50 0.4 %; concurrency 2: tok/s 0.0 %, TTFT p50 2.6 %; concurrency 4: tok/s 0.2 %, TTFT p50 2.3 %; concurrency 8: tok/s 0.1 %, TTFT p50 0.2 %; concurrency 16: tok/s 0.0 %, TTFT p50 2.9 %; concurrency 32: tok/s 0.2 %, TTFT p50 0.3 %.
- A mark names a cell with failed requests, a cell whose mean output tokens differ from the requested 256, a guard that failed, or a status other than ok. A row with a mark is shown as measured.
- Ready s is the time from the start command to the first answered request, with the shared compile cache in the state the earlier variants left it (a first start of a new shape includes its compile).
- Several rows do the same device work as V0 at some measured grid points, so their tok/s there is expected to equal V0's within the run-to-run spread, and what they can change is the ready time: V3 and V5 run the 1, 8 and 32 row decode graphs at c=1, 8 and 32 exactly as V0 does (only the ramp and the tail of a wave differ); V4 has one 32-row graph, so only c=32 runs the graph V0 runs (c=1 and c=8 pad to 32 rows); V10 runs the 2048-context decode graph under both context ladders, since every row is at most 1024 + 256 tokens; V9 at prompt 1024 fits one prefill tile under chunk 2048 and under chunk 4096, so only its prompt-3072 cells can show a change; V12 changes the host's streaming only. Read a difference at those points against the repeat spread above, and read ready s and the cache and warm-up records of variant_end (cache_files_before/after, warmup_lines) for the effect these rows can have.
- Rows are not compared with each other in this file and none is ranked.

