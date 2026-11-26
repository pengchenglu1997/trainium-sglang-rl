# SGLang-Neuron optimisation sweep, Qwen3-0.6B (r13)

Numbers only. Each cell shows output tokens per second (llmbench output_token_throughput_tps) and the median time to first token in seconds (ttft_p50_s), as "tok/s / TTFT p50". The released defaults are the image's example script qwen3_0p6b/run_server.sh with its own flags (--tp-size 4 --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240 --mem-fraction-static 0.4), the kit's sampler temperature, cores 0-3, compile cache folder sgl_cache_v0.1.0_release.

- result file: 81_opt_sweep_r13.jsonl; baseline files: llmbench_q06_vllm.csv (found), SGLang-Neuron baseline row from llmbench_q06_sgl.csv; repeat file llmbench_q06_sgl_rep.csv (found)

## Prompt 1024 tokens, 256 output tokens

| row | what differs from the released defaults | ready s | c=1 | c=4 | c=8 | c=32 | same text as V0 (of 5) | marks |
|---|---|---|---|---|---|---|---|---|
| vllm-neuron 0.29.0.1.2.0 (baseline) | TP4, max-model-len 4096, max-num-seqs 32 (76_baseline_pair.sh) | 461 | 209.9 / 0.021 | 766.7 / 0.047 | 1,230.3 / 0.076 | 2,375.3 / 0.276 | - |  |
| SGLang-Neuron v0.1.0 released defaults (V0) | none | 1,219 | 211.1 / 0.025 | 679.0 / 0.047 | 1,168.0 / 0.075 | 2,460.6 / 0.260 | reference |  |
| V19: sampler kernel (temperature sampler), 4 x TP1 containers | 4 x TP1 on the same four cores; 4 containers on cores 40,41,42,43; SGL_EXTRA='--tp-size 1 --max-total-tokens 65536 --max-running-requests 8' in each | 1,637 (replicas: 712, 315, 310, 300) | 222.8 / 0.039 | 890.6 / 0.041 | 1,437.9 / 0.057 | 3,098.2 / 0.138 | 5 |  |
| V20: sampler kernel (temperature sampler), --dp-size 4 of TP1 in one server | one container, --dp-size 4 of TP1 on cores 0-3; SGL_CORES=40-43; SGL_EXTRA='--tp-size 1 --dp-size 4 --max-total-tokens 65536 --max-running-requests 8' | 335 | 223.1 / 0.039 | - | 1,420.4 / 0.050 | 3,025.7 / 0.170 | 5 |  |
| V21: sampler kernel (temperature sampler), TP4 | SGL_CORES=40-43 | 1,099 | 320.7 / 0.025 | - | 1,628.9 / 0.075 | 2,744.4 / 0.254 | 5 |  |

## Replica rows

Rows V19 are a different serving layout on the same cores, not a kernel. A cell of such a row sums the tokens per second of one llm-bench process per replica (each at concurrency c / number of replicas, started together; concurrency 1 is one replica alone) and takes the median of their TTFT p50 values (the mean of the two middle ones for an even count). Each process ran 64 requests, so a cell holds 64 x replicas requests. V7 is "4 x TP1 on the same four cores". The processes are not time-aligned beyond starting together:

Each replica process runs 64 requests at concurrency c / replicas, so it has 64 / (c / replicas) waves of requests (the ramp and the tail of a wave weigh more with more waves; the baseline at c=32 has 2 waves). Replica k uses the llm-bench seed 42 + k (the baseline run and replica 0 use 42), so the replicas do not send one prompt sequence.

| row | concurrency | replicas | waves per replica | launch spread (s) | end spread (s) | tok/s of each replica |
|---|---|---|---|---|---|---|
| V19 | 1 | 1 | 64 | 0.0 | 0.0 | 222.8 |
| V19 | 4 | 4 | 64 | 0.0 | 0.13 | 222.5, 222.6, 222.8, 222.6 |
| V19 | 8 | 4 | 32 | 0.0 | 0.29 | 359.9, 359.4, 358.1, 360.5 |
| V19 | 32 | 4 | 8 | 0.0 | 0.06 | 774.5, 775.1, 773.4, 775.2 |

## Guard answers

Five fixed prompts, chat route, temperature 0, 48 new tokens, on the first replica; "same" is the text equal to V0's for that prompt. A tensor-parallel layout may change a late token: it is recorded, it does not fail a variant.

| variant | status | same text per prompt | guard ok on every replica |
|---|---|---|---|
| V0 | ok | reference | yes |
| V19 | ok | same same same same same | yes |
| V20 | ok | same same same same same | yes |
| V21 | ok | same same same same same | yes |

## Cache and warm-up records

Files in the shared compile-cache folder before and after the variant, and the engine's warm-up line of each server ("warmup done in <s>s (<n> shapes)"): a variant that adds no files started warm.

| variant | cache files before | after | warm-up lines |
|---|---|---|---|
| V0 | 505 | 508 | [neuron] warmup done in 358.2s (21 shapes); [neuron] warmup done in 358.2s (21 shapes); [neuron] warmup done in 358.2s (21 shapes); [neuron] warmup done in 358.2s (21 shapes) |
| V19 | 508 | 512 | [neuron] warmup done in 552.3s (12 shapes); [neuron] warmup done in 115.3s (3 shapes) / [neuron] warmup done in 181.6s (12 shapes); [neuron] warmup done in 82.8s (3 shapes) / [neuron] warmup done in 177.8s (12 shapes); [neuron] warmup done in 82.8s (3 shapes) / [neuron] warmup done in 180.2s (12 shapes); [neuron] warmup done in 82.1s (3 shapes) |
| V20 | 512 | 512 | [neuron] warmup done in 177.0s (12 shapes); [neuron] warmup done in 177.3s (12 shapes); [neuron] warmup done in 177.6s (12 shapes); [neuron] warmup done in 177.8s (12 shapes) |
| V21 | 512 | 512 | [neuron] warmup done in 965.1s (21 shapes); [neuron] warmup done in 965.1s (21 shapes); [neuron] warmup done in 965.1s (21 shapes); [neuron] warmup done in 965.1s (21 shapes) |

## Notes

- Each cell is one run of 64 requests (after 4 warm-up requests) on one server start; a replica row holds one such run per replica.
- Baseline repeat spread, prompt 1024 (|pass 1 - pass 2| / mean of the two, SGLang-Neuron released defaults, llmbench_q06_sgl_rep.csv): concurrency 1: tok/s 0.0 %, TTFT p50 0.4 %; concurrency 2: tok/s 0.0 %, TTFT p50 2.6 %; concurrency 4: tok/s 0.2 %, TTFT p50 2.3 %; concurrency 8: tok/s 0.1 %, TTFT p50 0.2 %; concurrency 16: tok/s 0.0 %, TTFT p50 2.9 %; concurrency 32: tok/s 0.2 %, TTFT p50 0.3 %.
- A mark names a cell with failed requests, a cell whose mean output tokens differ from the requested 256, a guard that failed, or a status other than ok. A row with a mark is shown as measured.
- Ready s is the time from the start command to the first answered request, with the shared compile cache in the state the earlier variants left it (a first start of a new shape includes its compile).
- Several rows do the same device work as V0 at some measured grid points, so their tok/s there is expected to equal V0's within the run-to-run spread, and what they can change is the ready time: V3 and V5 run the 1, 8 and 32 row decode graphs at c=1, 8 and 32 exactly as V0 does (only the ramp and the tail of a wave differ); V4 has one 32-row graph, so only c=32 runs the graph V0 runs (c=1 and c=8 pad to 32 rows); V10 runs the 2048-context decode graph under both context ladders, since every row is at most 1024 + 256 tokens; V9 at prompt 1024 fits one prefill tile under chunk 2048 and under chunk 4096, so only its prompt-3072 cells can show a change; V12 changes the host's streaming only. Read a difference at those points against the repeat spread above, and read ready s and the cache and warm-up records of variant_end (cache_files_before/after, warmup_lines) for the effect these rows can have.
- Rows are not compared with each other in this file and none is ranked.

