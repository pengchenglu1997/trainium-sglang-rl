# SGLang-Neuron optimisation sweep, Qwen3-0.6B (r15)

Numbers only. Each cell shows output tokens per second (llmbench output_token_throughput_tps) and the median time to first token in seconds (ttft_p50_s), as "tok/s / TTFT p50". The released defaults are the image's example script qwen3_0p6b/run_server.sh with its own flags (--tp-size 4 --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240 --mem-fraction-static 0.4), the kit's sampler temperature, cores 0-3, compile cache folder sgl_cache_v0.1.0_release.

- result file: 81_opt_sweep_r15.jsonl; baseline files: llmbench_q06_vllm.csv (found), SGLang-Neuron baseline row from llmbench_q06_sgl.csv; repeat file llmbench_q06_sgl_rep.csv (found)

## Prompt 1024 tokens, 256 output tokens

| row | what differs from the released defaults | ready s | c=1 | c=8 | c=32 | same text as V0 (of 5) | marks |
|---|---|---|---|---|---|---|---|
| vllm-neuron 0.29.0.1.2.0 (baseline) | TP4, max-model-len 4096, max-num-seqs 32 (76_baseline_pair.sh) | 461 | 209.9 / 0.021 | 1,230.3 / 0.076 | 2,375.3 / 0.276 | - |  |
| SGLang-Neuron v0.1.0 released defaults (V0) | none | 1,219 | 211.1 / 0.025 | 1,168.0 / 0.075 | 2,460.6 / 0.260 | reference |  |
| V22: V20 plus --stream-interval 4 (answers streamed four tokens at a time) | one container, --dp-size 4 of TP1 on cores 0-3; SGL_CORES=44-47; SGL_EXTRA='--tp-size 1 --dp-size 4 --max-total-tokens 65536 --max-running-requests 8 --stream-interval 4' | 350 | 222.1 / 0.044 | 1,417.9 / 0.050 | 3,038.6 / 0.163 | 5 |  |
| V23: V20 plus SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=4 | one container, --dp-size 4 of TP1 on cores 0-3; SGL_CORES=44-47; SGL_EXTRA='--tp-size 1 --dp-size 4 --max-total-tokens 65536 --max-running-requests 8' | 330 | 222.7 / 0.042 | 1,422.4 / 0.051 | 3,047.9 / 0.160 | 5 |  |
| V24: kernel, logical-core config 1, --dp-size 8 of TP1 on one chip | one container, LNC 1, --dp-size 8 of TP1 on the 8 cores of one chip; SGL_CORES=88-95; SGL_EXTRA='--tp-size 1 --dp-size 8 --max-total-tokens 65536 --max-running-requests 4' | - | - | - | - | - | not_ready |

## Guard answers

Five fixed prompts, chat route, temperature 0, 48 new tokens, on the first replica; "same" is the text equal to V0's for that prompt. A tensor-parallel layout may change a late token: it is recorded, it does not fail a variant.

| variant | status | same text per prompt | guard ok on every replica |
|---|---|---|---|
| V0 | ok | reference | yes |
| V22 | ok | same same same same same | yes |
| V23 | ok | same same same same same | yes |
| V24 | not_ready | - | - |

## Cache and warm-up records

Files in the shared compile-cache folder before and after the variant, and the engine's warm-up line of each server ("warmup done in <s>s (<n> shapes)"): a variant that adds no files started warm.

| variant | cache files before | after | warm-up lines |
|---|---|---|---|
| V0 | 512 | 512 | [neuron] warmup done in 345.8s (21 shapes); [neuron] warmup done in 345.8s (21 shapes); [neuron] warmup done in 345.8s (21 shapes); [neuron] warmup done in 345.8s (21 shapes) |
| V22 | 512 | 512 | [neuron] warmup done in 180.7s (12 shapes); [neuron] warmup done in 178.6s (12 shapes); [neuron] warmup done in 177.8s (12 shapes); [neuron] warmup done in 180.1s (12 shapes) |
| V23 | 512 | 512 | [neuron] warmup done in 178.7s (12 shapes); [neuron] warmup done in 178.4s (12 shapes); [neuron] warmup done in 178.0s (12 shapes); [neuron] warmup done in 179.2s (12 shapes) |
| V24 | 512 | 512 | - |

## Notes

- Each cell is one run of 64 requests (after 4 warm-up requests) on one server start; a replica row holds one such run per replica.
- Baseline repeat spread, prompt 1024 (|pass 1 - pass 2| / mean of the two, SGLang-Neuron released defaults, llmbench_q06_sgl_rep.csv): concurrency 1: tok/s 0.0 %, TTFT p50 0.4 %; concurrency 2: tok/s 0.0 %, TTFT p50 2.6 %; concurrency 4: tok/s 0.2 %, TTFT p50 2.3 %; concurrency 8: tok/s 0.1 %, TTFT p50 0.2 %; concurrency 16: tok/s 0.0 %, TTFT p50 2.9 %; concurrency 32: tok/s 0.2 %, TTFT p50 0.3 %.
- A mark names a cell with failed requests, a cell whose mean output tokens differ from the requested 256, a guard that failed, or a status other than ok. A row with a mark is shown as measured.
- Ready s is the time from the start command to the first answered request, with the shared compile cache in the state the earlier variants left it (a first start of a new shape includes its compile).
- Several rows do the same device work as V0 at some measured grid points, so their tok/s there is expected to equal V0's within the run-to-run spread, and what they can change is the ready time: V3 and V5 run the 1, 8 and 32 row decode graphs at c=1, 8 and 32 exactly as V0 does (only the ramp and the tail of a wave differ); V4 has one 32-row graph, so only c=32 runs the graph V0 runs (c=1 and c=8 pad to 32 rows); V10 runs the 2048-context decode graph under both context ladders, since every row is at most 1024 + 256 tokens; V9 at prompt 1024 fits one prefill tile under chunk 2048 and under chunk 4096, so only its prompt-3072 cells can show a change; V12 changes the host's streaming only. Read a difference at those points against the repeat spread above, and read ready s and the cache and warm-up records of variant_end (cache_files_before/after, warmup_lines) for the effect these rows can have.
- Rows are not compared with each other in this file and none is ranked.

