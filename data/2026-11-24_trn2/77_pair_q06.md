# Baseline pair Qwen3-0.6B (q06)

- vllm-neuron: grid 18 cells, repeat 6 cells, pre-check ok (files llmbench_q06_vllm.csv, llmbench_q06_vllm_rep.csv, 76_serve_q06_vllm.jsonl)
- sglang-neuron: grid 18 cells, repeat 6 cells, pre-check ok (files llmbench_q06_sgl.csv, llmbench_q06_sgl_rep.csv, 76_serve_q06_sgl.jsonl)

Tokens per second: total output tokens over the wall time of the measured batch (llmbench output_token_throughput_tps). TTFT p50: median seconds from send to first streamed chunk (ttft_p50_s). Grid pass values are shown; a flag names a cell that failed requests, a cell whose output length is off, or an engine whose pre-check did not pass (PRECHECK FAILED is set on every row of that engine).

Output length check: a cell is marked "output length off" when the mean output tokens per successful request (output_token_throughput_tps / request_throughput_rps; the CSV has no per-cell mean) differs from output_tokens_requested (LB_OUT) by more than 0.01 token. Checked for: vllm-neuron, sglang-neuron.

## Prompt 128 tokens

| concurrency | vllm-neuron tok/s | vllm-neuron TTFT p50 (s) | sglang-neuron tok/s | sglang-neuron TTFT p50 (s) | flags |
|---|---|---|---|---|---|
| 1 | 211.4 | 0.013 | 218.6 | 0.020 |  |
| 2 | 445.2 | 0.020 | 400.4 | 0.026 |  |
| 4 | 782.0 | 0.031 | 727.9 | 0.035 |  |
| 8 | 1,258.0 | 0.048 | 1,241.2 | 0.054 |  |
| 16 | 1,814.8 | 0.091 | 1,865.4 | 0.100 |  |
| 32 | 2,506.9 | 0.196 | 2,999.4 | 0.186 |  |

Spread: none. Every cell of this table was run once; no spread is known for any of them.

## Prompt 1024 tokens

| concurrency | vllm-neuron tok/s | vllm-neuron TTFT p50 (s) | sglang-neuron tok/s | sglang-neuron TTFT p50 (s) | flags |
|---|---|---|---|---|---|
| 1 | 209.9 | 0.021 | 211.1 | 0.025 |  |
| 2 | 439.9 | 0.032 | 375.4 | 0.034 |  |
| 4 | 766.7 | 0.047 | 679.0 | 0.047 |  |
| 8 | 1,230.3 | 0.076 | 1,168.0 | 0.075 |  |
| 16 | 1,740.8 | 0.142 | 1,793.8 | 0.138 |  |
| 32 | 2,375.3 | 0.276 | 2,460.6 | 0.260 |  |

Spread: only concurrency 1, 2, 4, 8, 16, 32 of this prompt length was run twice (repeat table below); every other cell of this table was run once and carries no spread.

### Repeat pass, prompt 1024 tokens (relative spread = |pass 1 - pass 2| / mean of the two)

| concurrency | engine | tok/s pass 1 | tok/s pass 2 | spread | TTFT p50 pass 1 (s) | TTFT p50 pass 2 (s) | spread | flags |
|---|---|---|---|---|---|---|---|---|
| 1 | vllm-neuron | 209.9 | 209.9 | 0.0 % | 0.021 | 0.021 | 0.4 % |  |
| 1 | sglang-neuron | 211.1 | 211.0 | 0.0 % | 0.025 | 0.025 | 0.4 % |  |
| 2 | vllm-neuron | 439.9 | 440.0 | 0.0 % | 0.032 | 0.032 | 0.0 % |  |
| 2 | sglang-neuron | 375.4 | 375.4 | 0.0 % | 0.034 | 0.033 | 2.6 % |  |
| 4 | vllm-neuron | 766.7 | 766.9 | 0.0 % | 0.047 | 0.047 | 0.9 % |  |
| 4 | sglang-neuron | 679.0 | 680.3 | 0.2 % | 0.047 | 0.046 | 2.3 % |  |
| 8 | vllm-neuron | 1,230.3 | 1,229.6 | 0.1 % | 0.076 | 0.076 | 0.6 % |  |
| 8 | sglang-neuron | 1,168.0 | 1,169.5 | 0.1 % | 0.075 | 0.074 | 0.2 % |  |
| 16 | vllm-neuron | 1,740.8 | 1,740.2 | 0.0 % | 0.142 | 0.142 | 0.4 % |  |
| 16 | sglang-neuron | 1,793.8 | 1,793.7 | 0.0 % | 0.138 | 0.134 | 2.9 % |  |
| 32 | vllm-neuron | 2,375.3 | 2,372.2 | 0.1 % | 0.276 | 0.278 | 0.8 % |  |
| 32 | sglang-neuron | 2,460.6 | 2,456.7 | 0.2 % | 0.260 | 0.259 | 0.3 % |  |

Only the rows of this table carry a spread (two passes on one server start); the other cells of the grid were run once.

## Prompt 3072 tokens

| concurrency | vllm-neuron tok/s | vllm-neuron TTFT p50 (s) | sglang-neuron tok/s | sglang-neuron TTFT p50 (s) | flags |
|---|---|---|---|---|---|
| 1 | 204.7 | 0.051 | 203.6 | 0.055 |  |
| 2 | 419.8 | 0.074 | 363.3 | 0.053 |  |
| 4 | 706.7 | 0.116 | 630.8 | 0.065 |  |
| 8 | 1,084.5 | 0.200 | 1,050.9 | 0.102 |  |
| 16 | 1,472.5 | 0.377 | 1,505.8 | 0.172 |  |
| 32 | 1,909.8 | 0.719 | 1,974.8 | 0.357 |  |

Spread: none. Every cell of this table was run once; no spread is known for any of them.

