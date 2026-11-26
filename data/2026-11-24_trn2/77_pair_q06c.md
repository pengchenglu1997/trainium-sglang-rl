# Baseline pair Qwen3-0.6B (q06c)

- sglang-neuron: grid 3 cells, repeat 3 cells, pre-check ok (files llmbench_q06c_sgl.csv, llmbench_q06c_sgl_rep.csv, 76_serve_q06c_sgl.jsonl)

Tokens per second: total output tokens over the wall time of the measured batch (llmbench output_token_throughput_tps). TTFT p50: median seconds from send to first streamed chunk (ttft_p50_s). Grid pass values are shown; a flag names a cell that failed requests, a cell whose output length is off, or an engine whose pre-check did not pass (PRECHECK FAILED is set on every row of that engine).

Output length check: a cell is marked "output length off" when the mean output tokens per successful request (output_token_throughput_tps / request_throughput_rps; the CSV has no per-cell mean) differs from output_tokens_requested (LB_OUT) by more than 0.01 token. Checked for: sglang-neuron.

## Prompt 1024 tokens

| concurrency | sglang-neuron tok/s | sglang-neuron TTFT p50 (s) | flags |
|---|---|---|---|
| 1 | 210.9 | 0.025 |  |
| 8 | 1,159.4 | 0.075 |  |
| 32 | 2,451.0 | 0.270 |  |

Spread: only concurrency 1, 8, 32 of this prompt length was run twice (repeat table below); every other cell of this table was run once and carries no spread.

### Repeat pass, prompt 1024 tokens (relative spread = |pass 1 - pass 2| / mean of the two)

| concurrency | engine | tok/s pass 1 | tok/s pass 2 | spread | TTFT p50 pass 1 (s) | TTFT p50 pass 2 (s) | spread | flags |
|---|---|---|---|---|---|---|---|---|
| 1 | sglang-neuron | 210.9 | 210.9 | 0.0 % | 0.025 | 0.026 | 3.8 % |  |
| 8 | sglang-neuron | 1,159.4 | 1,168.4 | 0.8 % | 0.075 | 0.075 | 0.1 % |  |
| 32 | sglang-neuron | 2,451.0 | 2,465.2 | 0.6 % | 0.270 | 0.256 | 5.3 % |  |

Only the rows of this table carry a spread (two passes on one server start); the other cells of the grid were run once.

