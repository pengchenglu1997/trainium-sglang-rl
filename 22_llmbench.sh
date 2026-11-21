#!/usr/bin/env bash
# Speed of a running server with the team's tool and grid (HolboxAI/llmbench at commit 08f4163;
# grid of the inference comparison rounds: concurrency 1 to 32, prompts of 128, 1,024 and 3,072
# tokens, 256 output tokens, 64 requests per cell after 4 warm-up requests; Slack, 2026-11-19).
#   22_llmbench.sh <tag> <base url> <model name of the API> <tokenizer dir or hub id> <results dir> [devices]
# LLMBENCH_DIR: where the tool is cloned (default $HOME/w/llmbench). It needs Python 3.10 to
# 3.13; uv fetches one. Nothing here starts or stops a server.
set -u
tag="${1:?tag}"; url="${2:?base url}"; model="${3:?model}"; tok="${4:?tokenizer}"; out="${5:?results dir}"
devices="${6:-1}"
dir="${LLMBENCH_DIR:-$HOME/w/llmbench}"
mkdir -p "$out"
if [ ! -d "$dir/.git" ]; then
  git clone https://github.com/HolboxAI/llmbench "$dir" || exit 1
fi
git -C "$dir" checkout -q 08f4163 || exit 1
export PATH="$HOME/.local/bin:$PATH"
# uv from the active environment's pip (a system Python may refuse a user install)
command -v uv >/dev/null 2>&1 || python -m pip install -q uv \
  || python3 -m pip install -q --user --break-system-packages uv || { echo "LLMBENCH_FAILED no uv"; exit 1; }
cfg="$out/llmbench_${tag}.yaml"
cat > "$cfg" <<EOF
run_name: ${tag}
seed: 42
output_dir: ${out}/llmbench

sweep:
  concurrency_levels: [1, 2, 4, 8, 16, 32]
  prompt_length_points: [128, 1024, 3072]
  output_tokens: 256
  requests_per_cell: 64
  warmup_requests_per_cell: 4
  vary_prompt_per_request: true

targets:
  - name: ${tag}
    base_url: ${url}
    model: ${model}
    tokenizer: ${tok}
    max_model_len: 4096
    api_path: /v1/completions
    stream: true
    num_gpus: ${devices}
    request_timeout_s: 1800
    extra_body:
      ignore_eos: true
      temperature: 0
EOF
cd "$dir" || exit 1
uv sync -q || exit 1
uv run llm-bench validate --config "$cfg" || { echo "LLMBENCH_FAILED validate"; exit 1; }
uv run llm-bench run --config "$cfg" > "$out/llmbench_${tag}.log" 2>&1 \
  && echo "LLMBENCH_DONE $tag" || { echo "LLMBENCH_FAILED run (see $out/llmbench_${tag}.log)"; exit 1; }
