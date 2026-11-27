#!/usr/bin/env bash
# Like 22_llmbench.sh, with the grid taken from the environment, and the summary CSV copied to a fixed name.
#   22b_llmbench_grid.sh <tag> <base url> <model name of the API> <tokenizer dir or hub id> <results dir> [devices]
# Tool: HolboxAI/llmbench at commit 08f416347db19bb45ff2489b8a69eacef62b6875 (the full hash of the 08f4163 of
# 22_llmbench.sh; that repository's script, HarryZ66/trainium-inference-parity payload/run_inference.sh line 46, pins it too).
# Environment (defaults = that repository's round-3 grid, run_inference.sh lines 123 to 126):
#   LB_CONC "1,2,4,8,16,32"   LB_PROMPTS "128,1024,3072"   LB_OUT 256   LB_REQ 64   LB_WARM 4   LB_MAXLEN 4096
#   LB_TEMP 0 (request temperature)   LB_EXTRA_BODY "" (extra request fields, "name=value" items separated by spaces; added 2026-11-26)
#   LLMBENCH_DIR  where the tool is cloned (default $HOME/w/llmbench); needs Python 3.10 to 3.13, uv fetches one
# Files: <results>/llmbench_<tag>.yaml (config), .log (tool output), llmbench_<tag>/<run id>/ (raw), and on
# success <results>/llmbench_<tag>.csv = that run's summary_all.csv (a copy). Nothing here starts or stops a server.
# Last line: "LLMBENCH_DONE <tag> csv=<path> rows=<n>" or "LLMBENCH_FAILED <reason>" (exit 1).
#
# Request body: /v1/completions, stream with usage, max_tokens = LB_OUT, plus extra_body below.
#   ignore_eos true: as that repository's config (run_inference.sh line 130), so every request produces LB_OUT tokens.
#   temperature 0: NOT in that repository's config, which has ignore_eos only. Why we pin it on both engines: a request
#   without temperature takes the server's own default. SGLang applies the model's generation_config as the request
#   default (rollout_client.py lines 49 to 52, seen 2026-11-20: Qwen3 top_k 20, top_p 0.95), and that repository's script
#   itself greps the vllm-neuron log for the sampling defaults it took from generation_config (run_inference.sh lines
#   93 and 94). Sampling defaults differ per engine and decide how much sampler work a step does; greedy on both
#   makes the two arms do the same work. 22_llmbench.sh sent it already (the 2,469 tok/s of 2026-11-20 is greedy).
# Pre-set llmbench facts, read at that commit: summary file name "summary_all.csv" in the run folder
#   src/llmbench/io/paths.py lines 35-36; run folder <slug of run_name>_<UTC time> under output_dir, paths.py line 9-11;
#   "--no-plots" exists, cli.py line 90; extra_body is merged into the body last, openai_compat.py line 62.
set -u
tag="${1:?tag}"; url="${2:?base url}"; model="${3:?model}"; tok="${4:?tokenizer}"; out="${5:?results dir}"
devices="${6:-1}"
PIN=08f416347db19bb45ff2489b8a69eacef62b6875
conc="${LB_CONC:-1,2,4,8,16,32}"; prompts="${LB_PROMPTS:-128,1024,3072}"
lb_out="${LB_OUT:-256}"; lb_req="${LB_REQ:-64}"; lb_warm="${LB_WARM:-4}"; lb_maxlen="${LB_MAXLEN:-4096}"
fail() { echo "LLMBENCH_FAILED $*"; exit 1; }
case "$tag" in *[!A-Za-z0-9_.-]*|"") fail "tag must be letters, digits, _ . - only: $tag" ;; esac
list_re='^[0-9]+(,[0-9]+)*$'; int_re='^[0-9]+$'
[[ $conc =~ $list_re ]] || fail "LB_CONC must be a comma list of integers: $conc"
[[ $prompts =~ $list_re ]] || fail "LB_PROMPTS must be a comma list of integers: $prompts"
for v in "$lb_out" "$lb_req" "$lb_warm" "$lb_maxlen" "$devices"; do
  [[ $v =~ $int_re ]] || fail "LB_OUT, LB_REQ, LB_WARM, LB_MAXLEN and devices must be integers: $v"
done
dir="${LLMBENCH_DIR:-$HOME/w/llmbench}"
mkdir -p "$out" || fail "cannot create $out"
if [ ! -d "$dir/.git" ]; then
  git clone https://github.com/HolboxAI/llmbench "$dir" || fail "git clone"
fi
git -C "$dir" checkout -q "$PIN" 2>/dev/null || { git -C "$dir" fetch -q origin && git -C "$dir" checkout -q "$PIN"; } \
  || fail "checkout $PIN"
[ "$(git -C "$dir" rev-parse HEAD)" = "$PIN" ] || fail "llmbench is not at $PIN"
export PATH="$HOME/.local/bin:$PATH"
# uv from the active environment's pip (a system Python may refuse a user install)
command -v uv >/dev/null 2>&1 || python -m pip install -q uv \
  || python3 -m pip install -q --user --break-system-packages uv || fail "no uv"
cfg="$out/llmbench_${tag}.yaml"
cat > "$cfg" <<EOF
run_name: ${tag}
seed: 42
output_dir: ${out}/llmbench_${tag}

sweep:
  concurrency_levels: [${conc}]
  prompt_length_points: [${prompts}]
  output_tokens: ${lb_out}
  requests_per_cell: ${lb_req}
  warmup_requests_per_cell: ${lb_warm}
  vary_prompt_per_request: true

targets:
  - name: ${tag}
    base_url: ${url}
    model: ${model}
    tokenizer: ${tok}
    max_model_len: ${lb_maxlen}
    api_path: /v1/completions
    stream: true
    num_gpus: ${devices}
    request_timeout_s: 1800
    extra_body:
      ignore_eos: true
      temperature: ${LB_TEMP:-0}
EOF
# LB_EXTRA_BODY "top_k=50 top_p=0.95": more request fields, one "name=value" per item (added 2026-11-26; unset = as before)
for kv in ${LB_EXTRA_BODY:-}; do echo "      ${kv%%=*}: ${kv#*=}" >> "$cfg"; done
cd "$dir" || fail "cd $dir"
uv sync -q || fail "uv sync"
uv run llm-bench validate --config "$cfg" || fail "validate"
uv run llm-bench run --config "$cfg" --no-plots > "$out/llmbench_${tag}.log" 2>&1 \
  || fail "run (see $out/llmbench_${tag}.log)"
n_conc=$(echo "$conc" | tr ',' '\n' | sort -u | wc -l); n_pt=$(echo "$prompts" | tr ',' '\n' | sort -u | wc -l)
want=$(( n_conc * n_pt ))
# newest run folder of this tag (a tag run twice leaves two)
src_csv=$(ls -t "$out/llmbench_${tag}"/*/summary_all.csv 2>/dev/null | head -1)
[ -n "$src_csv" ] || fail "no summary_all.csv under $out/llmbench_${tag} (see $out/llmbench_${tag}.log)"
rows=$(( $(wc -l < "$src_csv") - 1 ))
if [ "$rows" -ne "$want" ]; then
  cp "$src_csv" "$out/llmbench_${tag}.partial.csv"
  fail "summary has $rows rows, grid has $want cells; copied to $out/llmbench_${tag}.partial.csv"
fi
cp "$src_csv" "$out/llmbench_${tag}.csv" || fail "copy of the summary"
echo "LLMBENCH_DONE $tag csv=$out/llmbench_${tag}.csv rows=$rows"
