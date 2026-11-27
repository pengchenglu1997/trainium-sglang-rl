#!/usr/bin/env bash
# Session 4 setup (2026-11-26): image, Qwen3-0.6B, Qwen3-8B, GSM8K, llmbench at the pinned commit.
# Same commands as the prep step of the session before it, without the vllm image, the 30B fetch and the _alt copies.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
VENV=/opt/aws_neuronx_venv_torch_neuronx_2_14_0
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export DATA_ROOT="${DATA_ROOT:-$HOME/w}"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3
M="$DATA_ROOT/models"; R="$DATA_ROOT/results_s4"
IMAGE="public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04"
LB_PIN=08f416347db19bb45ff2489b8a69eacef62b6875
mkdir -p "$R/logs" "$R/server_logs" "$M"
say() { echo "== S $(date -u +%H:%M:%S) $*"; }
fetch() {
  if ls "$M/$2"/*.safetensors > /dev/null 2>&1 && [ -f "$M/$2/.fetch_ok" ]; then say "model $2 present"; return 0; fi
  docker run --rm -v "$M":/models "$IMAGE" hf download "$1" --local-dir "/models/$2" > "$R/logs/hf_$2.out" 2>&1 \
    || { say "model $2 FAILED: $(tail -n 2 "$R/logs/hf_$2.out" | cut -c1-200)"; return 1; }
  sudo -n chown -R "$(id -u):$(id -g)" "$M/$2" 2>/dev/null || true
  touch "$M/$2/.fetch_ok"
  say "model $2 ok $(du -sh "$M/$2" | cut -f1)"
}
bad=0
say "host: $(nproc) cpus; df $(df -h "$DATA_ROOT" | tail -n 1 | tr -s ' ')"
say "neuron-ls: $(neuron-ls 2>&1 | head -n 4 | tr '\n' '|' | cut -c1-300)"
docker pull "$IMAGE" > "$R/logs/docker_pull_sgl.out" 2>&1 || { say "prep FAILED docker pull"; echo "S_DONE prep status=fail"; exit 1; }
say "image ok"
fetch Qwen/Qwen3-0.6B Qwen3-0.6B & p1=$!
fetch Qwen/Qwen3-8B Qwen3-8B & p2=$!
wait "$p1" || bad=$((bad+1))
wait "$p2" || bad=$((bad+1))
mkdir -p "$M/gsm8k"
for s in train test; do
  [ -s "$M/gsm8k/$s.jsonl" ] || curl -sSfL -o "$M/gsm8k/$s.jsonl" \
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/$s.jsonl" \
    || { say "gsm8k $s FAILED"; bad=$((bad+1)); }
done
say "gsm8k lines: $(wc -l "$M"/gsm8k/*.jsonl | tail -n 1)"
lb="$DATA_ROOT/llmbench"
[ -d "$lb/.git" ] || git clone "${LLMBENCH_URL:?set LLMBENCH_URL to the llmbench repository named in 22b_llmbench_grid.sh}" "$lb" > "$R/logs/llmbench_clone.out" 2>&1
git -C "$lb" checkout -q "$LB_PIN" 2>> "$R/logs/llmbench_clone.out"
[ "$(git -C "$lb" rev-parse HEAD 2>/dev/null)" = "$LB_PIN" ] && say "llmbench at $LB_PIN" || { say "llmbench NOT at pin"; bad=$((bad+1)); }
export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null 2>&1 || "$PY" -m pip install -q uv > "$R/logs/uv_install.out" 2>&1 \
  || python3 -m pip install -q --user --break-system-packages uv >> "$R/logs/uv_install.out" 2>&1
if (cd "$lb" && uv sync -q && uv run llm-bench --help) > "$R/logs/llmbench_sync.out" 2>&1; then say "llmbench uv sync ok"
else say "llmbench uv sync FAILED: $(tail -n 2 "$R/logs/llmbench_sync.out" | tr '\n' ' ' | cut -c1-250)"; bad=$((bad+1)); fi
say "prep problems: $bad"
echo "S_DONE prep status=$([ "$bad" = 0 ] && echo ok || echo fail)"
