#!/usr/bin/env bash
# R7 / step H1 of lora_kernel/PLAN_lora_merge_kernel_2026-11-23.md: the LoRA merge kernel on ONE NeuronCore, no server.
# NOT YET RUN ON HARDWARE. Written 2026-11-24 on a laptop; every UNVERIFIED mark below is a value nobody has seen work.#
#   bash 78_kernel_h1.sh <results dir>
#
# Copies nothing. Expects next to itself: 73_merge_kernel_bench.py, lora_merge.py (byte copies of lora_kernel/, md5 of
# both are written to the result file), 78_nki_version.py. Runs them with python3 inside the SGLang-Neuron release
# image (the bench imports sglang_neuron.backend.stack.wrap_nki from it), image name as srv_sglang_neuron.sh:21.
# No patch folder is mounted: the image as released (SGL_PATCH_DIR is not used here).
#
# Environment (all optional):
#   H1_CORE      NEURON_RT_VISIBLE_CORES of the container: one logical core, default 48 (must be free; not checked)
#   H1_LNC       NEURON_LOGICAL_NC_CONFIG and the bench's --lnc, default 2 (73_merge_kernel_bench.py:78 defaults to 2)
#   H1_MODELS    the bench's --models, default 8b = the four Qwen3-8B TP4 shard shapes qkv, o, gate_up, down
#                (73_merge_kernel_bench.py:35-38; the bench has no switch for a single shape). 8b,0.6b adds the 0.6B ones.
#   H1_WARM      the bench's --warm, default 5
#   H1_TIMEOUT   seconds before the bench container is killed, default 1500 (UNVERIFIED: a guess; compile times unknown)
#   SGL_IMAGE    default the v0.1.0 release image (as srv_sglang_neuron.sh)
#   H1_CACHE     compile cache of THIS script, default ${DATA_ROOT:-$HOME/w}/sgl_cache_v0.1.0_kernel_h1. Image only, no patch.
#
# Cache folder: ONLY $H1_CACHE (default ${DATA_ROOT:-$HOME/w}/sgl_cache_v0.1.0_kernel_h1), mounted at /cache. The kit patch
# (patches/sglang_neuron_v0.1.0_inplace) and the PR 20 patch must never share a cache folder with each other or with
# this one: a launch refuses a program cache built from other NKI kernel sources, and this script compiles lora_merge.
# A stamp file .patch_in_use in the cache folder records what it belongs to; a different stamp is refused.
#
# Result files in <results dir>:
#   78_kernel_h1.jsonl          this script's own record, JSON lines, last line {"done": true}
#   73_merge_kernel_bench.jsonl the bench's rows (one JSON line per measurement; written by the bench; an old one is
#                               renamed to .prev.<unix time>); the bench ends it with {"kind": "done"}
# Exit code of this script: 0 H1 PASS; 3 stop rule (the bench exited 3: the kernel cannot be called outside a serving
# graph); 1 any other failure; 2 setup problem (nothing was run on the device).
# Last line printed, always one of: "H1 PASS ...", "H1 STOP ...", "H1 FAIL ...".
# (Status note added 2026-11-25, no code changed: "NOT YET RUN ON HARDWARE" above was true when the file was written.
# It ran on trn2 on 2026-11-24; result data/2026-11-24_trn2/78_kernel_h1.jsonl. The UNVERIFIED marks were not revisited.)
set -u
case "${1:-}" in -h|--help) sed -n '2,33p' "$0"; exit 0 ;; esac
KIT="$(cd "$(dirname "$0")" && pwd)"
R="${1:?usage: 78_kernel_h1.sh <results dir>}"
mkdir -p "$R" || { echo "H1 FAIL setup: cannot create $R"; exit 2; }
R="$(cd "$R" && pwd)"   # absolute before anything else (nothing here changes the directory of the script itself)
H1_CORE="${H1_CORE:-48}"; H1_LNC="${H1_LNC:-2}"; H1_MODELS="${H1_MODELS:-8b}"; H1_WARM="${H1_WARM:-5}"
H1_TIMEOUT="${H1_TIMEOUT:-1500}"
H1_CACHE="${H1_CACHE:-${DATA_ROOT:-$HOME/w}/sgl_cache_v0.1.0_kernel_h1}"
IMAGE="${SGL_IMAGE:-public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04}"
OUT="$R/78_kernel_h1.jsonl"
BENCH_OUT="$R/73_merge_kernel_bench.jsonl"
NAME="kernel_h1_bench"
say() { echo "== H1 $(date -u +%H:%M:%S) $*"; }
esc() { local s="${1:-}"; s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; printf '%s' "$s" | tr -d '\n\r\t'; }
rotate() { [ -e "$1" ] && mv "$1" "$1.prev.$(date +%s)"; return 0; }
rotate "$OUT"
emit() { printf '%s\n' "$1" >> "$OUT"; }
finish() {  # exit code, last-line text
  emit '{"done": true}'
  echo "$2"
  exit "$1"
}
setup_fail() { emit "{\"event\": \"setup_fail\", \"why\": \"$(esc "$1")\"}"; finish 2 "H1 FAIL setup: $1"; }

emit "{\"event\": \"start\", \"unix\": $(date +%s), \"core\": \"$(esc "$H1_CORE")\", \"lnc\": \"$(esc "$H1_LNC")\", \"models\": \"$(esc "$H1_MODELS")\", \"image\": \"$(esc "$IMAGE")\", \"cache\": \"$(esc "$H1_CACHE")\"}"
for f in 73_merge_kernel_bench.py lora_merge.py 78_nki_version.py; do
  [ -f "$KIT/$f" ] || setup_fail "$f is not next to the script ($KIT)"
done
emit "{\"event\": \"files\", \"bench_md5\": \"$(md5sum "$KIT/73_merge_kernel_bench.py" | cut -d' ' -f1)\", \"lora_merge_md5\": \"$(md5sum "$KIT/lora_merge.py" | cut -d' ' -f1)\"}"
command -v docker > /dev/null 2>&1 || setup_fail "docker not found"
devices=()
for d in /dev/neuron*; do [ -e "$d" ] && devices+=(--device "$d"); done
[ "${#devices[@]}" -gt 0 ] || setup_fail "no /dev/neuron* devices (this script runs on the trn2 host)"
docker image inspect "$IMAGE" > /dev/null 2>&1 || setup_fail "image $IMAGE is not on this host (docker pull it first; R0 setup does)"
mkdir -p "$H1_CACHE" || setup_fail "cannot create the cache folder $H1_CACHE"
STAMP="$H1_CACHE/.patch_in_use"
if [ -f "$STAMP" ]; then
  [ "$(cat "$STAMP")" = "image-only (78_kernel_h1.sh)" ] || setup_fail "cache folder $H1_CACHE belongs to '$(cat "$STAMP")', not to this script; give H1_CACHE another folder"
else
  echo "image-only (78_kernel_h1.sh)" > "$STAMP"
fi
say "image $IMAGE, core $H1_CORE, lnc $H1_LNC, models $H1_MODELS, cache $H1_CACHE (image only, no patch)"

run_image() {  # container name, then the arguments of python3
  local name="$1"; shift
  docker run --rm --init --ipc=host --cap-add SYS_NICE "${devices[@]}" --name "$name" \
    -v "$KIT":/kit:ro -v "$R":/results -v "$H1_CACHE":/cache \
    -e NEURON_RT_VISIBLE_CORES="$H1_CORE" -e NEURON_LOGICAL_NC_CONFIG="$H1_LNC" \
    -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo -e NKI_COMPILE_CACHE_URL=/cache/nki \
    -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -w /tmp \
    --entrypoint python3 "$IMAGE" "$@"
}

# 1. what the image has
VOUT="$R/78_nki_version.out"
run_image kernel_h1_version /kit/78_nki_version.py > "$VOUT" 2>&1
vrc=$?
NKIV="$(grep -h '^NKI_VERSION=' "$VOUT" | tail -n 1 | cut -d= -f2-)"
[ -n "$NKIV" ] || NKIV="not printed (exit $vrc)"
emit "{\"event\": \"nki_version\", \"nki\": \"$(esc "$NKIV")\", \"helper_exit\": $vrc, \"output_file\": \"78_nki_version.out\"}"
say "nki version found in the image: $NKIV (helper exit $vrc; full line in $VOUT)"
[ "$vrc" = 0 ] || finish 1 "H1 FAIL: the image cannot import nki, nki.isa and nki.language (helper exit $vrc, see $VOUT); nki version: $NKIV"

# 2. the bench
rotate "$BENCH_OUT"
say "bench start (cap ${H1_TIMEOUT}s); rows go to $BENCH_OUT"
timeout -k 30 "$H1_TIMEOUT" docker run --rm --init --ipc=host --cap-add SYS_NICE "${devices[@]}" --name "$NAME" \
  -v "$KIT":/kit:ro -v "$R":/results -v "$H1_CACHE":/cache \
  -e NEURON_RT_VISIBLE_CORES="$H1_CORE" -e NEURON_LOGICAL_NC_CONFIG="$H1_LNC" \
  -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo -e NKI_COMPILE_CACHE_URL=/cache/nki \
  -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -w /tmp \
  --entrypoint python3 "$IMAGE" /kit/73_merge_kernel_bench.py --out /results/73_merge_kernel_bench.jsonl \
  --models "$H1_MODELS" --lnc "$H1_LNC" --warm "$H1_WARM" > "$R/78_kernel_h1_bench.out" 2>&1
rc=$?
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then docker rm -f "$NAME" > /dev/null 2>&1; fi
sudo chown -R "$(id -u):$(id -g)" "$R" 2> /dev/null || true

# 3. read what the bench wrote (counts only; the numbers are read from the file by whoever reports them)
n_rows=0; mode="none"; bad_kernel=0; int_rows=0; int_bad=0; bench_done=0
if [ -f "$BENCH_OUT" ]; then
  n_rows=$(wc -l < "$BENCH_OUT")
  mode=$(grep -h '"kind": "invoke_mode"' "$BENCH_OUT" | tail -n 1 | grep -o '"mode": "[a-z]*"' | cut -d'"' -f4)
  [ -n "$mode" ] || mode="none"
  bad_kernel=$(grep -h '"kind": "kernel"\|"kind": "kernel_integer"\|"kind": "torch_device"' "$BENCH_OUT" | grep -c '"ok": false')
  int_rows=$(grep -c '"kind": "kernel_integer"' "$BENCH_OUT")
  # integer-valued factors must equal the reference bit for bit: a row without "differ_vs_reference": 0 is a failure
  int_bad=$(grep -h '"kind": "kernel_integer"' "$BENCH_OUT" | grep -vc '"differ_vs_reference": 0,')
  grep -q '"kind": "done"' "$BENCH_OUT" && bench_done=1
fi
emit "{\"event\": \"bench_exit\", \"code\": $rc, \"rows\": $n_rows, \"invoke_mode\": \"$(esc "$mode")\", \"failed_rows\": $bad_kernel, \"integer_rows\": $int_rows, \"integer_not_bit_exact\": $int_bad, \"bench_done_row\": $bench_done, \"bench_stdout\": \"78_kernel_h1_bench.out\"}"
say "bench exit $rc; rows $n_rows; invoke mode $mode; failed rows $bad_kernel; integer rows $int_rows (not bit-exact $int_bad); done row $bench_done"

if [ "$rc" = 3 ]; then
  emit '{"event": "verdict", "result": "stop", "meaning": "kernel cannot be called outside a serving graph"}'
  finish 3 "H1 STOP exit 3: kernel cannot be called outside a serving graph (stop rule of the plan: stop the kernel part, report the torch-device and host numbers only); bench exit code 3; nki $NKIV"
fi
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
  emit '{"event": "verdict", "result": "fail", "meaning": "bench killed at the time cap"}'
  finish 1 "H1 FAIL: bench killed after ${H1_TIMEOUT}s (exit $rc); rows so far in $BENCH_OUT; nki $NKIV"
fi
if [ "$rc" != 0 ]; then
  emit '{"event": "verdict", "result": "fail", "meaning": "bench crashed"}'
  finish 1 "H1 FAIL: bench exit code $rc (last lines of 78_kernel_h1_bench.out: $(tail -n 2 "$R/78_kernel_h1_bench.out" | tr '\n' ' ' | cut -c1-300)); nki $NKIV"
fi
if [ "$bench_done" != 1 ] || [ "$int_rows" = 0 ] || [ "$bad_kernel" != 0 ] || [ "$int_bad" != 0 ]; then
  emit '{"event": "verdict", "result": "fail", "meaning": "bench ran to its end but rows are missing or wrong"}'
  finish 1 "H1 FAIL: bench exit 0 but done_row=$bench_done integer_rows=$int_rows failed_rows=$bad_kernel integer_not_bit_exact=$int_bad (read $BENCH_OUT); nki $NKIV"
fi
emit '{"event": "verdict", "result": "pass", "meaning": "bench ran to its end; integer factors bit-exact; timings and read-back counts are in the bench file"}'
finish 0 "H1 PASS exit 0: invoke mode $mode, $n_rows rows in $BENCH_OUT, integer factors bit-exact in $int_rows shapes; nki $NKIV"
