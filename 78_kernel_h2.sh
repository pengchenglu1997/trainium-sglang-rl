#!/usr/bin/env bash
# R7 / step H2 of lora_kernel/PLAN_lora_merge_kernel_2026-11-23.md: adapter updates on a Qwen3-0.6B TP4 server through the
# NKI kernel (RLKIT_DEVICE_WRITE=kernel), with every read-back check on. NOT YET RUN ON HARDWARE (written 2026-11-24 on
# a laptop). Run it after 78_kernel_h1.sh passed (H1 stop rule: exit 3 means stop the kernel part).#
#   bash 78_kernel_h2.sh <results dir>
#
# Same as run_night_1121.sh adapter8 (50_adapter_server_test.py, SMALL flags) but for Qwen3-0.6B, with the kit's own
# patch folder patches/sglang_neuron_v0.1.0_inplace and the kernel switches. The test makes the adapters and the
# merged1 checkpoint itself (50_adapter_server_test.py:40-88), starts the server on the base, updates to adapter1,
# adapter2, adapter1, adapter0 (FOUR updates, :147), then restarts on merged1 (:158).
#
# Environment (all optional):
#   DATA_ROOT    default $HOME/w            MODELS_ROOT default $DATA_ROOT/models (Qwen3-0.6B must be in it: R0 setup)
#   H2_CORES     NEURON_RT_VISIBLE_CORES of the server, default 0-3 (four logical cores; must be free; not checked)
#   H2_PORT      host port of the server, default 30013 (the test's default, 50_adapter_server_test.py:126)
#   H2_TIMEOUT   seconds before the test is killed, default 2400 (UNVERIFIED: a guess; 0.6B started in 109-350 s on
#                2026-11-22 and the start here compiles the kernel path for the first time)
#   H2_CACHE     compile cache folder, default $DATA_ROOT/sgl_cache_v0.1.0_kernel_h2
#   PY           python of the host venv, default /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python
#
# CACHE FOLDER: ONLY $H2_CACHE = $DATA_ROOT/sgl_cache_v0.1.0_kernel_h2, mounted at /cache of the server container. It
# belongs to the KIT patch patches/sglang_neuron_v0.1.0_inplace (with the kernel files). The kit patch must NEVER share a
# cache folder with the PR 20 patch (or any other patch, or the stock image, or the night-1121 cache): the launch
# refuses a program cache built from other NKI kernel sources (container docs: docker/README.md "Keep the compile
# caches"), and patched plugin files change what is built. This script refuses a cache folder whose name does not end
# in _kernel_h2, and one whose stamp file .patch_in_use names another patch folder.
#
# What it sets for the server (SGL_ENV, read by srv_sglang_neuron.sh:48; names and values from the patch):
#   RLKIT_DEVICE_WRITE=kernel   host.py:358 reads it, host.py:367 selects the kernel path ("copy" is the default)
#   RLKIT_KERNEL_CHECK=all      rlkit_kernel_path.py:250 reads it; :334 reads the live parameter back after EVERY write
#                               (default "first": once per call form / shard shape / K / dtype)
#   RLKIT_KERNEL_IMPL=nki       rlkit_kernel_path.py:222; "auto" would also pick nki on a non-cpu device (:224); pinned
#                               so a silent torch fallback cannot pass for the kernel
#   RLKIT_KERNEL_FACTORS=fp32   rlkit_kernel_path.py:247 (the default; "split" is the bf16 hi/lo alternative)
# Left at their defaults: RLKIT_ADAPTER_ON_DEVICE (0; the kernel branch sets want_device False, host.py:368),
# RLKIT_LAZY_DELTA (1; used only for parameters the kernel refuses), RLKIT_ADAPTER_DIAG (0), RLKIT_KEEP_STAGED (0),
# RLKIT_DECODE_LADDER (1, as in run_night_1121.sh adapter8).
#
# Result files in <results dir>:
#   50_adapter_h2_06b.jsonl   the adapter test's rows (written by 50_adapter_server_test.py; an old one is renamed)
#   server_logs/server_srv_h2_06b_001.log (and _002.log after the restart): the server's docker logs, saved at stop
#   78_kernel_h2.jsonl        this script's record, JSON lines, last line {"done": true}; holds the verbatim JSON line of
#                             79_kernel_ran_check.py (kernel ran?) and of its --adapter-result mode (the test's checks)
# Exit code: 0 H2 PASS, 1 H2 FAIL, 2 setup problem (nothing was started).
# Last line printed, always: "H2 PASS ..." or "H2 FAIL ...". PASS needs ALL of: the test exited 0, every check of the
# plan's H2 row holds (79 --adapter-result), and the kernel-ran check passes (79 on the server logs, --updates 4).
# (Status note added 2026-11-25, no code changed: "NOT YET RUN ON HARDWARE" above was true when the file was written.
# It ran on trn2 on 2026-11-24; results data/2026-11-24_trn2/78_kernel_h2.jsonl and 50_adapter_h2_06b.jsonl.)
set -u
case "${1:-}" in -h|--help) sed -n '2,47p' "$0"; exit 0 ;; esac
R="${1:?usage: 78_kernel_h2.sh <results dir>}"
# the results dir is made absolute here, BEFORE the cd to the kit folder (a relative one means the caller's folder)
mkdir -p "$R" || { echo "H2 FAIL setup: cannot create $R"; exit 2; }
R="$(cd "$R" && pwd)"
cd "$(dirname "$0")" || { echo "H2 FAIL setup: cannot enter the kit folder"; exit 2; }
KIT="$PWD"
: "${DATA_ROOT:=$HOME/w}"
: "${PY:=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python}"
: "${MODELS_ROOT:=$DATA_ROOT/models}"
H2_CORES="${H2_CORES:-0-3}"; H2_PORT="${H2_PORT:-30013}"; H2_TIMEOUT="${H2_TIMEOUT:-2400}"
M="$MODELS_ROOT"
TAG=h2_06b
NAME="srv_$TAG"
SGL_CACHE="${H2_CACHE:-$DATA_ROOT/sgl_cache_v0.1.0_kernel_h2}"
SGL_PATCH_DIR="$KIT/patches/sglang_neuron_v0.1.0_inplace"
IMAGE="${SGL_IMAGE:-public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04}"
SMALL="--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768 --cuda-graph-bs-decode 32"
KSW="RLKIT_DEVICE_WRITE=kernel RLKIT_KERNEL_CHECK=all RLKIT_KERNEL_IMPL=nki RLKIT_KERNEL_FACTORS=fp32"
OUT="$R/78_kernel_h2.jsonl"
RESULT="$R/50_adapter_$TAG.jsonl"
LOGDIR="$R/server_logs"
say() { echo "== H2 $(date -u +%H:%M:%S) $*"; }
esc() { local s="${1:-}"; s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; printf '%s' "$s" | tr -d '\n\r\t'; }
rotate() { [ -e "$1" ] && mv "$1" "$1.prev.$(date +%s)"; return 0; }
rotate "$OUT"
emit() { printf '%s\n' "$1" >> "$OUT"; }
finish() {  # exit code, last-line text
  emit '{"done": true}'
  echo "$2"
  exit "$1"
}
setup_fail() { emit "{\"event\": \"setup_fail\", \"why\": \"$(esc "$1")\"}"; finish 2 "H2 FAIL setup: $1"; }

emit "{\"event\": \"start\", \"unix\": $(date +%s), \"cores\": \"$(esc "$H2_CORES")\", \"port\": \"$(esc "$H2_PORT")\", \"cache\": \"$(esc "$SGL_CACHE")\", \"patch\": \"$(esc "$SGL_PATCH_DIR")\", \"switches\": \"$(esc "$KSW")\", \"flags\": \"$(esc "$SMALL")\"}"
[ -x "$PY" ] || setup_fail "host python $PY not found (Neuron DLAMI venv)"
command -v docker > /dev/null 2>&1 || setup_fail "docker not found"
ls /dev/neuron* > /dev/null 2>&1 || setup_fail "no /dev/neuron* devices (this script runs on the trn2 host)"
docker image inspect "$IMAGE" > /dev/null 2>&1 || setup_fail "image $IMAGE is not on this host (R0 setup pulls it)"
ls "$M/Qwen3-0.6B"/*.safetensors > /dev/null 2>&1 || setup_fail "Qwen3-0.6B is not in $M (run_night_1121.sh prep fetches it)"
for f in 50_adapter_server_test.py 79_kernel_ran_check.py srv_sglang_neuron.sh server_ctl.py rollout_client.py rl_common.py backends.json lora_merge.py \
         patches/sglang_neuron_v0.1.0_inplace/general_plugin.py patches/sglang_neuron_v0.1.0_inplace/models/host.py \
         patches/sglang_neuron_v0.1.0_inplace/models/rlkit_kernel_path.py patches/sglang_neuron_v0.1.0_inplace/models/rlkit_lora_merge.py; do
  [ -f "$KIT/$f" ] || setup_fail "$f is missing from the kit folder"
done
# the kernel the server imports (rlkit_lora_merge.py) must be the kernel H1 ran (lora_merge.py): same bytes
m1="$(md5sum "$KIT/lora_merge.py" | cut -d' ' -f1)"
m2="$(md5sum "$SGL_PATCH_DIR/models/rlkit_lora_merge.py" | cut -d' ' -f1)"
[ "$m1" = "$m2" ] || setup_fail "patch rlkit_lora_merge.py ($m2) is not the same file as lora_merge.py ($m1): H1 would not have tested this kernel"
emit "{\"event\": \"files\", \"lora_merge_md5\": \"$m1\", \"host_md5\": \"$(md5sum "$SGL_PATCH_DIR/models/host.py" | cut -d' ' -f1)\", \"kernel_path_md5\": \"$(md5sum "$SGL_PATCH_DIR/models/rlkit_kernel_path.py" | cut -d' ' -f1)\", \"general_plugin_md5\": \"$(md5sum "$SGL_PATCH_DIR/general_plugin.py" | cut -d' ' -f1)\"}"
# cache folder guard: its own, and tied to this one patch folder
case "$(basename "$SGL_CACHE")" in *_kernel_h2) ;; *) setup_fail "cache folder $SGL_CACHE does not end in _kernel_h2: the kit patch must never share a cache folder with another patch (PR 20)";; esac
mkdir -p "$SGL_CACHE" || setup_fail "cannot create the cache folder $SGL_CACHE"
STAMP="$SGL_CACHE/.patch_in_use"
if [ -f "$STAMP" ]; then
  [ "$(cat "$STAMP")" = "$SGL_PATCH_DIR" ] || setup_fail "cache folder $SGL_CACHE belongs to patch '$(cat "$STAMP")', not to $SGL_PATCH_DIR"
else
  echo "$SGL_PATCH_DIR" > "$STAMP"
fi
mkdir -p "$LOGDIR"
# server logs of an earlier run with this tag would be counted by the kernel-ran check: set them aside
for old in "$LOGDIR"/server_${NAME}_*.log; do [ -e "$old" ] && mv "$old" "$old.prev.$(date +%s)"; done
rotate "$RESULT"
say "Qwen3-0.6B TP4, cores $H2_CORES, port $H2_PORT; kit patch $SGL_PATCH_DIR; cache $SGL_CACHE (own, kit patch only); switches: $KSW"

unset SGL_MOUNTS   # extra docker mounts of other runs (srv_sglang_neuron.sh); this run adds none
export MODELS_ROOT SGL_CACHE SGL_PATCH_DIR
export SGL_CORES="$H2_CORES" SGL_EXTRA="$SMALL" SGL_ENV="$KSW"
export SGL_SCRIPT=/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh   # the default of srv_sglang_neuron.sh:57, written out
timeout -k 60 "$H2_TIMEOUT" "$PY" 50_adapter_server_test.py --base "$M/Qwen3-0.6B" --work "$M/adapter_test_06b_h2" \
  --results "$R" --tag "$TAG" --port "$H2_PORT" > "$R/78_kernel_h2_test.out" 2>&1
trc=$?
if [ "$trc" = 124 ] || [ "$trc" = 137 ]; then
  say "test killed at the time cap; saving the container log, then stopping the server container"
  # the test died without its finally block, so nobody saved the log; the name is outside the server_${NAME}_*.log glob
  # (79 must not count it as a server log of a complete run)
  mkdir -p "$LOGDIR"
  bash srv_sglang_neuron.sh logs "$NAME" > "$LOGDIR/timeout_${NAME}.log" 2>&1
fi
bash srv_sglang_neuron.sh stop "$NAME" > /dev/null 2>&1   # harmless when the test already stopped it
sudo chown -R "$(id -u):$(id -g)" "$R" 2> /dev/null || true
emit "{\"event\": \"test_exit\", \"code\": $trc, \"test_stdout\": \"78_kernel_h2_test.out\", \"result_file\": \"50_adapter_$TAG.jsonl\"}"
say "50_adapter_server_test.py exit $trc"
say "updates (seconds, as the client saw them): $(grep -h '"event": "update"' "$RESULT" 2> /dev/null | grep -o '"to": "[a-z0-9]*"\|"seconds": [0-9.]*' | tr '\n' ' ')"

# the two judgements, each its own run of 79_kernel_ran_check.py; the JSON line goes into the record verbatim
LOGS=$(ls "$LOGDIR"/server_${NAME}_*.log 2> /dev/null)
if [ -n "$LOGS" ]; then
  # shellcheck disable=SC2086
  KLINE="$("$PY" 79_kernel_ran_check.py $LOGS --updates 4 --tp-size 4)"; krc=$?
else
  KLINE='{"pass": false, "error": "no server log file"}'; krc=1
fi
[ -n "$KLINE" ] || KLINE='{"pass": false, "error": "79_kernel_ran_check.py printed nothing"}'
emit "{\"event\": \"kernel_ran_check\", \"exit\": $krc, \"result\": $KLINE}"
say "kernel ran check (exit $krc): $(echo "$KLINE" | cut -c1-400)"
if [ -f "$RESULT" ]; then
  ALINE="$("$PY" 79_kernel_ran_check.py --adapter-result "$RESULT" --updates 4)"; arc=$?
else
  ALINE='{"pass": false, "error": "no adapter test result file"}'; arc=1
fi
[ -n "$ALINE" ] || ALINE='{"pass": false, "error": "79_kernel_ran_check.py printed nothing"}'
emit "{\"event\": \"adapter_checks\", \"exit\": $arc, \"result\": $ALINE}"
say "adapter test checks (exit $arc): $(echo "$ALINE" | grep -o '"pass": [a-z]*, "reasons": .*')"
# the first kernel line of the run as a sample of the time split (factors, upload, kernel, copy, read-back check)
SAMPLE="$(grep -h 'adapter update from .* by the kernel (' $LOGS 2> /dev/null | head -n 1 | cut -c1-900)"
[ -n "$SAMPLE" ] && emit "{\"event\": \"kernel_line_sample\", \"line\": \"$(esc "$SAMPLE")\"}" && say "first kernel line: $SAMPLE"

if [ "$trc" = 0 ] && [ "$arc" = 0 ] && [ "$krc" = 0 ]; then
  emit '{"event": "verdict", "result": "pass"}'
  finish 0 "H2 PASS: test exit 0, adapter checks pass, kernel ran in all 4 updates (read $OUT, $RESULT, $LOGDIR)"
fi
emit "{\"event\": \"verdict\", \"result\": \"fail\", \"test_exit\": $trc, \"adapter_checks_exit\": $arc, \"kernel_ran_exit\": $krc}"
finish 1 "H2 FAIL: test exit $trc, adapter checks exit $arc, kernel-ran check exit $krc (stop rule: kernel mode stays off; read $OUT)"
