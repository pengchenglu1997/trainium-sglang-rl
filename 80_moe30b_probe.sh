#!/usr/bin/env bash
# R8: Qwen3-30B-A3B start probe on the integration branch (upstream PRs 17 and 18 merged in our branch
# "integration"; first step of ROADMAP L3). Question: does SGLang-Neuron with that code start Qwen3-30B-A3B
# TP4 EP4 on a trn2 and answer a known-answer arithmetic prompt, and what does one small bench cell give.
#
#   bash 80_moe30b_probe.sh <results dir>
#
# THE PROBE MAKES NO BASELINE CLAIM. The integration code has never been run by us, there is no run of the
# release code (v0.1.0 as released) on this model beside it, and the bench cell is one point (concurrency 8,
# 1024 prompts of the llmbench client, 32 requests). Do not set the number beside the PR authors' numbers
# (220.2 to 371.0 tok/s, TPOT 45.8 to 17.1 ms on a trn2.3xlarge) and do not call it a baseline or a
# reproduction. Reproduction is ROADMAP L3, a later session.
#
# How the integration code reaches the image (option 1 of the task): patches/sglang_neuron_integration/ holds
# the 16 package .py files that differ from v0.1.0 (4e24326) in branch integration (12 changed, 4 new under
# models/llama/), exported byte-exact with "git show integration:<path>"; BLOBS.txt there has the git blob id of
# each. The launcher's SGL_PATCH_DIR already mounts every .py of a folder over the file of the same relative path
# in /usr/local/lib/python3.14/site-packages/sglang_neuron, so no launcher change is needed for them. The
# integration tree has no example script for Qwen3-30B-A3B (examples/ has none; the docs have the command in
# docs/model-recipes/qwen3-moe.md, "Serving"), so this script writes a small launch script with that command into
# the results folder and mounts it with SGL_MOUNTS (the one optional variable added to srv_sglang_neuron.sh).
# Why not a full directory mount of the package (option 2): it would hide any file of the image that the export
# does not hold (compiled files, data files) and would make "which code ran" depend on the whole tree; the
# per-file mount changes exactly the 16 files, and their md5 and blob ids are written to the result file.
#
# Flags: the docs' aggregated Qwen3-30B-A3B command (docs/model-recipes/qwen3-moe.md): --tp-size 4 --ep-size 4
# --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --mem-fraction-static 0.8, with
# --max-total-tokens 131072 instead of 266240 (same file, section "On Trainium2": 266240 is Trn3's; on a
# trn2.3xlarge the pool at 0.8 holds 215,040 tokens, so "state --max-total-tokens 131072"; ROADMAP L3 also says
# use 131072). Hybrid prefill: SGLANG_NEURON_MOE_PREFILL is left at its default "auto", which the code
# (ops/moe/moe_cte_bf16.py _use_hybrid) turns into the hybrid kernel on trn2 when the chunk is a multiple of 256
# tokens up to 2048 and the experts are whole per rank (EP 4 = TP 4); "hybrid" is not forced because it raises
# for any prefill shape outside that table, which the smaller warm-up graphs may be. PR 18's selective decode is
# on by default (SGLANG_NEURON_MOE_DECODE_SELECTIVE unset = 4 tokens). Extra server environment: PROBE_ENV
# ("K=V K2=V2").
#
# Environment (all optional):
#   DATA_ROOT      default $HOME/w          MODELS_ROOT  default $DATA_ROOT/models
#   SGL_CACHE      default $DATA_ROOT/sgl_cache_integration_moe30b (its own cache: built under this code, the
#                  kernel-source digest differs from the release and from our other patches)
#   PROBE_TAG      default r8               result file 80_moe30b_<tag>.jsonl
#   PROBE_PORT     default 30030            PROBE_CORES  default 0-3 (logical cores for TP4 EP4: four)
#   READY_CAP      default 2400 s to wait for ready (cold compile has no record)
#   FETCH_CAP      default 3600 s for the checkpoint download, or for waiting on the background fetch that
#                  "run_session3.sh prep" starts (pid file <results dir>/fetch_bg_Qwen3-30B-A3B.pid; while that
#                  process is alive this script waits and does not start a second download; a finished folder is
#                  recognised by the .fetch_ok marker); if the fetch is not there or failed, this script fetches
#   BENCH_CAP      default 1800 s for the bench cell
#   MAX_TOTAL_TOKENS default 131072         PROBE_SAMPLER default greedy (the docs' default, no sampling here)
#   HUB_ID         default Qwen/Qwen3-30B-A3B  (UNVERIFIED: the files read name the model "Qwen3-30B-A3B" and the
#                  folder /models/Qwen3-30B-A3B; the hub organisation prefix is not in any file read)
#   PY             default python3 (the Neuron venv is activated when it exists)
# Last line of the output: "READY ..." or "NOT_READY ...", each with the last 5 lines of the server log.
# Result file ends with {"done": true}. Exit 0 only on READY.
set -u
KIT="$(cd "$(dirname "$0")" && pwd)"
case "${1:-}" in -h|--help) sed -n '2,60p' "$0"; exit 0 ;; esac
RES_ARG="${1:?usage: bash 80_moe30b_probe.sh <results dir>}"
mkdir -p "$RES_ARG" || { echo "NOT_READY cannot create results dir $RES_ARG"; exit 1; }
RES="$(cd "$RES_ARG" && pwd)"
[ -f /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate ] && source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
PY="${PY:-python3}"
export DATA_ROOT="${DATA_ROOT:-$HOME/w}"
export MODELS_ROOT="${MODELS_ROOT:-$DATA_ROOT/models}"
export SGL_CACHE="${SGL_CACHE:-$DATA_ROOT/sgl_cache_integration_moe30b}"
M="$MODELS_ROOT"
TAG="${PROBE_TAG:-r8}"
PORT="${PROBE_PORT:-30030}"
CORES="${PROBE_CORES:-0-3}"
READY_CAP="${READY_CAP:-2400}"
FETCH_CAP="${FETCH_CAP:-3600}"
BENCH_CAP="${BENCH_CAP:-1800}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}"
SAMPLER="${PROBE_SAMPLER:-greedy}"
HUB_ID="${HUB_ID:-Qwen/Qwen3-30B-A3B}"
FOLDER="Qwen3-30B-A3B"
IMAGE="${SGL_IMAGE:-public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04}"
PATCH_DIR="${SGL_PATCH_DIR:-$KIT/patches/sglang_neuron_integration}"
SRV="$KIT/srv_sglang_neuron.sh"
NAME="srv_moe30b_$TAG"
OUT="$RES/80_moe30b_$TAG.jsonl"
HELPER="$RES/80_moe30b_helper.py"
LAUNCH="$RES/80_moe30b_server.sh"
LOGF="$RES/server_logs/80_moe30b_$TAG.log"
BASE_URL="http://127.0.0.1:$PORT"
SERVED="/models/$FOLDER"
mkdir -p "$RES/logs" "$RES/server_logs"
say() { echo "== 80 $(date -u +%H:%M:%S) $*"; }
export PROBE_KIT="$KIT"

# ---- the helper (standard library only): json lines, patch proof, ready wait, known-answer chat ----
cat > "$HELPER" <<'PYEOF'
import hashlib, json, os, re, subprocess, sys, time
sys.path.insert(0, os.environ["PROBE_KIT"])
import rollout_client as rc  # noqa: E402


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(path, event, **kv):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "t_utc": now(), **kv}) + "\n")


def parse(v):
    if v.startswith("@"):  # the last 4000 characters of a file
        try:
            return open(v[1:], encoding="utf-8", errors="replace").read()[-4000:]
        except OSError as e:
            return "unreadable %s: %s" % (v[1:], e)
    try:
        return json.loads(v)
    except ValueError:
        return v


def kvs(args):
    out = {}
    for a in args:
        k, _, v = a.partition("=")
        out[k] = parse(v)
    return out


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def patch_proof(d):
    """md5 and git blob id of every .py of the patch folder (the files the launcher mounts); BLOBS.txt of the
    folder lists the blob ids of the exported branch files."""
    files, bad = {}, []
    listed = {}
    try:
        for line in open(os.path.join(d, "BLOBS.txt"), encoding="utf-8"):
            p = line.split()
            if len(p) == 3:
                listed[p[2]] = p[0]
    except OSError as e:
        bad.append("BLOBS.txt: %s" % e)
    for root, dirs, names in os.walk(d):
        dirs[:] = [x for x in dirs if x not in ("orig", "__pycache__")]
        for n in names:
            if not n.endswith(".py"):
                continue
            full = os.path.join(root, n)
            rel = os.path.relpath(full, d).replace(os.sep, "/")
            data = open(full, "rb").read()
            files[rel] = {"md5": hashlib.md5(data).hexdigest(), "blob": blob_id(data)}
            if listed.get(rel) != files[rel]["blob"]:
                bad.append("blob of %s is %s, BLOBS.txt says %s" % (rel, files[rel]["blob"], listed.get(rel)))
    for rel in listed:
        if rel not in files:
            bad.append("%s is in BLOBS.txt but not in the folder" % rel)
    return files, bad


def main():
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "emit":  # emit FILE EVENT k=v ...
        emit(a[0], a[1], **kvs(a[2:]))
    elif cmd == "done":  # done FILE
        with open(a[0], "a", encoding="utf-8") as f:
            f.write(json.dumps({"done": True}) + "\n")
    elif cmd == "start":  # start FILE PATCH_DIR k=v ...
        files, bad = patch_proof(a[1])
        emit(a[0], "start", patch_dir=a[1], patch_files=files, patch_problems=bad, **kvs(a[2:]))
        if bad:
            print("; ".join(bad)[:600])
            sys.exit(1)
    elif cmd == "ready":  # ready BASE MODEL CAP_S SRV NAME
        base, model, cap, srv, name = a[0], a[1], float(a[2]), a[3], a[4]
        t0, last = time.time(), 0
        while time.time() - t0 < cap:
            try:
                alive = subprocess.run(["bash", srv, "alive", name], capture_output=True, text=True,
                                       timeout=60).stdout.strip() == "true"
            except Exception:  # noqa: BLE001
                alive = True
            if not alive:
                print("DIED %.1f" % (time.time() - t0))
                sys.exit(2)
            r = rc.chat(base, model, [{"role": "user", "content": "Say OK."}], temperature=0.0, max_tokens=1,
                        timeout=120)
            if r["status"] == 200:
                print("%.1f" % (time.time() - t0))
                return
            if time.time() - last > 120:
                last = time.time()
                sys.stderr.write("== 80 %s waiting %ds (status %s)\n" % (time.strftime("%H:%M:%S", time.gmtime()),
                                                                          time.time() - t0, r["status"]))
            time.sleep(5)
        print("TIMEOUT %.1f" % (time.time() - t0))
        sys.exit(1)
    elif cmd == "chat":  # chat FILE BASE MODEL: one greedy request, known answer 17 * 23 = 391
        path, base, model = a
        prompt = "What is 17 * 23? Answer with the number only."
        r = rc.chat(base, model, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64,
                    timeout=900)
        text = r["texts"][0] if r["texts"] else ""
        nums = [x.replace(",", "") for x in re.findall(r"\d[\d,]*", text)]
        ok = r["status"] == 200 and "391" in nums
        emit(path, "chat", prompt=prompt, expected="391", status=r["status"], text=text, correct=ok,
             seconds=round(r["seconds"], 2), completion_tokens=r["completion_tokens"], error=r["error"],
             temperature=0.0, thinking="off (chat_template_kwargs enable_thinking false, as the kit client)")
        print("CHAT status=%s correct=%s text=%r" % (r["status"], ok, text[:120]))
        sys.exit(0 if ok else 1)
    else:
        sys.exit("unknown command " + cmd)


main()
PYEOF

# ---- the launch script of the server (the docs' Qwen3-30B-A3B command), mounted into the container ----
body=$(cat <<'LEOF'
#!/bin/bash
# Qwen3-30B-A3B aggregated, TP4 EP4 on four NeuronCores. Flags of docs/model-recipes/qwen3-moe.md (Serving) with
# --max-total-tokens of the section "On Trainium2". Cache variables as the examples' run_server.sh sets them.
set -euo pipefail
MODEL_PATH=${MODEL_PATH:?}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
CACHE_DIR=${CACHE_DIR:-/cache}
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo \
    NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --tp-size 4 --ep-size 4 --context-length 4096 --chunked-prefill-size 2048 \
    --max-running-requests 64 --max-total-tokens @MAX_TOTAL@ --mem-fraction-static 0.8 \
    --host "$HOST" --port "$PORT" "$@"
LEOF
)
printf '%s\n' "${body//@MAX_TOTAL@/$MAX_TOTAL_TOKENS}" > "$LAUNCH"
chmod +x "$LAUNCH" 2>/dev/null || true

# ---- one exit path: save the log, stop the server, write verdict and the done line, print the last line ----
VERDICT=NOT_READY; REASON="not finished"; FINISHED=0; STOP_S=""
finish() {
  [ "$FINISHED" = 1 ] && return
  FINISHED=1
  trap - EXIT INT TERM
  bash "$SRV" logs "$NAME" > "$LOGF" 2>&1 || true
  local t0; t0=$(date +%s)
  bash "$SRV" stop "$NAME" > /dev/null 2>&1 || true
  STOP_S=$(( $(date +%s) - t0 ))
  local tail5; tail5=$(tail -n 5 "$LOGF" 2>/dev/null | cut -c1-240 | tr '\r\n' '  ' | sed 's/  */ /g')
  [ -s "$LOGF" ] || tail5="(server log empty or missing)"
  "$PY" "$HELPER" emit "$OUT" stop stop_seconds="$STOP_S" log="$LOGF" || true
  "$PY" "$HELPER" emit "$OUT" verdict verdict="$VERDICT" reason="$REASON" log_tail="@$LOGF" || true
  "$PY" "$HELPER" done "$OUT" || true
  echo "$VERDICT $REASON | result $OUT | log $LOGF | last 5 log lines: $tail5"
}
trap 'REASON="interrupted"; finish; exit 130' INT TERM
trap 'finish' EXIT

: > "$OUT"
# header of the result file: which code ran (md5 and blob id of every mounted file) and every setting
if ! "$PY" "$HELPER" start "$OUT" "$PATCH_DIR" probe=R8_moe30b_start_probe \
     baseline_claim=none image="$IMAGE" hub_id="$HUB_ID" hub_id_status="UNVERIFIED (no file read names it)" \
     model_dir="$M/$FOLDER" sgl_cache="$SGL_CACHE" cores="$CORES" port="$PORT" max_total_tokens="$MAX_TOTAL_TOKENS" \
     sampler="$SAMPLER" ready_cap_s="$READY_CAP" server_env="${PROBE_ENV:-}" launch_script="$LAUNCH" \
     moe_prefill_env="${SGLANG_NEURON_MOE_PREFILL:-unset (auto)}" integration_commit="4ad8056a4ffc56b4ad4105fc47dc2ad2d27cd980" \
     > "$RES/logs/80_start_$TAG.out" 2>&1; then
  REASON="patch folder does not match BLOBS.txt: $(cut -c1-300 "$RES/logs/80_start_$TAG.out")"
  say "$REASON"; finish; exit 1
fi
say "patch folder ok: $(grep -c . "$PATCH_DIR/BLOBS.txt") files, md5 and blob ids in $OUT"
case "$LAUNCH" in *" "*) REASON="the results path holds a space (SGL_MOUNTS splits items on spaces): $RES"; say "$REASON"; finish; exit 1 ;; esac

# ---- checkpoint ----
say "disk: $(df -h "$M" 2>/dev/null | tail -n 1 | tr -s ' ')"
mkdir -p "$M"
T_FETCH=$(date +%s)
have_model() { ls "$M/$FOLDER"/*.safetensors > /dev/null 2>&1 && [ -f "$M/$FOLDER/.fetch_ok" ]; }
# a fetch that "run_session3.sh prep" started in the background (detached; pid file $RES/fetch_bg_<folder>.pid, removed
# when it ends): wait for it up to FETCH_CAP instead of starting a second download into the same folder
fetch_running() { local p; [ -f "$PIDF" ] && p="$(cat "$PIDF" 2>/dev/null)" && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
PIDF="$RES/fetch_bg_$FOLDER.pid"
if ! have_model && fetch_running; then
  say "background fetch of $FOLDER is running (pid $(cat "$PIDF")); waiting up to $FETCH_CAP s for it; df: $(df -h "$M" 2>/dev/null | tail -n 1 | tr -s ' ')"
  while fetch_running && [ $(( $(date +%s) - T_FETCH )) -lt "$FETCH_CAP" ]; do sleep 20; done
  if fetch_running; then
    REASON="the background fetch of $FOLDER is still running after FETCH_CAP $FETCH_CAP s (it keeps running; see $RES/logs/hf_bg_$FOLDER.log, then run r8 again)"
    say "$REASON"; finish; exit 1
  fi
  if have_model; then say "background fetch finished: $(du -sh "$M/$FOLDER" 2>/dev/null | cut -f1)"; else say "background fetch ended without .fetch_ok; this script fetches itself"; fi
fi
if have_model; then
  say "model $FOLDER present"
else
  docker image inspect "$IMAGE" > /dev/null 2>&1 || docker pull "$IMAGE" > "$RES/logs/80_docker_pull.out" 2>&1 \
    || { REASON="docker pull failed: $(tail -n 1 "$RES/logs/80_docker_pull.out" | cut -c1-200)"; say "$REASON"; finish; exit 1; }
  docker rm -f "hf_$NAME" > /dev/null 2>&1 || true
  timeout "$FETCH_CAP" docker run --rm --name "hf_$NAME" -v "$M":/models "$IMAGE" hf download "$HUB_ID" \
    --local-dir "/models/$FOLDER" > "$RES/logs/80_hf_$FOLDER.out" 2>&1
  rc=$?
  if [ "$rc" != 0 ]; then
    docker rm -f "hf_$NAME" > /dev/null 2>&1 || true
    REASON="checkpoint fetch of $HUB_ID failed (exit $rc): $(tail -n 2 "$RES/logs/80_hf_$FOLDER.out" | cut -c1-200 | tr '\n' ' ')"
    say "$REASON"; finish; exit 1
  fi
  sudo chown -R "$(id -u):$(id -g)" "$M/$FOLDER" 2>/dev/null || true
  touch "$M/$FOLDER/.fetch_ok"
fi
SIZE_B=$(du -sb "$M/$FOLDER" 2>/dev/null | cut -f1); SIZE_H=$(du -sh "$M/$FOLDER" 2>/dev/null | cut -f1)
say "checkpoint $FOLDER size $SIZE_H ($SIZE_B bytes), fetch step $(( $(date +%s) - T_FETCH )) s"
"$PY" "$HELPER" emit "$OUT" fetch hub_id="$HUB_ID" size_bytes="${SIZE_B:-0}" size_human="$SIZE_H" \
  seconds="$(( $(date +%s) - T_FETCH ))" || true
ls "$M/$FOLDER"/*.safetensors > /dev/null 2>&1 || { REASON="no .safetensors in $M/$FOLDER"; say "$REASON"; finish; exit 1; }

# ---- server ----
say "starting $NAME on cores $CORES, port $PORT, cache $SGL_CACHE"
T_START=$(date +%s)
SGL_CORES="$CORES" SGL_SAMPLER="$SAMPLER" SGL_PATCH_DIR="$PATCH_DIR" SGL_ENV="${PROBE_ENV:-}" \
SGL_SCRIPT=/bin/bash SGL_EXTRA="/probe/80_moe30b_server.sh" SGL_MOUNTS="$LAUNCH:/probe/80_moe30b_server.sh:ro" \
  bash "$SRV" start "$M/$FOLDER" "$PORT" "$NAME" > "$RES/logs/80_launcher_$TAG.out" 2>&1
rc=$?
if [ "$rc" != 0 ]; then
  REASON="launcher exit $rc: $(tail -n 2 "$RES/logs/80_launcher_$TAG.out" | cut -c1-200 | tr '\n' ' ')"
  say "$REASON"; finish; exit 1
fi
sleep 3
say "container started; waiting up to $READY_CAP s for the first answer"
RDY=$("$PY" "$HELPER" ready "$BASE_URL" "$SERVED" "$READY_CAP" "$SRV" "$NAME")
rc=$?
if [ "$rc" != 0 ]; then
  case "$RDY" in
    DIED*) REASON="server process ended after ${RDY#DIED } s, no first answer" ;;
    *) REASON="not ready within $READY_CAP s" ;;
  esac
  "$PY" "$HELPER" emit "$OUT" ready ready_s=null outcome="$RDY" || true
  say "$REASON"; finish; exit 1
fi
say "ready after $RDY s"
"$PY" "$HELPER" emit "$OUT" ready ready_s="$RDY" outcome=ready || true

# ---- one known-answer chat, then one bench cell ----
CH=$("$PY" "$HELPER" chat "$OUT" "$BASE_URL" "$SERVED")
rc=$?
say "$CH"
if [ "$rc" != 0 ]; then
  REASON="ready after $RDY s but the known-answer request failed: $(echo "$CH" | cut -c1-200)"
  "$PY" "$HELPER" emit "$OUT" bench status=skipped reason="no sane answer, no benchmark of a wrong server" || true
  finish; exit 1
fi
if [ ! -f "$KIT/22b_llmbench_grid.sh" ]; then
  BENCH="missing 22b_llmbench_grid.sh"; BRC=127
  say "bench: $BENCH"
else
  say "bench cell: LB_CONC=8 LB_PROMPTS=1024 LB_REQ=32"
  LB_CONC=8 LB_PROMPTS=1024 LB_REQ=32 timeout "$BENCH_CAP" bash "$KIT/22b_llmbench_grid.sh" "moe30b_$TAG" "$BASE_URL" \
    "$SERVED" "$M/$FOLDER" "$RES" 4 > "$RES/logs/80_bench_$TAG.out" 2>&1
  BRC=$?
  BENCH=$(tail -n 1 "$RES/logs/80_bench_$TAG.out" | cut -c1-300)
  say "bench exit $BRC: $BENCH"
fi
"$PY" "$HELPER" emit "$OUT" bench exit_code="$BRC" last_line="$BENCH" log="$RES/logs/80_bench_$TAG.out" \
  cell="LB_CONC=8 LB_PROMPTS=1024 LB_REQ=32" note="one point; no baseline claim" || true
VERDICT=READY
REASON="ready after $RDY s, 17*23 answered 391, bench exit $BRC ($BENCH)"
finish
exit 0
