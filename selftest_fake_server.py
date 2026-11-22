#!/usr/bin/env python3
"""Self-test without any model: a tiny stand-in server, then the client-side scripts against it.

    python selftest_fake_server.py            runs everything and prints PASS or FAILED lines

The stand-in mimics what the docs say of SGLang-Neuron on a `temperature` sampler: it serves any
temperature, refuses a seed together with n > 1 with HTTP 400, and answers the tensor and
distributed weight-update calls with HTTP 400. Its "answers" are arithmetic it computes from the
question text, wrong on purpose for some samples, so rewards have spread.

It also answers the routes of one Miles disk-delta update the way the SGLang under SGLang-Neuron
v0.1.0 does (read in sglang/srt/entrypoints/http_server.py, 22 November; 64_miles_sequence.md has
the line numbers): POST /pause_generation, GET or POST /flush_cache, POST /update_weights_from_disk
(HTTP 200 when the path holds a config.json and a .safetensors file, else HTTP 400 with
"success": false), POST /continue_generation, GET /model_info, GET /ready, POST /update_weight_version,
and a plain 404 for POST /pull_weights, which that SGLang does not have. While paused, a chat request
waits until continue_generation, as the real tokenizer manager does. The checks of that replay are
the "miles" block of main(); 64_miles_replay.py is run against the same server."""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402

REFUSED = {"/update_weights_from_tensor", "/update_weights_from_distributed", "/init_weights_update_group"}
PAUSE_MODES = ("abort", "retract", "in_place")


class State:
    """What the real tokenizer manager holds that the Miles sequence touches."""

    def __init__(self):
        self.cond = threading.Condition()
        self.paused = False
        self.pause_mode = None
        self.model_path = "m"
        self.weight_version = "default"
        self.calls = []          # (method, path) in arrival order, for the order checks
        # long-context and update-under-load behaviour (checks of 67 and 68)
        self.running = 0                 # chat requests past the gates and generating
        self.update_pending = False      # an update waits for running == 0; new requests wait
        self.update_mode = "drain"       # "drain": update waits for running requests; "mix": it does not (a bug)
        self.lose_depth_past = None      # needle deeper than this many tokens is not found (a bug)
        self.lose_total_past = None      # prompt longer than this many tokens loses the needle (a bug)
        self.long_len = 2500             # tokens of a long answer ("Write a very long ...")
        self.gen_chunk = 100             # tokens produced between two looks at the weights
        self.gen_sleep = 0.01            # seconds per chunk


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    @property
    def st(self):
        return self.server.state

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, code, text):
        raw = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?")[0]
        self.st.calls.append(("GET", path))
        if path in ("/health", "/health_generate", "/v1/models"):
            return self._send(200, {"ok": True})
        if path == "/ready":      # http_server.py ready(): 503 while paused
            return self._send_text(503 if self.st.paused else 200, "")
        if path in ("/model_info", "/get_model_info"):   # http_server.py model_info()
            return self._send(200, {"model_path": self.st.model_path, "served_model_name": self.st.model_path,
                                    "is_generation": True, "weight_version": self.st.weight_version})
        if path == "/flush_cache":
            return self._flush()
        self._send(404, {"detail": "Not Found"})

    def _flush(self):
        # http_server.py flush_cache(): GET and POST; 200 with a text body when it flushed
        return self._send_text(200, "Cache flushed.\nPlease check backend logs for more details. "
                                    "(When there are running or waiting requests, the operation will not be "
                                    "performed.)\n")

    def _invalid(self, field):
        # FastAPI's 422 for a body that fails validation
        return self._send(422, {"detail": [{"type": "invalid", "loc": ["body", field],
                                            "msg": f"invalid or missing {field}"}]})

    def _miles_route(self, path, body):
        st = self.st
        if path == "/pause_generation":
            mode = body.get("mode", "abort")
            if mode not in PAUSE_MODES:
                return self._invalid("mode")
            with st.cond:
                st.paused, st.pause_mode = True, mode
            return self._send(200, {"message": "Generation paused successfully.", "status": "ok"})
        if path == "/continue_generation":
            with st.cond:
                st.paused, st.pause_mode = False, None
                st.cond.notify_all()
            return self._send(200, {"message": "Generation continued successfully.", "status": "ok"})
        if path == "/flush_cache":
            return self._flush()
        if path == "/update_weights_from_disk":
            mp = body.get("model_path")
            if not isinstance(mp, str):
                return self._invalid("model_path")
            p = Path(mp)
            if not (p.is_dir() and (p / "config.json").exists() and any(p.glob("*.safetensors"))):
                return self._send(400, {"success": False, "num_paused_requests": 0,
                                        "message": f"Failed to get weights iterator: no model weights in {mp}."})
            msg = "Succeeded to update model weights."
            with st.cond:
                if st.update_mode == "drain":     # the update waits for the requests that are generating
                    st.update_pending = True
                    st.cond.wait_for(lambda: st.running == 0, timeout=120)
                st.model_path = mp
                st.update_pending = False
                st.cond.notify_all()
                if body.get("weight_version") is not None:
                    st.weight_version = str(body["weight_version"])
                    msg += f" Weight version updated to {st.weight_version}."
            return self._send(200, {"success": True, "message": msg, "num_paused_requests": 0})
        if path == "/update_weight_version":
            if not isinstance(body.get("new_version"), str):
                return self._invalid("new_version")
            with st.cond:
                st.weight_version = body["new_version"]
            return self._send(200, {"success": True, "message": f"Weight version updated to {body['new_version']}",
                                    "new_version": body["new_version"]})
        return False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        self.st.calls.append(("POST", self.path))
        if self.path in REFUSED:
            return self._send(400, {"detail": f"{self.path[1:]} is not supported by sglang-neuron: "
                                              "the model's weights are loaded into its compiled graphs once"})
        if self._miles_route(self.path, body) is not False:
            return
        if self.path != "/v1/chat/completions":
            return self._send(404, {"detail": "Not Found"})   # also /pull_weights: not in that SGLang
        with self.st.cond:    # tokenizer_manager.py: a request waits at the pause gate until continue_generation
            self.st.cond.wait_for(lambda: not self.st.paused and not self.st.update_pending, timeout=120)
            self.st.running += 1
        try:
            self._chat(body)
        finally:
            with self.st.cond:
                self.st.running -= 1
                self.st.cond.notify_all()

    def _chat(self, body):
        text_all = " ".join(str(m.get("content", "")) for m in body["messages"])
        prompt_tokens = max(1, len(text_all) // 4)      # proportional to the prompt length
        m = re.search(r"access code for the vault is (\d{6})", text_all)
        if m and "secret access code" in text_all.split(m.group(0))[-1]:
            return self._needle(body, text_all, m, prompt_tokens)
        if "write a very long" in text_all.lower():
            return self._long(body, text_all, prompt_tokens)
        n = int(body.get("n", 1))
        temp = body.get("temperature", 1.0)
        if n > 1 and body.get("seed") is not None and temp:
            return self._send(400, {"detail": "sampling_seed with n > 1 would return n identical samples"})
        text = body["messages"][-1]["content"]
        nums = [int(x) for x in re.findall(r"\d+", text)]
        truth = nums[0] * nums[1] - nums[2] if len(nums) >= 3 else sum(nums)
        rng = random.Random(body.get("seed")) if body.get("seed") is not None else random.Random()
        choices = []
        for i in range(n):
            wrong = bool(temp) and rng.random() < 0.5
            choices.append({"index": i, "message": {"role": "assistant",
                                                    "content": f"Work.\n#### {truth + (1 if wrong else 0)}"},
                            "logprobs": {"content": [{"token": "x", "logprob": -0.1}] * 3}
                            if body.get("logprobs") else None})
        self._send(200, {"choices": choices, "usage": {"completion_tokens": 6 * n, "prompt_tokens": prompt_tokens}})

    def _needle(self, body, text_all, m, prompt_tokens):
        """Answers the code of the needle sentence; loses it past a depth when the stand-in is set to."""
        st = self.st
        depth = m.start() // 4
        lost = ((st.lose_depth_past is not None and depth > st.lose_depth_past)
                or (st.lose_total_past is not None and prompt_tokens > st.lose_total_past))
        code = m.group(1)
        if lost:
            code = str((int(code) + 1) % 1000000).zfill(6)
        self._send(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": f"The code is {code}."},
                                      "logprobs": None}],
                         "usage": {"completion_tokens": 6, "prompt_tokens": prompt_tokens}})

    def _long(self, body, text_all, prompt_tokens):
        """A long answer, token by token from a hash of (weights, prompt, position). The weights are read
        before every chunk: in mode "mix" an update that lands in the middle changes the rest of the answer."""
        st = self.st
        total = min(int(body.get("max_tokens", 16)), st.long_len)
        toks, lps = [], []
        while len(toks) < total:
            with st.cond:
                wset = st.model_path
            for i in range(len(toks), min(total, len(toks) + st.gen_chunk)):
                h = zlib.crc32(f"{wset}|{text_all}|{i}".encode())
                toks.append(f"w{h % 97}")
                lps.append(-0.01 - (h % 5000) / 1000.0)
            time.sleep(st.gen_sleep)
        content = [{"token": t, "logprob": p} for t, p in zip(toks, lps)] if body.get("logprobs") else None
        self._send(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(toks)},
                                      "logprobs": {"content": content} if content else None}],
                         "usage": {"completion_tokens": total, "prompt_tokens": prompt_tokens}})


def check(name, cond, results):
    print(("PASS " if cond else "FAILED ") + name, flush=True)
    results.append(bool(cond))


def miles_checks(srv, url, out):
    """One Miles disk-delta update replayed by hand with rollout_client.post / get, then by 64_miles_replay.py.
    The routes, methods and fields are those of Miles's UpdateWeightFromDiskDelta._reload_engines (see
    64_miles_sequence.md); the answers asserted are those of the SGLang under SGLang-Neuron (same file)."""
    st = srv.state
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "local_ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}", encoding="utf-8")
        (ckpt / "model.safetensors").write_bytes(b"x")
        st.calls.clear()
        # 1. pull_weights: the route the SGLang under SGLang-Neuron does not have
        code, b, _ = rc.post(url + "/pull_weights", {"local_checkpoint_dir": str(ckpt),
                                                     "source_dir": "/shared/delta", "target_version": 1})
        check("miles: pull_weights answers 404 Not Found", code == 404 and b == {"detail": "Not Found"}, out)
        code, b = rc.get(url + "/model_info")
        check("miles: model_info starts at weight_version default",
              code == 200 and json.loads(b)["weight_version"] == "default", out)
        # 2. pause_generation, mode retract (the mode Miles's own test sets; Miles's default was not read)
        code, b, _ = rc.post(url + "/pause_generation", {"mode": "retract"})
        check("miles: pause_generation 200 status ok", code == 200 and b.get("status") == "ok", out)
        code, b, _ = rc.post(url + "/pause_generation", {"mode": "sideways"})
        check("miles: a pause mode outside abort/retract/in_place is 422", code == 422, out)
        check("miles: /ready is 503 while paused", rc.get(url + "/ready")[0] == 503, out)
        late = {}
        m = C.messages_for("A farmer has 3 rows of 4 trees. He cuts 5.")
        th = threading.Thread(target=lambda: late.update(r=rc.chat(url, "m", m, 0.0, 8)))
        th.start()
        th.join(0.4)
        check("miles: a chat request waits at the pause gate", th.is_alive(), out)
        # 3. flush_cache is a GET in Miles; the real route takes GET and POST
        code, text = rc.get(url + "/flush_cache")
        check("miles: GET flush_cache 200 'Cache flushed'", code == 200 and text.startswith("Cache flushed."), out)
        check("miles: POST flush_cache 200", rc.post(url + "/flush_cache", {})[0] == 200, out)
        # 4. update_weights_from_disk with the two fields Miles sends
        code, b, _ = rc.post(url + "/update_weights_from_disk",
                             {"model_path": str(Path(d) / "nothing_here"), "weight_version": "42"})
        check("miles: update_weights_from_disk on a path with no weights is 400 success false",
              code == 400 and b.get("success") is False and "num_paused_requests" in b, out)
        check("miles: the failed update left the weight version alone",
              json.loads(rc.get(url + "/model_info")[1])["weight_version"] == "default", out)
        code, b, _ = rc.post(url + "/update_weights_from_disk", {"model_path": str(ckpt), "weight_version": "42"})
        check("miles: update_weights_from_disk 200 success true with the answer fields",
              code == 200 and b.get("success") is True and set(b) == {"success", "message", "num_paused_requests"}
              and "Weight version updated to 42" in b["message"], out)
        code, b, _ = rc.post(url + "/update_weights_from_disk", {"weight_version": "1"})
        check("miles: update_weights_from_disk without model_path is 422", code == 422, out)
        # 5. continue_generation releases the waiting request
        code, b, _ = rc.post(url + "/continue_generation", {})
        check("miles: continue_generation 200 status ok", code == 200 and b.get("status") == "ok", out)
        th.join(10)
        check("miles: the waiting chat request is answered after continue",
              not th.is_alive() and late.get("r", {}).get("status") == 200, out)
        check("miles: /ready is 200 again", rc.get(url + "/ready")[0] == 200, out)
        # 6. the read-back Miles makes under --ci-test
        code, b = rc.get(url + "/model_info")
        info = json.loads(b)
        check("miles: model_info shows weight_version 42 and the new model_path",
              info["weight_version"] == "42" and info["model_path"] == str(ckpt), out)
        code, b, _ = rc.post(url + "/update_weight_version", {"new_version": "43", "abort_all_requests": False})
        check("miles: update_weight_version answers success and new_version",
              code == 200 and b.get("success") is True and b.get("new_version") == "43", out)
        order = [(mth, p) for mth, p in st.calls if p != "/v1/chat/completions"]
        want = [("POST", "/pull_weights"), ("GET", "/model_info"), ("POST", "/pause_generation"),
                ("POST", "/pause_generation"), ("GET", "/ready"), ("GET", "/flush_cache"),
                ("POST", "/flush_cache"), ("POST", "/update_weights_from_disk"), ("GET", "/model_info"),
                ("POST", "/update_weights_from_disk"), ("POST", "/update_weights_from_disk"),
                ("POST", "/continue_generation"), ("GET", "/ready"), ("GET", "/model_info"),
                ("POST", "/update_weight_version")]
        check("miles: the server saw the calls in the order sent", order == want, out)

        # 7. the same sequence through 64_miles_replay.py, as a program run against a server URL
        st.calls.clear()
        st.weight_version, st.paused = "default", False
        rd = Path(d) / "res"
        for mode, expect in (("retract", ["/pull_weights", "/pause_generation", "/flush_cache",
                                          "/update_weights_from_disk", "/continue_generation", "/model_info"]),
                             ("in_place", ["/pull_weights", "/pause_generation",
                                           "/update_weights_from_disk", "/continue_generation", "/model_info"])):
            p = subprocess.run([sys.executable, str(HERE / "64_miles_replay.py"), "--base-url", url,
                                "--ckpt", str(ckpt), "--weight-version", "7", "--pause-mode", mode,
                                "--results", str(rd), "--tag", mode], capture_output=True, text=True)
            f = rd / f"64_miles_replay_{mode}.jsonl"
            rows = C.read_jsonl(f) if p.returncode == 0 else []
            calls = [r for r in rows if r.get("event") == "call"]
            summ = [r for r in rows if r.get("event") == "summary"]
            check(f"64 replay ({mode}) writes a clean results file", p.returncode == 0 and C.finished_ok(f), out)
            check(f"64 replay ({mode}) sends the routes in the Miles order",
                  [r["route"] for r in calls] == expect, out)
            by = {r["name"]: r for r in calls}
            check(f"64 replay ({mode}) records status and seconds per call",
                  all(isinstance(r["status"], int) and r["seconds"] >= 0 for r in calls), out)
            check(f"64 replay ({mode}) sees pull_weights 404 and nothing else refused",
                  by.get("pull_weights", {}).get("status") == 404
                  and all(r["miles_accepts"] for r in calls if r["name"] != "pull_weights"), out)
            check(f"64 replay ({mode}) says Miles would stop at pull_weights",
                  bool(summ) and summ[0]["miles_would_stop_at"] == "pull_weights"
                  and summ[0]["miles_would_succeed"] is False
                  and summ[0]["pause_to_continue_seconds"] is not None, out)
            check(f"64 replay ({mode}) reloaded weight version 7",
                  by.get("model_info", {}).get("weight_version_answered") == "7", out)
        check("64 replay left the server unpaused", st.paused is False, out)
        # Miles's verify_engine_version raises on a version mismatch; the replay must count that as a stop
        import importlib.util
        spec = importlib.util.spec_from_file_location("miles_replay_64", HERE / "64_miles_replay.py")
        mr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mr)
        check("64 replay refuses a model_info weight_version that differs from the one sent",
              mr.miles_accepts(200, {"weight_version": "6"}, False, "7") is False
              and mr.miles_accepts(200, {"weight_version": "7"}, False, "7") is True
              and mr.miles_accepts(200, {"model_path": "m"}, False, "7") is False, out)


def run_script(name, args):
    p = subprocess.run([sys.executable, str(HERE / name)] + args, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-800:], flush=True)
    return p


def needle_run(srv, url, d, tag, extra=None, **flags):
    """67_long_context_needle.py against the stand-in set to the given misbehaviour; returns its rows by event."""
    st = srv.state
    st.lose_depth_past, st.lose_total_past = flags.get("lose_depth_past"), flags.get("lose_total_past")
    p = run_script("67_long_context_needle.py", ["--base-url", url, "--model", "m", "--results", str(d),
                                                 "--tag", tag] + (extra or []))
    f = Path(d) / f"67_long_context_{tag}.jsonl"
    rows = C.read_jsonl(f) if f.exists() else []
    st.lose_depth_past = st.lose_total_past = None
    return p.returncode == 0 and C.finished_ok(f), rows


def long_checks(srv, url, out):
    """67 (needle at depth) and 68 (update under load) against the stand-in, healthy and misbehaving."""
    st = srv.state
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # ---- 67: healthy -----------------------------------------------------------------------
        ok, rows = needle_run(srv, url, d, "ok")
        conds = [r for r in rows if r.get("event") == "condition"]
        summ = [r for r in rows if r.get("event") == "summary"]
        check("67 healthy: results file clean, 6 depth and 3 late conditions",
              ok and [c["case"] for c in conds] == ["depth"] * 6 + ["late"] * 3, out)
        check("67 healthy: calibration written, tokens per sentence positive",
              any(r.get("event") == "calibration" and r["tokens_per_sentence"] > 0 for r in rows), out)
        check("67 healthy: every needle found, no HTTP error",
              all(c["found"] == 5 and c["needles"] == 5 and c["http_errors"] == 0 for c in conds), out)
        check("67 healthy: needle depth within 3 percent of the target (from usage.prompt_tokens)",
              all(abs(c["needle_depth_measured"] - c["target_tokens"]) <= 0.03 * c["target_tokens"]
                  for c in conds if c["case"] == "depth"), out)
        check("67 healthy: late prompts within 3 percent of the target length",
              all(abs(c["prompt_tokens_mean"] - c["target_tokens"]) <= 0.03 * c["target_tokens"]
                  for c in conds if c["case"] == "late"), out)
        check("67 healthy: summary says nothing lost above the boundary",
              bool(summ) and summ[0]["depth"]["prompt_above_boundary"]["rate"] == 1.0
              and summ[0]["late"]["prompt_above_boundary"]["rate"] == 1.0
              and summ[0]["depth"]["first_condition_with_loss"] is None, out)
        # ---- 67: needle lost past a depth (the bug of the plan, in the prefill view) -------------------
        ok, rows = needle_run(srv, url, d, "lost_depth", lose_depth_past=8192)
        conds = [r for r in rows if r.get("event") == "condition"]
        summ = [r for r in rows if r.get("event") == "summary"]
        dep = {c["target_tokens"]: c["found"] for c in conds if c["case"] == "depth"}
        check("67 needle lost past depth 8192: found above, not found below",
              ok and dep == {1000: 5, 4000: 5, 7000: 5, 9000: 0, 12000: 0, 15000: 0}, out)
        check("67 needle lost past depth 8192: the summary reports it and the first condition with loss",
              bool(summ) and summ[0]["depth"]["prompt_above_boundary"]["rate"] == 0.0
              and summ[0]["depth"]["prompt_up_to_boundary"]["rate"] == 1.0
              and summ[0]["depth"]["first_condition_with_loss"]["target_tokens"] == 9000, out)
        check("67 needle lost past depth 8192: the early-needle case is not affected",
              all(c["found"] == 5 for c in conds if c["case"] == "late"), out)
        # ---- 67: decode side: early needle, long prompt ---------------------------------------------
        ok, rows = needle_run(srv, url, d, "lost_total", lose_total_past=8192)
        conds = [r for r in rows if r.get("event") == "condition"]
        summ = [r for r in rows if r.get("event") == "summary"]
        late = {c["target_tokens"]: c["found"] for c in conds if c["case"] == "late"}
        check("67 needle lost with a prompt past 8192: the late case loses it at 9000, 12000, 15000",
              ok and late == {9000: 0, 12000: 0, 15000: 0}, out)
        check("67 needle lost with a prompt past 8192: the summary of the late case says so",
              bool(summ) and summ[0]["late"]["prompt_above_boundary"]["rate"] == 0.0
              and summ[0]["late"]["first_condition_with_loss"]["target_tokens"] == 9000, out)
        # ---- 67: a context too small for a condition ---------------------------------------------------
        ok, rows = needle_run(srv, url, d, "small_ctx", extra=["--context-length", "8192"])
        sk = [r for r in rows if r.get("event") == "skipped"]
        check("67 skips the conditions that do not fit the context",
              ok and len(sk) >= 5 and all(r["target_tokens"] >= 9000 or r["case"] == "depth" for r in sk)
              and not any(r.get("event") == "condition" and r["target_tokens"] >= 9000 for r in rows), out)

        # ---- 68: update under load ---------------------------------------------------------------------
        base, alt = d / "base", d / "alt"
        for p in (base, alt):
            p.mkdir()
            (p / "config.json").write_text("{}", encoding="utf-8")
            (p / "model.safetensors").write_bytes(b"x")
        common = ["--base-url", url, "--model", "m", "--base", str(base), "--alt", str(alt), "--results", str(d),
                  "--threads", "4", "--updates", "3", "--pause", "0.3", "--warmup", "0.2"]

        def load_run(tag, mode, gen_sleep, long_len=2500, extra=None):
            st.update_mode, st.gen_sleep, st.long_len = mode, gen_sleep, long_len
            st.model_path = str(base)
            p = run_script("68_update_under_load.py", common + ["--tag", tag] + (extra or []))
            f = d / f"68_update_under_load_{tag}.jsonl"
            rows = C.read_jsonl(f) if f.exists() else []
            return p.returncode == 0 and C.finished_ok(f), rows

        ok, rows = load_run("drain", "drain", 0.02)
        by = {}
        for r in rows:
            by.setdefault(r.get("event"), []).append(r)
        summ = (by.get("summary") or [{}])[0]
        reqs = by.get("request", [])
        ups = by.get("update", [])
        check("68 drain: results file clean, control repeats the base answers",
              ok and by["control"][0]["base_repeat_identical"] == 4 and by["references"][0]["alt_differs_from_base"] == 4, out)
        check("68 drain: 3 updates, all HTTP 200, seconds written, requests running at the call",
              len(ups) == 3 and all(u["status"] == 200 and u["ok"] and u["seconds"] >= 0 for u in ups)
              and all(u["running_at_call"] > 0 for u in ups), out)
        check("68 drain: every request is class base or alt, none neither, none HTTP error",
              len(reqs) >= 8 and all(r["class"] in ("base", "alt") for r in reqs) and summ["http_errors"] == 0
              and summ["consistent_with_one_weight_set"] is True, out)
        check("68 drain: some requests overlapped an update and are still consistent",
              summ["requests_overlapping_an_update"] > 0
              and summ["classes_of_overlapping_requests"]["neither"] == 0, out)
        check("68 drain: both weight sets were served and the answers have 2400 tokens",
              {r["class"] for r in reqs} == {"base", "alt"} and all(r["tokens"] == 2400 for r in reqs)
              and summ["requests_below_min_tokens"] == 0, out)
        check("68 drain: the server ends on the weights the script believes",
              by["after"][0]["answers_match_reference"] == 4, out)
        # the stand-in lets the update land in the middle of answers
        ok, rows = load_run("mix", "mix", 0.02, extra=["--pause", "1.0"])   # a request spans one update at most
        by = {}
        for r in rows:
            by.setdefault(r.get("event"), []).append(r)
        summ = (by.get("summary") or [{}])[0]
        neither = [r for r in by.get("request", []) if r["class"] == "neither"]
        check("68 mixed weights: the script reports answers of neither weight set and says not consistent",
              ok and len(neither) > 0 and summ["classes"]["neither"] == len(neither)
              and summ["consistent_with_one_weight_set"] is False, out)
        check("68 mixed weights: the stop position and the splice share are written",
              bool(neither) and all(0 < r["stops_matching_base_at"] < r["tokens"] or 0 < r["stops_matching_alt_at"] < r["tokens"]
                                    for r in neither)
              and max(r["after_base_stop_matches_alt"] or 0 for r in neither) >= 0.9, out)
        check("68 mixed weights: those answers overlapped an update",
              all(r["overlapped_updates"] for r in neither), out)
        # answers that end early are reported, not repaired
        ok, rows = load_run("short", "drain", 0.0, long_len=300, extra=["--updates", "1"])
        by = {}
        for r in rows:
            by.setdefault(r.get("event"), []).append(r)
        summ = (by.get("summary") or [{}])[0]
        check("68 short answers: reference_short written and every request counted below min tokens",
              ok and bool(by.get("reference_short")) and summ["requests_below_min_tokens"] == summ["requests"] > 0
              and all(r["tokens"] == 300 for r in by["request"]), out)
        st.update_mode, st.gen_sleep, st.long_len = "drain", 0.01, 2500


def main():
    out = []
    # pure functions
    check("gold answer", C.gold_answer("He has 3.\n#### 1,234") == 1234.0, out)
    check("extract after ####", C.extract_answer("so 5\n#### $42.") == 42.0, out)
    check("extract last number", C.extract_answer("first 3 then 17") == 17.0, out)
    check("extract skips thinking", C.extract_answer("<think>99</think>#### 7") == 7.0, out)
    check("no number", C.extract_answer("none") is None, out)
    check("advantages equal rewards", C.group_advantages([1, 1, 1, 1]) == [0.0] * 4, out)
    adv = C.group_advantages([1, 0, 0, 0])
    check("advantages sum to zero", abs(sum(adv)) < 1e-6 and adv[0] > 0 > adv[1], out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.state = State()
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    try:
        check("wait_ready", rc.wait_ready(url, "m", 10, poll_s=0.2) is not None, out)
        m = C.messages_for("A farmer has 17 rows of 23 trees. He cuts 41. How many are left?")
        g = rc.sample_group(url, "m", m, 8, 0.7, 64, mode="auto", seed=5)
        check("auto falls back to parallel when seed + n is refused",
              g["mode"] == "parallel" and len(g["texts"]) == 8 and g["errors"] == 0, out)
        g2 = rc.sample_group(url, "m", m, 8, 0.7, 64, mode="n", seed=None)
        check("n=8 without seed", g2["mode"] == "n" and len(g2["texts"]) == 8, out)
        rs = [C.reward(t, 17 * 23 - 41) for t in g["texts"]]
        check("rewards are 0 or 1", set(rs) <= {0.0, 1.0}, out)
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "gsm.jsonl"
            data.write_text("".join(json.dumps(
                {"question": f"A farmer has {a} rows of {b} trees. He cuts {c}. How many are left?",
                 "answer": f"x\n#### {a * b - c}"}) + "\n" for a, b, c in
                [(3, 4, 5), (10, 10, 1), (7, 8, 6), (2, 9, 4)]), encoding="utf-8")
            p = subprocess.run([sys.executable, str(HERE / "21_gsm8k_acc.py"), "--base-url", url,
                                "--model", "m", "--data", str(data), "--results", d, "--tag", "t",
                                "--concurrency", "4"], capture_output=True, text=True)
            rows = C.read_jsonl(Path(d) / "21_gsm8k_t.jsonl") if p.returncode == 0 else []
            ps = [r for r in rows if r.get("event") == "pass"]
            check("21 greedy accuracy 1.0 on the stand-in",
                  p.returncode == 0 and ps and ps[0]["accuracy"] == 1.0 and C.finished_ok(Path(d) / "21_gsm8k_t.jsonl"), out)
            p = subprocess.run([sys.executable, str(HERE / "21_gsm8k_acc.py"), "compare", "--a",
                                str(Path(d) / "21_gsm8k_t.jsonl"), "--b", str(Path(d) / "21_gsm8k_t.jsonl")],
                               capture_output=True, text=True)
            check("21 compare", p.returncode == 0 and "paired questions 4" in p.stdout, out)
            p = subprocess.run([sys.executable, str(HERE / "20_rl_probes.py"), "--base-url", url,
                                "--model", "m", "--results", d, "--tag", "t"],
                               capture_output=True, text=True)
            rows = C.read_jsonl(Path(d) / "20_rl_probes_t.jsonl") if p.returncode == 0 else []
            by = {r["probe"]: r for r in rows if r.get("event") == "probe"}
            check("20 runs to the done line", p.returncode == 0 and C.finished_ok(Path(d) / "20_rl_probes_t.jsonl"), out)
            check("20 sees the seed + n refusal", by.get("sampled n=8 with seed", {}).get("status") == 400, out)
            check("20 sees n=8 served", by.get("sampled n=8", {}).get("answers") == 8, out)
            check("20 sees the weight-update refusal",
                  by.get("post /update_weights_from_disk", {}).get("status") == 400, out)
            check("20 seed honoured", by.get("seed honoured", {}).get("same_seed_same_text") is True, out)
            check("20 logprobs counted", by.get("greedy logprobs", {}).get("logprob_tokens") == 3, out)
        miles_checks(srv, url, out)
        long_checks(srv, url, out)
    finally:
        srv.shutdown()
    print("selftest", "passed" if all(out) else "FAILED", f"({sum(out)} of {len(out)})")
    sys.exit(0 if all(out) else 1)


if __name__ == "__main__":
    main()
