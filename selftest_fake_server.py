#!/usr/bin/env python3
"""Self-test without any model: a tiny stand-in server, then the client-side scripts against it.

    python selftest_fake_server.py            runs everything and prints PASS or FAILED lines

The stand-in mimics what the docs say of SGLang-Neuron on a `temperature` sampler: it answers
chat requests, refuses a request without a temperature... (no: it serves any temperature),
refuses a seed together with n > 1 with HTTP 400, and answers the weight-update calls with
HTTP 400. Its "answers" are arithmetic it computes from the question text, wrong on purpose
for some samples, so rewards have spread."""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402

REFUSED = {"/update_weights_from_disk", "/update_weights_from_tensor",
           "/update_weights_from_distributed", "/init_weights_update_group"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            return self._send(200, {"ok": True})
        self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path in REFUSED:
            return self._send(400, {"detail": f"{self.path[1:]} is not supported by sglang-neuron: "
                                              "the model's weights are loaded into its compiled graphs once"})
        if self.path != "/v1/chat/completions":
            return self._send(404, {"detail": "Not Found"})
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
        self._send(200, {"choices": choices, "usage": {"completion_tokens": 6 * n}})


def check(name, cond, results):
    print(("PASS " if cond else "FAILED ") + name, flush=True)
    results.append(bool(cond))


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
    finally:
        srv.shutdown()
    print("selftest", "passed" if all(out) else "FAILED", f"({sum(out)} of {len(out)})")
    sys.exit(0 if all(out) else 1)


if __name__ == "__main__":
    main()
