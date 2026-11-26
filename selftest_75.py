#!/usr/bin/env python3
"""Self-test of 75_pr20_check.py without a device: the stand-in server of selftest_fake_server.py, extended here.

    python selftest_75.py          runs every case and prints PASS or FAILED lines, then "selftest_75 passed (n of m)"

How 48 is tested in selftest_fake_server.py: it is not. That file runs the client scripts against an in-process
stand-in server by URL (--base-url). 48, 49 and 75 start their own server through server_ctl.Server and
backends.json, so here the stand-in is started as a child process per start (`python selftest_75.py --serve ...`),
through a backends.json written for each case; server_ctl then starts, stops and restarts it as it would a container.
The stand-in answers greedy chat requests with token logprobs that are a hash of (weights set, prompt, position),
takes /update_weights_from_disk of selftest_fake_server.py (drain: it waits for running requests), and has one
misbehaviour per case (--mode):
    ok            as it should be
    stale_cache   the first prompt's answer is cached and the cache is not cleared by an update
    drift         every answer after an in-place update differs by 1e-4 in each logprob (breaks C == A)
    alt_drift     answers after an in-place update to alt differ by 1e-4 (B != D, C == A)
    fail_update:N the Nth update call answers HTTP 400 "success": false
    inflight_error  a request that is running when an update arrives answers HTTP 500
    load_skip     an update that arrives while a request runs answers 200 and changes nothing
    repeat_differs  a cache hit on the first prompt before any update differs by 1e-4 (75's soft check radix_repeat;
                    C == A and B == D still hold, so the run ends with exit 0 and a "warn" line)
What this does not cover: the real route's answers, a real radix cache, device numerics, docker (srv_sglang_neuron.sh),
the path mapping under MODELS_ROOT, and a server that dies during the smoke (the stand-in is stopped only by the test).
Windows is enough: the process group kill of server_ctl falls back to terminate()."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import selftest_fake_server as F  # noqa: E402

N_TOK = 40
STALE_KEY = "17 rows of 23"


class H75(F.H):
    def _miles_route(self, path, body):
        st = self.st
        if path == "/update_weights_from_disk":
            with st.cond:
                st.n_updates += 1
                n = st.n_updates
                running = st.running
            if st.mode.startswith("fail_update:") and n == int(st.mode.split(":")[1]):
                return self._send(400, {"success": False, "message": "Failed to update weights in place: injected",
                                        "num_paused_requests": 0})
            if st.mode == "load_skip" and running > 0:
                return self._send(200, {"success": True, "message": "Succeeded to update model weights.",
                                        "num_paused_requests": 0})
            r = super()._miles_route(path, body)
            if Path(str(body.get("model_path", ""))).name == Path(st.model_path).name:
                with st.cond:
                    st.updated_in_place = True
                    if st.mode != "stale_cache":
                        st.cache.clear()
            return r
        return super()._miles_route(path, body)

    def _chat(self, body):
        st = self.st
        text_all = " ".join(str(m.get("content", "")) for m in body["messages"])
        n_tok = min(int(body.get("max_tokens", 16)), N_TOK)
        key = (text_all, n_tok)
        with st.cond:
            cached = st.cache.get(key)
        cached_hit = cached is not None
        if cached is None:
            with st.cond:
                wset = st.model_path
                inplace = st.updated_in_place
            drift = 0.0
            if st.mode == "drift" and inplace:
                drift = 1e-4
            if st.mode == "alt_drift" and inplace and Path(wset).name == "alt":
                drift = 1e-4
            content = []
            for i in range(n_tok):
                h = zlib.crc32(f"{Path(wset).name}|{text_all}|{i}".encode())
                content.append({"token": f"w{h % 97}", "logprob": -0.01 - (h % 5000) / 1000.0 - drift})
            cached = content
            if st.mode != "stale_cache" or STALE_KEY in text_all:
                with st.cond:
                    st.cache[key] = content
        if st.mode == "repeat_differs" and cached_hit and STALE_KEY in text_all and not st.updated_in_place:
            cached = [dict(t, logprob=t["logprob"] - 1e-4) for t in cached]   # a hit served with other numerics
        time.sleep(st.gen_sleep)      # a request takes time whether or not the answer came from the cache
        if st.mode == "inflight_error" and st.update_pending:
            return self._send(500, {"detail": "injected: request cut by the update"})
        self._send(200, {"choices": [{"index": 0, "message": {"role": "assistant",
                                                              "content": " ".join(t["token"] for t in cached)},
                                      "logprobs": {"content": cached}}],
                         "usage": {"completion_tokens": n_tok, "prompt_tokens": max(1, len(text_all) // 4)}})


def serve(port, model, mode):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H75)
    st = F.State()
    st.model_path, st.mode, st.n_updates, st.cache, st.updated_in_place = model, mode, 0, {}, False
    st.gen_sleep = 0.05
    srv.state = st
    srv.serve_forever()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def check(name, cond, out):
    print(("PASS " if cond else "FAILED ") + name, flush=True)
    out.append(bool(cond))


def run_case(root, name, mode, extra=None):
    """75_pr20_check.py against the stand-in in this mode; returns (exit code, rows, stdout lines, port, manifest)."""
    d = Path(root) / name
    models = d / "models"
    for p in ("base", "alt"):
        (models / p).mkdir(parents=True)
        (models / p / "config.json").write_text("{}", encoding="utf-8")
        (models / p / "model.safetensors").write_bytes(b"x")
    port = free_port()
    backends = d / "backends.json"
    backends.write_text(json.dumps({"fake": {"start": [sys.executable, str(Path(__file__).resolve()), "--serve",
                                                       "--port", "{port}", "--model", "{model}", "--mode", mode],
                                             "ready_timeout_s": 90, "stop_grace_s": 10}}), encoding="utf-8")
    manifest = d / "manifest.json"
    man = {"patch_dir": "/fake/pr20", "files": {"general_plugin.py": "md5-1", "models/host.py": "md5-2"}}
    manifest.write_text(json.dumps(man), encoding="utf-8")
    args = [sys.executable, str(HERE / "75_pr20_check.py"), "--base", str(models / "base"), "--alt", str(models / "alt"),
            "--results", str(d / "res"), "--tag", name, "--backend", "fake", "--backends-json", str(backends),
            "--port", str(port), "--smoke-updates", "4", "--smoke-threads", "4"]
    if not (extra and "--manifest-json" in extra):
        args += ["--manifest-json", str(manifest)]
    args += extra or []
    env = dict(os.environ, SGL_PATCH_DIR="/fake/pr20", SGL_CACHE="/fake/cache", SGL_CORES="0-3",
               SGL_EXTRA="--context-length 1024")
    env.pop("SGL_SCRIPT", None)
    env.pop("SGL_ENV", None)
    p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=600)
    f = d / "res" / f"75_pr20_check_{name}.jsonl"
    rows = C.read_jsonl(f) if f.exists() else []
    return p.returncode, rows, p.stdout.strip().splitlines(), port, man, p.stderr[-600:]


def ev(rows, event, **kv):
    return [r for r in rows if r.get("event") == event and all(r.get(k) == v for k, v in kv.items())]


def chk(rows, name):
    return next((r for r in rows if r.get("event") == "check" and r.get("name") == name), {})


def fails(rows):
    return [r for r in rows if r.get("status") == "fail"]


def server_gone(port):
    return rc.get(f"http://127.0.0.1:{port}/health", timeout=2)[0] == 0


def unit_checks(out):
    spec = importlib.util.spec_from_file_location("t75", HERE / "75_pr20_check.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)

    def ans(text, lps):
        return {"status": 200, "text": text, "lp": list(lps), "toks": [f"t{i}" for i in range(len(lps))], "error": None}

    x = [ans("a", [-1.0, -2.0]), ans("b", [-0.5])]
    check("75 unit: 20 prompts, all different", len(t.PROMPTS) == 20 and len(set(t.PROMPTS)) == 20, out)
    check("75 unit: equal answers are identical with max diff 0.0",
          t.compare(x, [dict(v) for v in x])["identical"] and t.compare(x, x)["max_abs_logprob_diff"] == 0.0, out)
    y = [ans("a", [-1.0, -2.0 - 1e-9]), ans("b", [-0.5])]
    c = t.compare(x, y)
    check("75 unit: a difference of 1e-9 is not identical and is reported unrounded",
          not c["identical"] and abs(c["max_abs_logprob_diff"] - 1e-9) < 1e-12 and c["max_abs_logprob_diff"] != 1e-9
          and c["differing_prompt_indices"] == [0], out)   # unrounded: the float difference, not 1e-9 rounded to 12 places
    z = t.compare(x, [ans("a", [-1.0, -2.0 - 3e-13]), x[1]])
    check("75 unit: a difference of 3e-13 is reported unrounded (not 0.0) and is not identical",
          not z["identical"] and z["max_abs_logprob_diff"] > 0.0, out)
    check("75 unit: different text, different length, a failed request, no tokens are not identical",
          not t.compare(x, [ans("z", [-1.0, -2.0]), x[1]])["identical"]
          and not t.compare(x, [ans("a", [-1.0]), x[1]])["identical"]
          and not t.compare(x, [dict(x[0], status=500), x[1]])["identical"]
          and not t.compare([ans("", [])], [ans("", [])])["identical"]
          and not t.compare(x, x[:1])["identical"], out)


def main():
    out = []
    unit_checks(out)
    p = subprocess.run([sys.executable, str(HERE / "75_pr20_check.py"), "--help"], capture_output=True, text=True)
    check("75 --help lists the arguments", p.returncode == 0 and all(
        k in p.stdout for k in ("--base", "--alt", "--results", "--tag", "--backend", "--backends-json", "--port",
                                "--prompts", "--smoke-updates", "--smoke-threads", "--skip-fresh",
                                "--manifest-json")), out)
    p = subprocess.run([sys.executable, str(HERE / "75_pr20_check.py"), "--base", "b", "--alt", "a", "--results", "r",
                        "--tag", "t", "--prompts", "21"], capture_output=True, text=True)
    check("75 --prompts 21 is refused", p.returncode == 2, out)

    cases = {"ok": ("ok", None), "stale": ("stale_cache", None), "alt_drift": ("alt_drift", None),
             "drift": ("drift", None), "fail3": ("fail_update:3", None), "inflight": ("inflight_error", None),
             "skip": ("load_skip", None), "repeat": ("repeat_differs", None), "nofresh": ("ok", ["--skip-fresh"]),
             "badman": ("ok", ["--manifest-json", "does_not_exist.json"]),
             "fewer": ("ok", ["--prompts", "5", "--smoke-updates", "3"])}
    with tempfile.TemporaryDirectory() as root:
        with ThreadPoolExecutor(max_workers=len(cases)) as ex:
            futs = {k: ex.submit(run_case, root, k, m, e) for k, (m, e) in cases.items()}
            res = {k: f.result() for k, f in futs.items()}

        # ---- ok ---------------------------------------------------------------------------------------
        rcode, rows, so, port, man, err = res["ok"]
        if rcode != 0:
            print(err)
        check("ok: exit 0, file ends with the done line, no fail line",
              rcode == 0 and C.finished_ok(Path(root) / "ok/res/75_pr20_check_ok.jsonl") and not fails(rows), out)
        first = rows[0] if rows else {}
        check("ok: first line holds the SGL_* values and the manifest",
              first.get("event") == "open" and first["environment"]["SGL_PATCH_DIR"] == "/fake/pr20"
              and first["environment"]["SGL_CACHE"] == "/fake/cache" and first["environment"]["SGL_CORES"] == "0-3"
              and first["environment"]["SGL_EXTRA"] == "--context-length 1024"
              and first["environment"]["SGL_SCRIPT"] is None and first["environment"]["SGL_ENV"] is None
              and first["patch_manifest"] == man and "manifest_error" not in first, out)
        check("ok: three answer sets of 20 prompts (A, B, C) and D", len(ev(rows, "answers")) == 4 and all(
            len(r["answers"]) == 20 for r in ev(rows, "answers")), out)
        ups = ev(rows, "update")
        check("ok: update lines B, C, 4 smoke updates, 1 under load; all ok with status, seconds and body",
              [u["step"] for u in ups] == ["B", "C", "smoke 1", "smoke 2", "smoke 3", "smoke 4", "smoke under load"]
              and all(u["ok"] and u["status"] == 200 and u["seconds"] >= 0 and u["body"] for u in ups), out)
        check("ok: smoke updates alternate alt, base and the under-load update goes to alt",
              [u["to"] for u in ups] == ["alt", "base", "alt", "base", "alt", "base", "alt"], out)
        ca = chk(rows, "back_to_base_equals_A")
        check("ok: C == A, max difference exactly 0.0 over 20 prompts",
              ca.get("identical") is True and ca.get("max_abs_logprob_diff") == 0.0 and ca.get("prompts") == 20, out)
        bd = chk(rows, "inplace_equals_fresh")
        check("ok: B == D, max difference exactly 0.0", bd.get("identical") is True
              and bd.get("max_abs_logprob_diff") == 0.0 and bd.get("texts_identical") is True, out)
        rx = chk(rows, "radix_cache_indirect")
        check("ok: radix check present, says indirect, passes",
              rx.get("first_prompt_after_update_equals_fresh") is True and "indirect" in rx.get("note", ""), out)
        check("ok: the first prompt was repeated before the update and changed by it",
              chk(rows, "first_prompt_repeat_before_update").get("identical") is True
              and chk(rows, "first_prompt_changed_by_update").get("changed") is True, out)
        smk = ev(rows, "smoke")
        check("ok: the smoke line says smoke only, not a load test",
              len(smk) == 1 and "smoke only, not a load test" in smk[0]["note"], out)
        ul = (ev(rows, "smoke_under_load") or [{}])[0]
        check("ok: under load, requests were in flight at the call, none failed, answers match the new weights",
              ul.get("in_flight_at_call", 0) > 0 and ul.get("requests_finished", 0) > 0 and ul.get("http_errors") == 0
              and ul.get("answers_equal_reference_of_new_weights") is True, out)
        check("ok: summary passed, last printed line is PASSED, server stopped",
              ev(rows, "summary")[0]["passed"] is True and so[-1].startswith("75_pr20_check PASSED")
              and server_gone(port), out)

        # ---- stale cache ------------------------------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["stale"]
        check("stale cache: exit 1, radix_indirect fail line, done line last, FAILED printed",
              rcode == 1 and any(r.get("stage") == "radix_indirect" for r in fails(rows))
              and rows[-1] == {"done": True} and so[-1].startswith("75_pr20_check FAILED") and server_gone(port), out)
        check("stale cache: the first prompt is reported unchanged by the update and different from the fresh start",
              chk(rows, "first_prompt_changed_by_update").get("changed") is False
              and chk(rows, "radix_cache_indirect").get("first_prompt_after_update_equals_fresh") is False, out)

        # ---- B != D ------------------------------------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["alt_drift"]
        bd = chk(rows, "inplace_equals_fresh")
        check("alt drift: exit 1, B vs D difference reported (about 1e-4), C == A still passes",
              rcode == 1 and bd.get("identical") is False and bd.get("max_abs_logprob_diff") is not None
              and 5e-5 < bd["max_abs_logprob_diff"] < 2e-4 and chk(rows, "back_to_base_equals_A").get("identical") is True
              and any(r.get("stage") == "fresh_vs_inplace" for r in fails(rows)), out)

        # ---- C != A: stop before the smoke -----------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["drift"]
        check("drift: exit 1 at step C, no smoke, no restart, server stopped, done line last",
              rcode == 1 and chk(rows, "back_to_base_equals_A").get("identical") is False
              and any(r.get("stage") == "C_answers" for r in fails(rows)) and not ev(rows, "smoke")
              and not ev(rows, "restart_alt") and rows[-1] == {"done": True} and server_gone(port), out)

        # ---- an update refused in the smoke -------------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["fail3"]
        sm = [u for u in ev(rows, "update") if u["step"].startswith("smoke")]
        check("failed smoke update: exit 1, recorded with status 400, no further updates on that server",
              rcode == 1 and len(sm) == 1 and sm[0]["ok"] is False and sm[0]["status"] == 400
              and any(r.get("stage") == "smoke_updates" for r in fails(rows)) and not ev(rows, "smoke_under_load"), out)
        check("failed smoke update: the restart and the fresh-start check still ran and passed",
              chk(rows, "inplace_equals_fresh").get("identical") is True and rows[-1] == {"done": True}, out)

        # ---- HTTP error in flight ------------------------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["inflight"]
        ul = (ev(rows, "smoke_under_load") or [{}])[0]
        check("HTTP error in flight: exit 1, errors counted with status and text",
              rcode == 1 and ul.get("http_errors", 0) > 0 and ul["first_errors"][0]["status"] == 500
              and any(r.get("stage") == "smoke_under_load" for r in fails(rows)), out)

        # ---- an update under load that does nothing -------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["skip"]
        ul = (ev(rows, "smoke_under_load") or [{}])[0]
        check("update under load acknowledged but not applied: exit 1, answers not those of the new weights",
              rcode == 1 and ul.get("update_ok") is True and ul.get("answers_equal_reference_of_new_weights") is False
              and any(r.get("stage") == "smoke_under_load" for r in fails(rows)), out)

        # ---- the repeat of the first prompt differs: soft, the run goes on ---------------------------------------------
        rcode, rows, so, port, _, _ = res["repeat"]
        warns = [r for r in rows if r.get("status") == "warn" and r.get("stage") == "radix_repeat"]
        summ = (ev(rows, "summary") or [{}])[0]
        check("repeat differs: exit 0, one warn line radix_repeat, no fail line, B, C and D still ran and passed",
              rcode == 0 and len(warns) == 1 and not fails(rows) and chk(rows, "first_prompt_repeat_before_update").get(
                  "identical") is False and chk(rows, "back_to_base_equals_A").get("identical") is True
              and chk(rows, "inplace_equals_fresh").get("identical") is True and len(ev(rows, "answers")) == 4
              and rows[-1] == {"done": True} and server_gone(port), out)
        check("repeat differs: the summary lists the warning and still says passed; PASSED line names it",
              summ.get("warnings") == ["radix_repeat"] and summ.get("passed") is True
              and so[-1].startswith("75_pr20_check PASSED WITH WARNING radix_repeat"), out)

        # ---- --skip-fresh ----------------------------------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["nofresh"]
        check("--skip-fresh: exit 0, no restart, the fresh and radix checks written as not_run",
              rcode == 0 and not ev(rows, "restart_alt") and chk(rows, "inplace_equals_fresh").get("result") == "not_run"
              and chk(rows, "radix_cache_indirect").get("result") == "not_run" and rows[-1] == {"done": True}, out)

        # ---- manifest missing ------------------------------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["badman"]
        check("unreadable --manifest-json: exit 1, error in the first line, done line, no server started",
              rcode == 1 and "manifest_error" in rows[0] and rows[-1] == {"done": True} and not ev(rows, "start_base")
              and so[-1].startswith("75_pr20_check FAILED"), out)

        # ---- fewer prompts, odd smoke count ------------------------------------------------------------------------
        rcode, rows, so, port, _, _ = res["fewer"]
        ul = (ev(rows, "smoke_under_load") or [{}])[0]
        check("--prompts 5 --smoke-updates 3: exit 0, 5 answers per set, smoke ends on alt so the load update goes to base",
              rcode == 0 and all(len(r["answers"]) == 5 for r in ev(rows, "answers")) and ul.get("to") == "base"
              and ul.get("answers_equal_reference_of_new_weights") is True, out)
    print("selftest_75", "passed" if all(out) else "FAILED", f"({sum(out)} of {len(out)})")
    sys.exit(0 if all(out) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--mode", default="ok")
    a = ap.parse_args()
    if a.serve:
        serve(a.port, a.model, a.mode)
    else:
        main()
