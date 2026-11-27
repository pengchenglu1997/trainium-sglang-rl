#!/usr/bin/env python3
"""Session 4 (2026-11-26): one server, three identical rollout passes of 32 concurrent chat requests, no /start_profile.

The device profile, when wanted, comes from the launch-time serve window: put SGLANG_NEURON_PROFILE_SERVE_RANKS,
SGLANG_NEURON_PROFILE_SERVE_START, SGLANG_NEURON_PROFILE_SERVE_STEPS and SGLANG_NEURON_PROFILE_DIR into SGL_ENV
(59_decode_profile.py's /start_profile call is refused by a server launched with the window, so it is left out here).
With --ignore-eos every request produces exactly --max-tokens tokens, so all rows are live for the whole pass and
the pass has max-tokens decode steps (the live rows and the answer lengths are recorded next to the capture).

    (environment of srv_sglang_neuron.sh)
    python s4_profile.py --model DIR --train-jsonl FILE --results DIR --tag q8_rel --cores 0-3 --port 30101 \
        --temperature 0.8 [--top-k 50 --top-p 0.95] --ignore-eos
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402


def one_pass(url, name, rows, seed0, a):
    extra = {}
    if a.top_k is not None:
        extra["top_k"] = a.top_k
    if a.top_p is not None:
        extra["top_p"] = a.top_p
    if a.ignore_eos:
        extra["ignore_eos"] = True

    def one(ir):
        i, r = ir
        return rc.chat(url, name, C.messages_for(r[1]), temperature=a.temperature, max_tokens=a.max_tokens,
                       seed=seed0 + i, extra=extra or None, timeout=900)
    t = time.time()
    with ThreadPoolExecutor(max_workers=len(rows)) as ex:
        outs = list(ex.map(one, list(enumerate(rows))))
    sec = time.time() - t
    tok = [o["completion_tokens"] or 0 for o in outs]
    return {"seconds": round(sec, 3), "requests": len(rows), "completion_tokens": sum(tok),
            "shortest_answer_tokens": min(tok) if tok else 0, "longest_answer_tokens": max(tok) if tok else 0,
            "tok_per_s": round(sum(tok) / sec, 1) if sec > 0 else None,
            "errors": sum(o["status"] != 200 for o in outs),
            "first_error": next((o["error"] for o in outs if o["status"] != 200), None),
            "first_text": (outs[0]["texts"][0][:160] if outs and outs[0]["texts"] else None),
            "request_extra": extra, "temperature": a.temperature}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--backend", default="sglang_neuron")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    res = C.Results(a.results, f"s4_profile_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=f"prof_{a.tag}", env={"SGL_CORES": a.cores})
    rows = C.load_gsm8k(a.train_jsonl, limit=2000)[-a.requests:]
    ok = False
    try:
        st = srv.start(a.model)
        res.write(event="server_start", **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready")
        url, name = srv.base_url, srv.served_name()
        bad = 0
        for p in range(a.passes):
            r = one_pass(url, name, rows, 5000000 + 100000 * p, a)
            bad += r["errors"]
            res.write(event="pass", which=p, **r)
        ok = bad == 0
    except BaseException as e:  # noqa: BLE001
        res.fail("profile", e)
        raise
    finally:
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
        res.done()
        print(f"S4_PROFILE_{'DONE' if ok else 'FAILED'} {a.tag}")


if __name__ == "__main__":
    main()
