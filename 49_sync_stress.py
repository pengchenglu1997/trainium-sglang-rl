#!/usr/bin/env python3
"""What breaks when weights are synced in place many times, and while requests are in flight?

    (same environment as 48_inplace_server_test.py: patched SGLang-Neuron)
    python 49_sync_stress.py --base DIR --alt DIR --results DIR --tag stress \
        [--quiet-cycles 100] [--load-cycles 40] [--load-threads 8]

Reference: greedy answers with token logprobs to fixed prompts on --base (A) and, after one
update, on --alt (B). The two checkpoints differ by small noise, so A and B differ in logprobs.
Phase "quiet": --quiet-cycles times, update to the other checkpoint, ask, and compare with that
  checkpoint's reference (text and every token logprob). Recorded: seconds of each update (first
  ten, last ten, mean, max), answers that did not match, device memory lines of the server log
  are left to the log.
Phase "load": --load-threads threads keep asking one prompt; the main thread updates back and
  forth --load-cycles times with a short pause. Every answer is classed by its logprob vector:
  equal to A, equal to B, or neither (an answer computed with mixed or half-written weights, or
  cut by the update). Recorded: counts per class, HTTP errors, seconds of the updates under load.
An answer that is "neither" is the finding: it means the update is not fenced from requests."""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402

PROMPTS = [
    "A farmer has 17 rows of 23 apple trees. He cuts down 41 trees. How many trees are left?",
    "Tom reads 12 pages a day for 9 days and then 30 pages more. How many pages did he read?",
]


def ask_one(base_url, model, q, max_tokens=32, timeout=300):
    body = {"model": model, "messages": C.messages_for(q), "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": True, "chat_template_kwargs": {"enable_thinking": False}}
    code, out, sec = rc.post(base_url.rstrip("/") + "/v1/chat/completions", body, timeout=timeout)
    if code != 200 or not isinstance(out, dict):
        return {"status": code, "text": None, "lp": None, "error": str(out)[:200]}
    ch = out["choices"][0]
    lp = tuple(round(t.get("logprob"), 5) for t in ((ch.get("logprobs") or {}).get("content") or []))
    return {"status": 200, "text": ch["message"]["content"], "lp": lp}


def ask_all(base_url, model):
    return [ask_one(base_url, model, q) for q in PROMPTS]


def equal(x, y):
    return all(a["status"] == 200 and b["status"] == 200 and a["text"] == b["text"] and a["lp"] == b["lp"]
               for a, b in zip(x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, default=30012)
    ap.add_argument("--quiet-cycles", type=int, default=100)
    ap.add_argument("--load-cycles", type=int, default=40)
    ap.add_argument("--load-threads", type=int, default=8)
    ap.add_argument("--load-pause", type=float, default=0.5)
    a = ap.parse_args()
    res = C.Results(a.results, f"49_sync_stress_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=f"srv_{a.tag}")
    try:
        st = srv.start(a.base)
        res.write(event="start", **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready")
        name = srv.served_name()
        paths = {"A": a.base, "B": a.alt}
        ref = {"A": ask_all(srv.base_url, name)}
        u = srv.update_weights_from_disk(a.alt)
        if not u["ok"]:
            raise RuntimeError(f"first update refused: {u}")
        ref["B"] = ask_all(srv.base_url, name)
        res.write(event="reference", a_differs_from_b=not equal(ref["A"], ref["B"]),
                  tokens=[len(x["lp"] or ()) for x in ref["A"]], first_update=u)
        if equal(ref["A"], ref["B"]):
            raise RuntimeError("the two checkpoints give identical answers: nothing to tell apart")

        # ---- quiet: many updates, each verified --------------------------------------------
        now, secs, bad, failed = "B", [], 0, 0
        for i in range(a.quiet_cycles):
            nxt = "A" if now == "B" else "B"
            u = srv.update_weights_from_disk(paths[nxt])
            if not u["ok"]:
                failed += 1
                res.write(event="quiet_update_failed", cycle=i, **u)
                continue
            now = nxt
            secs.append(u["seconds"])
            if not equal(ask_all(srv.base_url, name), ref[now]):
                bad += 1
        res.write(event="quiet", cycles=a.quiet_cycles, updates_ok=len(secs), updates_failed=failed,
                  answers_not_matching_reference=bad, update_s=C.describe(secs),
                  first_ten_s=secs[:10], last_ten_s=secs[-10:], server_alive=srv.alive())

        # ---- load: updates while requests are in flight -------------------------------------
        stop = threading.Event()
        counts = {"A": 0, "B": 0, "neither": 0, "http_error": 0}
        examples = []
        lock = threading.Lock()
        ra, rb = ref["A"][0], ref["B"][0]

        def worker():
            while not stop.is_set():
                x = ask_one(srv.base_url, name, PROMPTS[0])
                if x["status"] != 200:
                    k = "http_error"
                elif x["text"] == ra["text"] and x["lp"] == ra["lp"]:
                    k = "A"
                elif x["text"] == rb["text"] and x["lp"] == rb["lp"]:
                    k = "B"
                else:
                    k = "neither"
                with lock:
                    counts[k] += 1
                    if k in ("neither", "http_error") and len(examples) < 5:
                        examples.append({"class": k, "status": x["status"], "error": x.get("error"),
                                         "tokens": len(x["lp"] or ()),
                                         "max_abs_diff_to_a": None if not x["lp"] or len(x["lp"]) != len(ra["lp"])
                                         else round(max(abs(p - q) for p, q in zip(x["lp"], ra["lp"])), 5),
                                         "max_abs_diff_to_b": None if not x["lp"] or len(x["lp"]) != len(rb["lp"])
                                         else round(max(abs(p - q) for p, q in zip(x["lp"], rb["lp"])), 5)})
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.load_threads)]
        for t in threads:
            t.start()
        time.sleep(2)
        lsecs, lfailed, t0 = [], 0, time.time()
        for i in range(a.load_cycles):
            nxt = "A" if now == "B" else "B"
            u = srv.update_weights_from_disk(paths[nxt])
            if u["ok"]:
                now = nxt
                lsecs.append(u["seconds"])
            else:
                lfailed += 1
                if lfailed <= 3:
                    res.write(event="load_update_failed", cycle=i, **u)
            time.sleep(a.load_pause)
        stop.set()
        for t in threads:
            t.join(timeout=120)
        res.write(event="load", cycles=a.load_cycles, threads=a.load_threads, seconds=round(time.time() - t0, 1),
                  updates_ok=len(lsecs), updates_failed=lfailed, update_s=C.describe(lsecs),
                  answers=dict(counts), examples=examples, server_alive=srv.alive())
        # the server must still give the reference after all of it
        final_ok = equal(ask_all(srv.base_url, name), ref[now])
        res.write(event="after", serving=now, answers_match_reference=final_ok)
    except BaseException as e:  # noqa: BLE001
        res.fail("stress", e)
        raise
    finally:
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
