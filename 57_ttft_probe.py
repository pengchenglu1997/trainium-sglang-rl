#!/usr/bin/env python3
"""Time to first token of rollout requests, and whether the K requests of one question share the
prompt's prefill (prefix cache).

    python 57_ttft_probe.py --base-url URL --model NAME --train-jsonl FILE --results DIR --tag ttft

Streaming chat requests, as the loop sends them (temperature 0.7, full vocabulary, thinking off):
  A. one request at a time, fresh questions: time to first token and total time (no queueing)
  B. K = 8 requests of the SAME question at once (one rollout group): time to first token of each;
     the reported cached prompt tokens of each (usage.prompt_tokens_details.cached_tokens, when the
     server reports it)
  C. the same group again (the prompt is now in the cache from B)
  D. 32 requests at once, 4 questions x 8 (one server's share of a step)
Standard library only."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402


def stream(base_url, model, question, seed, max_tokens=64):
    body = {"model": model, "messages": C.messages_for(question), "max_tokens": max_tokens, "temperature": 0.7,
            "top_p": 1.0, "top_k": -1, "min_p": 0.0, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except Exception:  # noqa: BLE001
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        n += 1
                        if first is None:
                            first = time.time() - t0
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    det = (usage or {}).get("prompt_tokens_details") or {}
    return {"ttft_s": None if first is None else round(first, 4), "total_s": round(time.time() - t0, 3), "chunks": n,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": det.get("cached_tokens")}


def summary(rs):
    ok = [r for r in rs if r.get("ttft_s") is not None]
    t = sorted(r["ttft_s"] for r in ok)
    return {"requests": len(rs), "ok": len(ok), "ttft_min": t[0] if t else None,
            "ttft_median": t[len(t) // 2] if t else None, "ttft_max": t[-1] if t else None,
            "total_max": max((r["total_s"] for r in ok), default=None),
            "prompt_tokens": sorted({r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None}),
            "cached_tokens": [r.get("cached_tokens") for r in ok],
            "first_error": next((r["error"] for r in rs if r.get("error")), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    res = C.Results(a.results, f"57_ttft_{a.tag}")
    try:
        qs = [r[1] for r in C.load_gsm8k(a.train_jsonl, limit=3000)][-40:]
        stream(a.base_url, a.model, qs[0], 1)                                    # warm the path
        res.write(event="ttft", case="A one at a time, fresh questions",
                  **summary([stream(a.base_url, a.model, q, 100 + i) for i, q in enumerate(qs[1:9])]))
        for case, q in (("B 8 at once, same fresh question", qs[10]), ("C the same group again", qs[10])):
            with ThreadPoolExecutor(max_workers=8) as ex:
                rs = list(ex.map(lambda i: stream(a.base_url, a.model, q, 200 + i), range(8)))
            res.write(event="ttft", case=case, **summary(rs))
        jobs = [(q, 300 + 8 * j + i) for j, q in enumerate(qs[12:16]) for i in range(8)]
        with ThreadPoolExecutor(max_workers=32) as ex:
            rs = list(ex.map(lambda x: stream(a.base_url, a.model, x[0], x[1], 256), jobs))
        res.write(event="ttft", case="D 32 at once, 4 fresh questions x 8, 256 tokens", **summary(rs))
    except BaseException as e:  # noqa: BLE001
        res.fail("ttft", e)
        raise
    finally:
        res.done()


if __name__ == "__main__":
    main()
