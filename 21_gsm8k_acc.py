#!/usr/bin/env python3
"""GSM8K accuracy of a running server, one line per question (index, parsed number, right or
wrong; no text), so two engines can be compared question by question.

    python 21_gsm8k_acc.py --base-url URL --model NAME --data gsm8k_test.jsonl --results DIR \
        --tag sgl_tp4 [--limit 0] [--temperature 0] [--passes 1] [--concurrency 32]
    python 21_gsm8k_acc.py compare --a FILE --b FILE        (paired: both right, only a, only b)

Settings follow an earlier inference comparison: greedy, at most 512 new
tokens, 32 requests at once, thinking off; sampled passes at temperature 0.6 when asked. The
prompt is this kit's own (rl_common.messages_for), so numbers are comparable between engines
run with this script, not with other scripts' numbers."""
from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402


def run(a):
    res = C.Results(a.results, f"21_gsm8k_{a.tag}")
    try:
        rows = C.load_gsm8k(a.data, limit=a.limit or None)
        res.write(event="start", tag=a.tag, n=len(rows), temperature=a.temperature,
                  passes=a.passes, max_tokens=a.max_tokens, concurrency=a.concurrency)
        for p in range(a.passes):
            t0 = time.time()

            def one(r, p=p):
                x = rc.chat(a.base_url, a.model, C.messages_for(r[1]), a.temperature,
                            a.max_tokens, seed=(1000 * (p + 1) + r[0]) if a.temperature else None,
                            timeout=a.timeout)
                ok = x["status"] == 200 and bool(x["texts"])
                text = x["texts"][0] if ok else None
                return {"event": "answer", "pass": p, "index": r[0], "status": x["status"],
                        "pred": C.extract_answer(text) if ok else None,
                        "correct": bool(ok and C.is_correct(text, r[2])),
                        "completion_tokens": x["completion_tokens"],
                        "hit_token_limit": (x["completion_tokens"] or 0) >= a.max_tokens,
                        "seconds": round(x["seconds"], 2)}
            with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
                outs = list(ex.map(one, rows))
            for o in outs:
                res.write(**o)
            n = len(outs)
            res.write(event="pass", **{"pass": p}, n=n, errors=sum(o["status"] != 200 for o in outs),
                      accuracy=round(sum(o["correct"] for o in outs) / n, 4),
                      no_number=sum(o["pred"] is None and o["status"] == 200 for o in outs),
                      hit_token_limit=sum(o["hit_token_limit"] for o in outs),
                      seconds=round(time.time() - t0, 1))
    except BaseException as e:  # noqa: BLE001
        res.fail("gsm8k", e)
        raise
    finally:
        res.done()


def compare(a):
    def load(p):
        return {(r["pass"], r["index"]): r["correct"] for r in C.read_jsonl(p)
                if r.get("event") == "answer" and r["status"] == 200}
    x, y = load(a.a), load(a.b)
    keys = sorted(set(x) & set(y))
    both = sum(x[k] and y[k] for k in keys)
    only_a = sum(x[k] and not y[k] for k in keys)
    only_b = sum(y[k] and not x[k] for k in keys)
    n = len(keys)
    k, m = min(only_a, only_b), only_a + only_b
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    print(f"paired questions {n} | both right {both} | only a {only_a} | only b {only_b} | "
          f"accuracy a {(both + only_a) / n:.4f} b {(both + only_b) / n:.4f} | "
          f"exact McNemar p {p:.3f}" if n else "no paired questions")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--results")
    ap.add_argument("--tag")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    if a.cmd == "compare":
        return compare(a)
    for k in ("base_url", "model", "data", "results", "tag"):
        if not getattr(a, k):
            ap.error(f"--{k.replace('_', '-')} is required")
    run(a)


if __name__ == "__main__":
    main()
