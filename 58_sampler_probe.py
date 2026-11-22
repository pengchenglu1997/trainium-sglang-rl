#!/usr/bin/env python3
"""SGLang-Neuron sampler modes for RL rollouts: does `topk_topp` take top_p / top_k, where does its top-p filter cut,
and what does it cost?

    (environment of srv_sglang_neuron.sh; SGL_SAMPLER and SGL_TOP_LOGPROBS are set by this script per server)
    python 58_sampler_probe.py --model DIR --train-jsonl FILE --results DIR --tag s1 --cores 12-15

For each sampler mode in --modes one server is started and stopped. Lines written:
  accept     HTTP status of one sampled request for each of several (top_p, top_k) settings
  next_dist  the first-token distribution of a fixed prompt (greedy request, top_logprobs)
  boundary   N one-token samples at temperature 1 with top_p set between the cumulative mass of the first j-1 and
             the first j tokens. A filter that keeps the token that crosses p (vLLM, standard SGLang) returns j
             distinct tokens; one that drops it returns j-1. Counts per token are written; the reading is left to
             the reader of the file.
  speed      three passes of Q groups of K answers, tokens per second, for full-vocabulary sampling and for
             top_p 0.95 / top_k 20
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402

PROMPT = "Name one colour of the rainbow. Answer with one word."


MORE_PROMPTS = [
    "Pick a number between 1 and 10. Answer with the number only.",
    "Flip a coin. Answer with one word: heads or tails.",
    "Name a fruit. Answer with one word.",
    "Pick one letter of the alphabet. Answer with the letter only.",
    "Name a day of the week. Answer with one word.",
    "Pick one: cat, dog, bird or fish. Answer with one word.",
]


def one_token(base_url, model, temperature, top_p, top_k, seed, top_logprobs=None, prompt=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt or PROMPT}], "max_tokens": 1,
            "temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": 0.0, "seed": seed,
            "logprobs": True, "chat_template_kwargs": {"enable_thinking": False}}
    if top_logprobs:
        body["top_logprobs"] = top_logprobs
    code, out, sec = rc.post(base_url.rstrip("/") + "/v1/chat/completions", body, timeout=300)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"status": code, "error": str(out)[:300]}
    ch = out["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content") or [])
    tok = content[0] if content else {}
    return {"status": 200, "token": tok.get("token"), "logprob": tok.get("logprob"),
            "top": [(t.get("token"), t.get("logprob")) for t in (tok.get("top_logprobs") or [])],
            "text": (ch.get("message") or {}).get("content")}


def speed(res, base_url, model, rows, k, passes, label, extra):
    for p in range(passes):
        def one(ir):
            i, r = ir
            return rc.chat(base_url, model, C.messages_for(r[1]), temperature=0.7, max_tokens=256, n=1,
                           seed=7000000 + p * 100000 + i, extra=extra, timeout=900)
        jobs = [(i, r) for i, r in enumerate(rows) for _ in range(k)]
        jobs = [(j, r) for j, (_, r) in enumerate(jobs)]
        t = time.time()
        with ThreadPoolExecutor(max_workers=32) as ex:
            outs = list(ex.map(one, jobs))
        sec = time.time() - t
        tok = sum(o["completion_tokens"] or 0 for o in outs)
        bad = [o for o in outs if o["status"] != 200]
        res.write(event="speed", sampling=label, n=p, requests=len(jobs), seconds=round(sec, 2),
                  completion_tokens=tok, tokens_per_s=round(tok / sec, 1), errors=len(bad),
                  first_error=(bad[0]["error"] if bad else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--modes", default="topk_topp,temperature")
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, default=30080)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--questions", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--no-speed", action="store_true")
    a = ap.parse_args()
    res = C.Results(a.results, f"58_sampler_{a.tag}")
    backend = server_ctl.load_backend(a.backends_json, a.backend)
    rows = C.load_gsm8k(a.train_jsonl, limit=2000)[-a.questions:]
    try:
        for mi, mode in enumerate(a.modes.split(",")):
            srv = server_ctl.Server(backend, a.port + mi, Path(a.results) / "server_logs", name=f"smp_{a.tag}_{mode}",
                                    env={"SGL_CORES": a.cores, "SGL_SAMPLER": mode,
                                         "SGL_TOP_LOGPROBS": str(a.top_logprobs)})
            try:
                st = srv.start(a.model)
                res.write(event="server_start", mode=mode, **st)
                if st["ready_s"] is None:
                    res.write(event="mode_skipped", mode=mode, reason="server did not become ready")
                    continue
                url, name = srv.base_url, srv.served_name()
                for tp, tk in ((1.0, -1), (0.95, -1), (1.0, 20), (0.95, 20), (0.5, 5)):
                    r = one_token(url, name, 0.7, tp, tk, seed=1)
                    res.write(event="accept", mode=mode, top_p=tp, top_k=tk, status=r["status"],
                              error=r.get("error"))
                g = one_token(url, name, 0.0, 1.0, -1, seed=1, top_logprobs=a.top_logprobs)
                if g["status"] != 200 or not g.get("top"):
                    res.write(event="next_dist", mode=mode, status=g["status"], error=g.get("error"),
                              note="no top_logprobs returned; boundary test left out")
                else:
                    top = sorted(((t, math.exp(lp)) for t, lp in g["top"] if lp is not None), key=lambda x: -x[1])
                    res.write(event="next_dist", mode=mode, top=[(t, round(p, 5)) for t, p in top[:10]])
                    cum, plans = 0.0, []
                    for j, (t, p) in enumerate(top[:6], start=1):
                        if j >= 2 and p >= 0.03:      # p in the middle of token j's mass; token j must be visible
                            plans.append((j, round(cum + p / 2.0, 6)))
                        cum += p
                    for j, tp in plans[:3]:
                        def draw(i, tp=tp):
                            return one_token(url, name, 1.0, tp, -1, seed=1000 + i)
                        with ThreadPoolExecutor(max_workers=16) as ex:
                            outs = list(ex.map(draw, range(a.samples)))
                        ok = [o for o in outs if o["status"] == 200]
                        counts = Counter(o["token"] for o in ok)
                        res.write(event="boundary", mode=mode, top_p=tp, crossing_token_rank=j,
                                  crossing_token=top[j - 1][0], crossing_token_prob=round(top[j - 1][1], 5),
                                  expected_distinct_if_kept=j, expected_distinct_if_dropped=j - 1,
                                  samples=len(ok), errors=len(outs) - len(ok),
                                  first_error=next((o.get("error") for o in outs if o["status"] != 200), None),
                                  distinct=len(counts), counts=counts.most_common(8))
                # the same boundary test on prompts with a flatter first-token distribution (added 2026-11-22: the
                # first prompt puts 0.87 on one token, so no plan was made)
                for q in MORE_PROMPTS:
                    g = one_token(url, name, 0.0, 1.0, -1, seed=1, top_logprobs=a.top_logprobs, prompt=q)
                    if g["status"] != 200 or not g.get("top"):
                        continue
                    top = sorted(((t, math.exp(lp)) for t, lp in g["top"] if lp is not None), key=lambda x: -x[1])
                    if len(top) < 3 or top[1][1] < 0.08:
                        res.write(event="next_dist", mode=mode, prompt=q, top=[(t, round(p, 4)) for t, p in top[:4]],
                                  note="second token below 0.08, not used")
                        continue
                    res.write(event="next_dist", mode=mode, prompt=q, top=[(t, round(p, 5)) for t, p in top[:8]])
                    cum = 0.0
                    for j, (t, p) in enumerate(top[:4], start=1):
                        if j >= 2 and p >= 0.05:
                            tp = round(cum + p / 2.0, 6)

                            def draw(i, tp=tp, q=q):
                                return one_token(url, name, 1.0, tp, -1, seed=2000 + i, prompt=q)
                            with ThreadPoolExecutor(max_workers=16) as ex:
                                outs = list(ex.map(draw, range(a.samples)))
                            ok = [o for o in outs if o["status"] == 200]
                            counts = Counter(o["token"] for o in ok)
                            res.write(event="boundary", mode=mode, prompt=q, top_p=tp, crossing_token_rank=j,
                                      crossing_token=t, crossing_token_prob=round(p, 5),
                                      expected_distinct_if_kept=j, expected_distinct_if_dropped=j - 1,
                                      samples=len(ok), errors=len(outs) - len(ok), distinct=len(counts),
                                      counts=counts.most_common(8))
                        cum += p
                if a.no_speed:
                    continue
                speed(res, url, name, rows, a.k, a.passes, "full vocabulary",
                      {"top_p": 1.0, "top_k": -1, "min_p": 0.0})
                if mode != "temperature":
                    speed(res, url, name, rows, a.k, a.passes, "top_p 0.95 top_k 20",
                          {"top_p": 0.95, "top_k": 20, "min_p": 0.0})
            finally:
                try:
                    srv.stop()
                except Exception:  # noqa: BLE001
                    pass
    except BaseException as e:  # noqa: BLE001
        res.fail("sampler", e)
        raise
    finally:
        res.done()


if __name__ == "__main__":
    main()
