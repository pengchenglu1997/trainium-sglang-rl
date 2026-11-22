#!/usr/bin/env python3
"""Does the server still use text that lies past token 8K? (plan row W1.3)

    (environment of srv_sglang_neuron.sh; MODELS_ROOT and SGL_CACHE set)
    SGL_PATCH_DIR=<kit>/patches/sglang_neuron_v0.1.0_inplace_pr4 \
    python 67_long_context_needle.py --model "$MODELS_ROOT/Qwen3-0.6B" --results DIR --tag pr4 \
        --cores 0-3 --context-length 16384 [--extra "--max-running-requests 4 ..."]
    (the same command with --patch-dir <kit>/patches/sglang_neuron_v0.1.0_inplace, or no patch, is the
     "without PR 4" run; --base-url URL --model NAME tests a server that is already running)

One server. A "needle" sentence with a random 6-digit code ("The secret access code for the vault is
NNNNNN.") is put into filler text, and the model is asked for the code. Greedy, 5 needles per condition,
each with its own seeded filler and code. The filler is generated sentences (seeded, no download) that hold
only numbers of one or two digits, so no other 6-digit number is in the prompt.
Token positions are controlled from usage.prompt_tokens, since this machine has no tokenizer: two short
calibration requests give tokens per filler sentence; then a probe request (max_tokens 1) measures the
prompt up to the needle (case "depth") or the whole prompt (case "late") and the sentence count is
corrected until it is within 1 percent of the target or --fit-rounds rounds are used.

Cases:
  depth  the needle sits at depth D tokens (default 1000,4000,7000,9000,12000,15000) and the question
         follows after --tail-sentences more sentences. Prompt length is D plus about 200 tokens.
  late   the needle is early (--early-depth, default 500 tokens) and the question comes after filler up to
         a total prompt length L (default 9000,12000,15000). Every decode step then runs with a prior
         longer than 8K while the fact it needs lies in the first 500 tokens.
A condition whose prompt plus --max-tokens does not fit --context-length is written as a "skipped" line.
Written per condition: found (the code occurs in the answer) out of n, the head of each answer, the
measured depth, the prompt tokens of each request (usage), HTTP errors. The last line "summary" gives the
rate found for conditions with prompts up to --boundary tokens (default 8192) and above it, and the
first condition that lost a needle.
Expected by the plan: without PR 4 the needle is lost above 8K; with PR 4 it is found. Not measured on a
dense model before; a result either way is the answer. found = the code is in the text of the answer, so
an answer that quotes the question's wording but gives another code is not found."""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402

SUBJECTS = ["The warehouse on Elm Street", "A carpenter named Joan", "The northern ferry", "Our small bakery",
            "The village library", "A retired pilot", "The night shift", "The river museum", "A young farmer",
            "The harbour office", "The chemistry club", "A travelling tailor"]
VERBS = ["received", "counted", "repaired", "painted", "sold", "moved", "checked", "stored", "packed", "borrowed"]
OBJECTS = ["crates of apples", "wooden chairs", "blue notebooks", "bags of flour", "old maps", "tin lamps",
           "boxes of nails", "woollen scarves", "glass bottles", "paper boats"]
WHENS = ["on Monday", "on Tuesday", "last week", "before noon", "in the spring", "after the storm",
         "during the fair", "late in the evening", "at the start of June", "twice that month"]
NEEDLE = "The secret access code for the vault is {code}."
QUESTION = ("\n\nQuestion: What is the secret access code for the vault? "
            "Answer with the 6-digit number only.")
HEAD = "Read the following notes carefully. A code is hidden somewhere in them.\n\n"


def sentences(rng, n):
    out = []
    for _ in range(n):
        out.append(f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.randint(2, 99)} {rng.choice(OBJECTS)} "
                   f"{rng.choice(WHENS)}, and {rng.randint(2, 99)} more {rng.choice(WHENS)}.")
    return out


def prefix_text(seed, n_before):
    return HEAD + " ".join(sentences(random.Random(seed), n_before))


def full_text(seed, n_before, n_after):
    """Needle after n_before sentences, n_after sentences behind it, then the question. Deterministic in
    (seed, n_before, n_after); the code is drawn from the seed alone."""
    rng = random.Random(seed)
    code = f"{random.Random(seed * 7919 + 13).randint(100000, 999999)}"
    before = " ".join(sentences(rng, n_before))
    after = " ".join(sentences(rng, n_after))
    return HEAD + before + " " + NEEDLE.format(code=code) + " " + after + QUESTION, code


def request(base_url, model, text, max_tokens, timeout=1800):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": rc.THINK_OFF}
    code, out, sec = rc.post(base_url.rstrip("/") + "/v1/chat/completions", body, timeout=timeout)
    r = {"status": code, "seconds": round(sec, 2), "text": None, "prompt_tokens": None, "error": None}
    if code == 200 and isinstance(out, dict) and out.get("choices"):
        r["text"] = (out["choices"][0].get("message") or {}).get("content") or ""
        r["prompt_tokens"] = (out.get("usage") or {}).get("prompt_tokens")
    else:
        r["error"] = (out if isinstance(out, str) else str(out))[:300]
    return r


def tokens_of(base_url, model, text):
    r = request(base_url, model, text, 1)
    if r["status"] != 200 or not isinstance(r["prompt_tokens"], int):
        raise RuntimeError(f"probe request failed: status {r['status']} {r['error']}")
    return r["prompt_tokens"]


def calibrate(base_url, model, small, large):
    """(tokens per sentence, tokens of the empty frame) from two prompts of different length."""
    t1 = tokens_of(base_url, model, prefix_text(1, small))
    t2 = tokens_of(base_url, model, prefix_text(1, large))
    slope = (t2 - t1) / float(large - small)
    if slope <= 0:
        raise RuntimeError(f"prompt_tokens did not grow with the prompt: {t1} for {small}, {t2} for {large}")
    return slope, t1 - slope * small


def fit(measure, target, slope, intercept, rounds, offset=0):
    """Sentence count n with measure(n) close to target tokens. Returns (n, measured, rounds used).
    offset: sentences that measure() adds on its own (the sentences before the needle)."""
    n = max(0, int(round((target - intercept) / slope)) - offset)
    got = measure(n)
    used = 0
    while abs(got - target) > max(0.01 * target, slope) and used < rounds:
        n = max(0, n + int(round((target - got) / slope)))
        got = measure(n)
        used += 1
    return n, got, used


def run_condition(a, url, name, case, target, slope, intercept, res):
    seed0 = 100000 * (1 if case == "depth" else 2) + target
    if case == "depth":
        n_before, got, used = fit(lambda n: tokens_of(url, name, prefix_text(seed0, n)), target, slope, intercept,
                                  a.fit_rounds)
        n_after = a.tail_sentences
        depth_measured = got
    else:
        n_before = max(0, int(round((a.early_depth - intercept) / slope)))
        depth_measured = None
        n_after, got, used = fit(lambda n: tokens_of(url, name, full_text(seed0, n_before, n)[0]), target, slope,
                                 intercept, a.fit_rounds, offset=n_before)
    rows = []
    for i in range(a.needles):
        text, code = full_text(seed0 + i, n_before, n_after)
        r = request(url, name, text, a.max_tokens)
        found = r["status"] == 200 and code in (r["text"] or "")
        rows.append({"i": i, "found": bool(found), "status": r["status"], "prompt_tokens": r["prompt_tokens"],
                     "seconds": r["seconds"], "code": code, "head": (r["text"] or "")[:60], "error": r["error"]})
    toks = [r["prompt_tokens"] for r in rows if r["prompt_tokens"] is not None]
    total = round(sum(toks) / len(toks)) if toks else None
    ev = {"event": "condition", "case": case, "target_tokens": target, "n_before": n_before, "n_after": n_after,
          "fit_rounds_used": used, "needle_depth_measured": depth_measured if case == "depth" else None,
          "prompt_tokens": toks, "prompt_tokens_mean": total, "needles": len(rows),
          "found": sum(r["found"] for r in rows), "http_errors": sum(r["status"] != 200 for r in rows),
          "answers": [{"found": r["found"], "head": r["head"], "status": r["status"], "error": r["error"]}
                      for r in rows]}
    res.write(**ev)
    return ev


def summarize(res, conds, boundary):
    def rate(sel):
        n = sum(c["needles"] for c in sel)
        return {"conditions": len(sel), "needles": n, "found": sum(c["found"] for c in sel),
                "rate": round(sum(c["found"] for c in sel) / n, 3) if n else None}
    out = {}
    for case in ("depth", "late"):
        sel = [c for c in conds if c["case"] == case and c["prompt_tokens_mean"] is not None]
        out[case] = {"prompt_up_to_boundary": rate([c for c in sel if c["prompt_tokens_mean"] <= boundary]),
                     "prompt_above_boundary": rate([c for c in sel if c["prompt_tokens_mean"] > boundary])}
        lost = [c for c in sel if c["found"] < c["needles"]]
        lost.sort(key=lambda c: c["prompt_tokens_mean"])
        out[case]["first_condition_with_loss"] = (
            {"target_tokens": lost[0]["target_tokens"], "prompt_tokens_mean": lost[0]["prompt_tokens_mean"],
             "found": lost[0]["found"], "needles": lost[0]["needles"]} if lost else None)
    res.write(event="summary", boundary=boundary, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir under MODELS_ROOT, or the served name with --base-url")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-url", default="", help="use this running server instead of starting one")
    ap.add_argument("--cores", default="0-3")
    ap.add_argument("--context-length", type=int, default=16384)
    ap.add_argument("--extra", default="", help="more launch flags, after SGL_EXTRA of the environment")
    ap.add_argument("--patch-dir", default="", help="SGL_PATCH_DIR of the server (unset = the one of the environment)")
    ap.add_argument("--env", default="", help="SGL_ENV of the server, K=V pairs")
    ap.add_argument("--port", type=int, default=30067)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--depths", default="1000,4000,7000,9000,12000,15000")
    ap.add_argument("--late-lengths", default="9000,12000,15000")
    ap.add_argument("--early-depth", type=int, default=500)
    ap.add_argument("--tail-sentences", type=int, default=8)
    ap.add_argument("--needles", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--boundary", type=int, default=8192)
    ap.add_argument("--fit-rounds", type=int, default=2)
    ap.add_argument("--cal-small", type=int, default=20)
    ap.add_argument("--cal-large", type=int, default=80)
    a = ap.parse_args()
    res = C.Results(a.results, f"67_long_context_{a.tag}")
    srv = None
    try:
        if a.base_url:
            url, name = a.base_url, a.model
            res.write(event="server", base_url=url, started=False)
        else:
            extra = re.sub(r"--context-length\s+\d+", "", os.environ.get("SGL_EXTRA", ""))
            extra = f"{extra} --context-length {a.context_length} {a.extra}".strip()
            env = {"SGL_CORES": a.cores, "SGL_EXTRA": extra}
            if a.patch_dir:
                # docker -v needs an absolute host path, and the launcher runs from the kit folder
                pd = Path(a.patch_dir).resolve()
                if not pd.is_dir():
                    raise RuntimeError(f"--patch-dir is not a folder: {pd}")
                a.patch_dir = str(pd)
                env["SGL_PATCH_DIR"] = a.patch_dir
            if a.env:
                env["SGL_ENV"] = a.env
            srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                                    Path(a.results) / "server_logs", name=f"needle_{a.tag}", env=env)
            st = srv.start(a.model)
            res.write(event="server_start", extra=extra, patch_dir=a.patch_dir or os.environ.get("SGL_PATCH_DIR", ""),
                      env=a.env, **st)
            if st["ready_s"] is None:
                raise RuntimeError("server did not become ready")
            url, name = srv.base_url, srv.served_name()
        slope, intercept = calibrate(url, name, a.cal_small, a.cal_large)
        res.write(event="calibration", tokens_per_sentence=round(slope, 3), frame_tokens=round(intercept, 1),
                  sentences=[a.cal_small, a.cal_large])
        conds = []
        plan = [("depth", int(x)) for x in a.depths.split(",") if x] + \
               [("late", int(x)) for x in a.late_lengths.split(",") if x]
        for case, target in plan:
            need = target + (a.tail_sentences * slope + 60 if case == "depth" else 60) + a.max_tokens
            if need > a.context_length:
                res.write(event="skipped", case=case, target_tokens=target, need_tokens=round(need),
                          context_length=a.context_length)
                continue
            conds.append(run_condition(a, url, name, case, target, slope, intercept, res))
        summarize(res, conds, a.boundary)
    except BaseException as e:  # noqa: BLE001
        res.fail("needle", e)
        raise
    finally:
        if srv is not None:
            try:
                srv.stop()
            except Exception:  # noqa: BLE001
                pass
        res.done()


if __name__ == "__main__":
    main()
