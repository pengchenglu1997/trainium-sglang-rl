#!/usr/bin/env python3
"""What does an in-place weight update do to requests that are running? (plan row W1.5)

    (environment of srv_sglang_neuron.sh; MODELS_ROOT and SGL_CACHE set; patched server)
    python 68_update_under_load.py --base "$MODELS_ROOT/Qwen3-0.6B" --alt "$MODELS_ROOT/Qwen3-0.6B-noise" \
        --results DIR --tag w15 --cores 0-3 [--threads 8] [--updates 4] [--min-tokens 2000]
    (--base-url URL --model NAME tests a server that is already running; the paths are then sent as given)

Reference, with the server on --base (made by mk_alt.py when missing, the same perturbed copy 48 uses):
  ref_base   --threads greedy requests at once, one prompt each, long answers (the prompt asks for a story
             of 3000 words), token logprobs on, max_tokens --max-tokens;
  ref_base2  the same again: if it differs from ref_base the server is not repeatable at this load, and
             "neither" below can come from that and not from the update (written as "control");
  one update to --alt, then ref_alt: the same requests on --alt.
Load: each of --threads threads keeps one such request running (a new one as soon as one ends) while the
main thread makes --updates updates, alternating --base and --alt, --pause seconds apart, the first after
--warmup seconds. Written per update: seconds, HTTP status, requests running at the call and at the return
(counted by this script). Written per request: status, completion tokens, seconds, start and end time, the
weights this script believed live at start and end, the updates it overlapped, and its class:
  base / alt   every token and logprob (within --tol) equals the reference of that checkpoint and the length
               equals it too (token strings and logprobs; the API gives no token ids)
  neither      anything else; stops_matching_base_at and stops_matching_alt_at are the first positions
               that differ (49_sync_stress.py classes whole answers by their logprob vector; this adds
               the position of the first difference); after_base_stop_matches_alt is
               the share of later positions equal to the alt reference (a splice of two weight sets shows
               as a high share, a different continuation after a changed token as about zero)
  http_error   status other than 200
An answer that ends before --min-tokens is reported (requests_below_min_tokens, and "reference_short"),
not repaired; --ignore-eos sends ignore_eos with the request (SGLang's chat request has the field;
whether the Neuron sampler honours it was not tested).
Expected by the plan: the update waits for or pauses the running requests, no HTTP error, and every answer
is class base or alt. A "neither" answer after a switch is the finding: the update is not fenced from
requests (or cut them). The script cannot tell a splice from a request restarted on the new weights;
the overlap fields and the positions are there for that reading."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402

TOPICS = ["a lighthouse keeper in winter", "a bakery that opens only at night", "two cousins crossing a desert",
          "a small town that loses its river", "a clockmaker and a stubborn apprentice",
          "a ferry captain on a foggy lake", "a school choir before a big concert",
          "a mountain village waiting for the first snow"]


def prompt_for(i):
    return (f"Write a very long, detailed story about {TOPICS[i % len(TOPICS)]}. Write at least 3000 words in "
            "numbered paragraphs and do not stop or summarise early.")


def generate(url, model, prompt, max_tokens, ignore_eos, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "logprobs": True, "chat_template_kwargs": rc.THINK_OFF}
    if ignore_eos:
        body["ignore_eos"] = True
    code, out, sec = rc.post(url.rstrip("/") + "/v1/chat/completions", body, timeout=timeout)
    r = {"status": code, "seconds": round(sec, 2), "toks": [], "lps": [], "completion_tokens": None, "error": None}
    if code == 200 and isinstance(out, dict) and out.get("choices"):
        ch = out["choices"][0]
        content = (ch.get("logprobs") or {}).get("content") or []
        r["toks"] = [t.get("token") for t in content]
        r["lps"] = [t.get("logprob") for t in content]
        r["completion_tokens"] = (out.get("usage") or {}).get("completion_tokens")
    else:
        r["error"] = (out if isinstance(out, str) else str(out))[:300]
    return r


def close(x, y, tol):
    return x is not None and y is not None and abs(x - y) <= tol


def match_len(r, ref, tol):
    n = 0
    for t, p, rt, rp in zip(r["toks"], r["lps"], ref["toks"], ref["lps"]):
        if t != rt or not close(p, rp, tol):
            break
        n += 1
    return n


def classify(r, rb, ra, tol):
    """(class, detail) of one request against the two references of its prompt."""
    if r["status"] != 200:
        return "http_error", {"error": r["error"]}
    n = len(r["toks"])
    mb, ma = match_len(r, rb, tol), match_len(r, ra, tol)
    full_b = mb == n == len(rb["toks"])
    full_a = ma == n == len(ra["toks"])
    if full_b and not full_a:
        return "base", {}
    if full_a and not full_b:
        return "alt", {}
    if full_a and full_b:
        return "both", {}
    later = list(range(mb, n))
    share = None
    if later:
        hit = sum(1 for i in later if i < len(ra["toks"]) and r["toks"][i] == ra["toks"][i]
                  and close(r["lps"][i], ra["lps"][i], tol))
        share = round(hit / len(later), 3)
    return "neither", {"stops_matching_base_at": mb, "stops_matching_alt_at": ma,
                       "after_base_stop_matches_alt": share}


def wave(url, model, a, n):
    """n requests at once, one prompt each; returns the list of results in prompt order."""
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(lambda i: generate(url, model, prompt_for(i), a.max_tokens, a.ignore_eos,
                                              a.request_timeout), range(n)))


def brief(rs):
    return [{"status": r["status"], "tokens": len(r["toks"]), "completion_tokens": r["completion_tokens"],
             "seconds": r["seconds"], "error": r["error"]} for r in rs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="", help="served name; only with --base-url")
    ap.add_argument("--base-url", default="", help="use this running server instead of starting one")
    ap.add_argument("--cores", default="0-3")
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--extra", default="", help="more launch flags, after SGL_EXTRA of the environment")
    ap.add_argument("--patch-dir", default="", help="SGL_PATCH_DIR of the server (unset = the one of the environment)")
    ap.add_argument("--env", default="", help="SGL_ENV of the server, K=V pairs")
    ap.add_argument("--port", type=int, default=30068)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--min-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--pause", type=float, default=15.0, help="seconds between updates")
    ap.add_argument("--warmup", type=float, default=5.0, help="seconds of load before the first update")
    ap.add_argument("--tol", type=float, default=1e-3, help="largest logprob difference counted as equal")
    ap.add_argument("--request-timeout", type=float, default=3600)
    ap.add_argument("--skip-control", action="store_true", help="leave out the repeat of ref_base")
    ap.add_argument("--noise", type=float, default=0.01)
    a = ap.parse_args()
    if a.max_tokens < a.min_tokens:
        sys.exit("--max-tokens must be at least --min-tokens")
    res = C.Results(a.results, f"68_update_under_load_{a.tag}")
    srv = None
    try:
        base, alt = Path(a.base), Path(a.alt)
        if not any(alt.glob("*.safetensors")):
            t = time.time()
            subprocess.run([sys.executable, str(Path(__file__).with_name("mk_alt.py")), "--base", str(base),
                            "--alt", str(alt), "--noise", str(a.noise)], check=True)
            res.write(event="alt_made", seconds=round(time.time() - t, 1), noise=a.noise)
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
            # the container sees only MODELS_ROOT (at /models): an update to a path outside it would be refused
            # only after the reference waves
            root = os.environ.get("MODELS_ROOT", "").rstrip("/")
            outside = [str(p) for p in (base, alt) if not (root and str(p).startswith(root + "/"))]
            if outside:
                raise RuntimeError(f"--base and --alt must lie under MODELS_ROOT ({root or 'unset'}): {outside}")
            if a.env:
                env["SGL_ENV"] = a.env
            srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                                    Path(a.results) / "server_logs", name=f"load_{a.tag}", env=env)
            st = srv.start(str(base))
            res.write(event="server_start", extra=extra, patch_dir=a.patch_dir or os.environ.get("SGL_PATCH_DIR", ""),
                      env=a.env, **st)
            if st["ready_s"] is None:
                raise RuntimeError("server did not become ready")
            url, name = srv.base_url, srv.served_name()

        def do_update(path):
            if srv is not None:
                return srv.update_weights_from_disk(str(path))
            code, out, sec = rc.post(url.rstrip("/") + "/update_weights_from_disk", {"model_path": str(path)},
                                     timeout=1800)
            ok = code == 200 and (not isinstance(out, dict) or out.get("success", True))
            return {"status": code, "ok": bool(ok), "seconds": round(sec, 2), "body": str(out)[:300]}

        n = a.threads
        # ---- references --------------------------------------------------------------------------
        ref_b = wave(url, name, a, n)
        res.write(event="ref_base", requests=brief(ref_b))
        short = [i for i, r in enumerate(ref_b) if len(r["toks"]) < a.min_tokens]
        if short or any(r["status"] != 200 for r in ref_b):
            res.write(event="reference_short", prompts_below_min_tokens=short, min_tokens=a.min_tokens,
                      tokens=[len(r["toks"]) for r in ref_b],
                      note="answers end before --min-tokens; the load is shorter than asked")
        if any(r["status"] != 200 for r in ref_b):
            raise RuntimeError("a reference request failed: " + str(next(r["error"] for r in ref_b if r["status"] != 200)))
        if not a.skip_control:
            ref_b2 = wave(url, name, a, n)
            same = [match_len(x, y, a.tol) == len(x["toks"]) == len(y["toks"]) and x["status"] == y["status"] == 200
                    for x, y in zip(ref_b2, ref_b)]
            res.write(event="control", base_repeat_identical=sum(same), of=n, requests=brief(ref_b2),
                      note="if below of, 'neither' answers may come from batch-dependent numbers, not the update")
        u0 = do_update(alt)
        res.write(event="ref_update", to="alt", **u0)
        if not u0["ok"]:
            raise RuntimeError(f"first update refused: {u0}")
        ref_a = wave(url, name, a, n)
        res.write(event="ref_alt", requests=brief(ref_a))
        if any(r["status"] != 200 for r in ref_a):
            raise RuntimeError("a reference request on alt failed")
        differ = [not (match_len(x, y, a.tol) == len(x["toks"]) == len(y["toks"])) for x, y in zip(ref_a, ref_b)]
        res.write(event="references", alt_differs_from_base=sum(differ), of=n)
        if not any(differ):
            raise RuntimeError("alt and base give identical answers: nothing to tell apart")

        # ---- load with updates ---------------------------------------------------------------------
        lock = threading.Lock()
        running = [0]
        cur = ["alt"]                # weights believed live; "->base" while an update to base is in the call
        stop = threading.Event()
        t0 = time.time()
        reqs = []

        def worker(w):
            while not stop.is_set():
                with lock:
                    running[0] += 1
                    t_start, w_start = time.time() - t0, cur[0]
                r = generate(url, name, prompt_for(w), a.max_tokens, a.ignore_eos, a.request_timeout)
                with lock:
                    running[0] -= 1
                    t_end, w_end = time.time() - t0, cur[0]
                    reqs.append({"thread": w, "prompt": w % n, "r": r, "t_start": round(t_start, 2),
                                 "t_end": round(t_end, 2), "weights_at_start": w_start, "weights_at_end": w_end})

        threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n)]
        for t in threads:
            t.start()
        time.sleep(a.warmup)
        ups = []
        for i in range(a.updates):
            to = "base" if cur[0] == "alt" else "alt"
            with lock:
                at_call, tc = running[0], time.time() - t0
                cur[0] = "->" + to
            u = do_update(base if to == "base" else alt)
            with lock:
                at_ret, tr = running[0], time.time() - t0
                cur[0] = to if u["ok"] else ("alt" if to == "base" else "base")
            ups.append({"i": i, "to": to, "t_call": round(tc, 2), "t_return": round(tr, 2),
                        "running_at_call": at_call, "running_at_return": at_ret, "status": u["status"],
                        "ok": u["ok"], "seconds": u["seconds"], "body": u["body"][:120]})
            res.write(event="update", **ups[-1])
            if i + 1 < a.updates:
                time.sleep(a.pause)
        stop.set()
        for t in threads:
            t.join(timeout=a.request_timeout + 60)

        # ---- classify ------------------------------------------------------------------------------
        counts = {"base": 0, "alt": 0, "both": 0, "neither": 0, "http_error": 0}
        overlap_counts = {"base": 0, "alt": 0, "both": 0, "neither": 0, "http_error": 0}
        n_overlap = n_short = 0
        for q in sorted(reqs, key=lambda x: x["t_start"]):
            r = q["r"]
            cls, det = classify(r, ref_b[q["prompt"]], ref_a[q["prompt"]], a.tol)
            ov = [u["i"] for u in ups if q["t_start"] < u["t_return"] and u["t_call"] < q["t_end"]]
            short_ = r["status"] == 200 and len(r["toks"]) < a.min_tokens
            counts[cls] += 1
            if ov:
                n_overlap += 1
                overlap_counts[cls] += 1
            n_short += bool(short_)
            res.write(event="request", thread=q["thread"], prompt=q["prompt"], status=r["status"],
                      tokens=len(r["toks"]), completion_tokens=r["completion_tokens"], seconds=r["seconds"],
                      t_start=q["t_start"], t_end=q["t_end"], weights_at_start=q["weights_at_start"],
                      weights_at_end=q["weights_at_end"], overlapped_updates=ov, **{"class": cls},
                      below_min_tokens=short_, **det)
        bad_updates = [u for u in ups if not u["ok"]]
        res.write(event="summary", threads=n, updates=len(ups), updates_failed=len(bad_updates),
                  update_http_statuses=sorted({u["status"] for u in ups}),
                  update_seconds=C.describe([u["seconds"] for u in ups]),
                  updates_with_requests_running_at_call=sum(u["running_at_call"] > 0 for u in ups),
                  requests=len(reqs), classes=counts, requests_overlapping_an_update=n_overlap,
                  classes_of_overlapping_requests=overlap_counts, requests_below_min_tokens=n_short,
                  consistent_with_one_weight_set=counts["neither"] == 0 and counts["both"] == 0,
                  http_errors=counts["http_error"], any_update_failed=bool(bad_updates))
        # the server must still give the reference of the weights it ended on
        last = ref_b if cur[0] == "base" else ref_a
        fin = wave(url, name, a, n)
        ok_final = sum(f["status"] == 200 and match_len(f, g, a.tol) == len(f["toks"]) == len(g["toks"])
                       for f, g in zip(fin, last))
        res.write(event="after", serving=cur[0], answers_match_reference=ok_final, of=n)
    except BaseException as e:  # noqa: BLE001
        res.fail("update_under_load", e)
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
