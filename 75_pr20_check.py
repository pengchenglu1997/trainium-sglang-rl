#!/usr/bin/env python3
"""Does upstream PR 20's in-place reload serve what a fresh start serves? (run R2, ROADMAP B1a) With a short smoke.

    (environment of srv_sglang_neuron.sh: MODELS_ROOT, SGL_CACHE, SGL_CORES, SGL_SCRIPT, SGL_EXTRA, SGL_ENV, and
     SGL_PATCH_DIR = the folder that holds the three files of upstream-pr/20)
    python 75_pr20_check.py --base "$MODELS_ROOT/Qwen3-0.6B" --alt "$MODELS_ROOT/Qwen3-0.6B_alt" --results DIR \
        --tag pr20_q06 --manifest-json DIR/patch_manifest_pr20.json [--skip-fresh]

What PR 20 does (read in the head of upstream pull request 20, read-only, 24 Nov 2026; the PR is two
commits, 40705ac and ecee5f3, over 4e24326; files general_plugin.py, models/host.py, models/transformer/model.py):
  * general_plugin.py hooks (AROUND) WeightUpdater.update_weights_from_disk of the SGLang under it. A model whose
    class sets reloads_weights_in_place gets model.reload_weights(model_path); then model_config.model_path is set
    and the hook returns (True, "Succeeded to update model weights."). Any other model, or any exception, returns
    (False, message) and the HTTP answer is then "success": false (the exception text is in "message").
    The refusal of TokenizerManager.update_weights_from_disk is removed; the tensor, distributed and ipc updates are
    still refused.
  * model.py (transformer family): reload_weights() reads this rank's shards on the CPU, checks every tensor name and
    shape BEFORE the first copy (ecee5f3), then copies into the device parameters the compiled graphs hold. No graph
    is rebuilt.
  * The diff holds no cache flush, no pause and no abort of running requests. Whether SGLang's scheduler flushes the
    radix cache after a successful disk update, and whether the update waits for running requests, is in SGLang and
    was not read here (UNVERIFIED); the server reports no cached-token count, so the radix check below is indirect.

Steps (every one a JSON line in <results>/75_pr20_check_<tag>.jsonl; the first line carries SGL_PATCH_DIR, the other
SGL_* values and the content of --manifest-json, which patch_manifest.py writes):
  A  start on --base; --prompts fixed greedy prompts with token logprobs; the first prompt once more (a repeat that
     may be served from the radix cache)
  B  POST /update_weights_from_disk to --alt (timed); the same prompts
  C  update back to --base (timed); the same prompts; check C == A
  smoke (server of step C, base weights; "smoke only, not a load test"):
     --smoke-updates updates in a row, alternating alt and base, each timed and each ok; then the prompts must give
     the answers of the weights the server ends on; then one update to the other weights while --smoke-threads short
     requests (max 64 new tokens) are in flight: HTTP errors are counted, and after it the prompts must give the
     answers of the new weights
  D  (unless --skip-fresh) restart on --alt; the same prompts; check B == D
"==" is: same text on every prompt, same number of tokens, and the largest absolute token-logprob difference is
exactly 0.0 (the stop rule of the run list). The difference is written unrounded.
Indirect radix-cache check: the first prompt is sent twice before the update and once more right after it (B); that
answer must equal the first prompt's answer of the fresh start D (and must differ from A's). The server reports no
cached-token count, so a stale cache entry shows only as a difference; this is written in the result line.

The repeat of the first prompt on the base weights is SOFT: if it differs (a radix-cache hit may run through another
graph on Neuron), a line {"status": "warn", "stage": "radix_repeat"} is written, the summary lists it under "warnings",
the PASSED line says "WITH WARNING radix_repeat", and the run goes on to B, C, D; the exit code is not changed by it.
The hard checks stay C == A, B == D and the indirect cache check after the update.
A failed check writes a line with "status": "fail". Steps A to C stop at their first failure (stop rule); failures of
the smoke are recorded and the restart D still runs (it is a new server). The server is stopped at the end in every
case, the last line of the file is {"done": true}, and the exit code is 1 if any line says fail. The last line printed
is "75_pr20_check PASSED ..." or "75_pr20_check FAILED ...".
Standard library only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402

# The first three are the prompts of 48_inplace_server_test.py; the others are written for this file, in the style of
# GSM8K word problems (their answers are not used: only the equality of two servers' answers is tested).
PROMPTS = [
    "A farmer has 17 rows of 23 apple trees. He cuts down 41 trees. How many trees are left?",
    "Tom reads 12 pages a day for 9 days and then 30 pages more. How many pages did he read?",
    "A box holds 24 pencils. A school buys 15 boxes and gives away 80 pencils. How many remain?",
    "Maria buys 8 packs of stickers with 14 stickers in each pack. She gives 35 stickers to her friends. "
    "How many stickers does she have left?",
    "A bakery makes 36 muffins every hour for 7 hours. It sells 190 of them. How many muffins are not sold?",
    "Sam saves 25 dollars each week for 12 weeks and then spends 140 dollars on a bicycle helmet. "
    "How much money does he have left?",
    "A train has 9 cars with 64 seats each. On one trip 418 seats are taken. How many seats are empty?",
    "Lena runs 3 miles on each of 5 days and 6 miles on the weekend. How many miles does she run in all?",
    "A library shelf holds 45 books. There are 13 full shelves and one shelf with 19 books. How many books are there?",
    "Priya has 240 marbles. She puts them into bags of 15 marbles each and sells each bag for 4 dollars. "
    "How much money does she make?",
    "A garden has 6 beds with 18 tomato plants each. A storm destroys 27 plants. How many plants survive?",
    "Omar drives 55 miles per hour for 4 hours and then 40 miles per hour for 3 hours. How many miles does he drive?",
    "A class of 28 students each brings 3 cans of food for a food drive, and the teacher adds 16 more cans. "
    "How many cans are collected?",
    "A tank holds 500 liters of water. Each day 35 liters are used for 9 days, then 120 liters are added. "
    "How many liters are in the tank?",
    "Grace buys 5 notebooks at 3 dollars each and 4 pens at 2 dollars each. She pays with a 50 dollar bill. "
    "How much change does she get?",
    "A farmer collects 14 eggs from each of 22 hens in a week and sells 6 dozen eggs. How many eggs does he have left?",
    "Ben has 4 times as many stickers as Ana. Ana has 27 stickers. Together they give away 50 stickers. "
    "How many stickers do they have left?",
    "A movie theater has 12 rows of 18 seats. On Friday 175 tickets are sold. How many seats are not sold?",
    "A factory packs 48 bottles in each crate. It fills 25 crates and has 17 bottles left over. "
    "How many bottles did it make?",
    "Hannah is paid 15 dollars an hour and works 6 hours a day for 5 days. She spends 120 dollars on groceries. "
    "How much money does she have left?",
]
MAX_TOKENS = 48          # as 48_inplace_server_test.py
SMOKE_MAX_TOKENS = 64    # "short requests (max 64 new tokens)"
ENV_KEYS = ["SGL_PATCH_DIR", "SGL_CACHE", "SGL_CORES", "SGL_SCRIPT", "SGL_EXTRA", "SGL_ENV", "MODELS_ROOT"]


class CheckFailed(Exception):
    pass


class R(C.Results):
    """Results whose first line (the "open" line) also carries the patch identity and the environment."""
    first_extra: dict | None = None

    def write(self, **kv):
        if self.first_extra is not None:
            extra, self.first_extra = self.first_extra, None
            kv = {**kv, **extra}
        super().write(**kv)


def ask_one(url, model, q, max_tokens=MAX_TOKENS, timeout=600):
    body = {"model": model, "messages": C.messages_for(q), "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": True, "chat_template_kwargs": rc.THINK_OFF}
    code, out, _sec = rc.post(url.rstrip("/") + "/v1/chat/completions", body, timeout=timeout)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"status": code, "text": None, "lp": [], "toks": [], "error": str(out)[:300]}
    ch = out["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    return {"status": 200, "text": (ch.get("message") or {}).get("content"),
            "lp": [t.get("logprob") for t in content], "toks": [t.get("token") for t in content], "error": None}


def ask_all(url, model, prompts):
    return [ask_one(url, model, q) for q in prompts]


def brief(xs):
    return [{"status": a["status"], "chars": len(a["text"] or ""), "tokens": len(a["lp"]),
             "sum_logprob": round(sum(p for p in a["lp"] if p is not None), 4), "error": a["error"]} for a in xs]


def compare(x, y):
    """Two lists of answers, prompt by prompt. identical = every request answered 200, same text, same number of
    tokens, same token strings, and the largest |logprob difference| is exactly 0.0 over at least one token."""
    n = max(len(x), len(y))
    differing, tokens, worst = [], 0, None
    texts_eq = counts_eq = toks_eq = True
    for i in range(n):
        if i >= len(x) or i >= len(y):
            differing.append(i)
            texts_eq = counts_eq = toks_eq = False
            continue
        a, b = x[i], y[i]
        if a["status"] != 200 or b["status"] != 200:
            differing.append(i)
            texts_eq = counts_eq = toks_eq = False
            continue
        bad = False
        if a["text"] != b["text"]:
            texts_eq = False
            bad = True
        if len(a["lp"]) != len(b["lp"]):
            counts_eq = False
            bad = True
        if a["toks"] != b["toks"]:
            toks_eq = False
            bad = True
        for p, q in zip(a["lp"], b["lp"]):
            tokens += 1
            if p is None or q is None:
                if (p is None) != (q is None):
                    bad = True
                    worst = float("inf")
                continue
            d = abs(p - q)
            if worst is None or d > worst:
                worst = d
            if d > 0.0:
                bad = True
        if bad:
            differing.append(i)
    ident = n > 0 and not differing and tokens > 0 and worst == 0.0
    return {"identical": ident, "prompts": n, "prompts_differing": len(differing),
            "differing_prompt_indices": differing[:10], "texts_identical": texts_eq,
            "token_counts_equal": counts_eq, "token_strings_equal": toks_eq, "tokens_compared": tokens,
            "max_abs_logprob_diff": worst}   # unrounded: "identical" uses worst == 0.0, so the reported value must match it


def read_manifest(path):
    if not path:
        return None, None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"[:300]


def run(a):
    """Returns (exit code, last line to print)."""
    manifest, manifest_err = read_manifest(a.manifest_json)
    first = {"script": "75_pr20_check.py", "argv": sys.argv[1:],
             "environment": {k: os.environ.get(k) for k in ENV_KEYS},
             "patch_manifest_file": a.manifest_json or None, "patch_manifest": manifest}
    if manifest_err:
        first["manifest_error"] = manifest_err
    elif not a.manifest_json:
        first["manifest_missing"] = ("no --manifest-json: the run does not prove which files were mounted; "
                                     "SGL_PATCH_DIR above is only the path asked for")
    R.first_extra = first
    res = R(a.results, f"75_pr20_check_{a.tag}")
    print("result file:", res.path, flush=True)
    srv = None
    failures = []
    warnings = []   # soft findings that do not fail the run (radix_repeat)
    checks = {}
    stage = "setup"

    def soft_fail(stg, msg, **kv):
        failures.append(stg)
        res.write(status="fail", stage=stg, error_type="CheckFailed", message=msg[:400], **kv)

    def need_ok(answers, which):
        bad = [i for i, x in enumerate(answers) if x["status"] != 200]
        if bad:
            raise CheckFailed(f"{which}: {len(bad)} of {len(answers)} requests failed, first: "
                              f"{answers[bad[0]]['status']} {answers[bad[0]]['error']}")

    try:
        if manifest_err:
            raise CheckFailed(f"--manifest-json could not be read: {manifest_err}")
        base, alt = Path(a.base), Path(a.alt)
        backend = server_ctl.load_backend(a.backends_json, a.backend)
        if backend.get("served_map_env"):
            root = os.environ.get(backend["served_map_env"], "").rstrip("/")
            outside = [str(p) for p in (base, alt) if not (root and str(p).startswith(root + "/"))]
            if outside:
                raise CheckFailed(f"--base and --alt must lie under {backend['served_map_env']} "
                                  f"({root or 'unset'}): {outside}")
        if not any(alt.glob("*.safetensors")):
            stage = "make_alt"
            t = time.time()
            subprocess.run([sys.executable, str(Path(__file__).with_name("mk_alt.py")), "--base", str(base),
                            "--alt", str(alt), "--noise", str(a.noise)], check=True)
            res.write(event="alt_made", noise=a.noise, seconds=round(time.time() - t, 1))
        prompts = PROMPTS[:a.prompts]
        paths = {"base": base, "alt": alt}
        srv = server_ctl.Server(backend, a.port, Path(a.results) / "server_logs", name=f"pr20_{a.tag}")

        def update(to, step):
            u = srv.update_weights_from_disk(str(paths[to]))
            res.write(event="update", step=step, to=to, status=u["status"], ok=u["ok"], seconds=u["seconds"],
                      body=u["body"][:160])
            return u

        # ---- A ------------------------------------------------------------------------------------------
        stage = "A_start_base"
        st = srv.start(str(base))
        res.write(event="start_base", **st)
        if st["ready_s"] is None:
            raise CheckFailed("server did not become ready on the base model")
        url, name = srv.base_url, srv.served_name()
        stage = "A_answers"
        A = ask_all(url, name, prompts)
        res.write(event="answers", which="A base, fresh start", answers=brief(A))
        need_ok(A, "A")
        A0b = ask_one(url, name, prompts[0])
        need_ok([A0b], "A repeat of the first prompt")
        cmp = compare(A[:1], [A0b])
        res.write(event="check", name="first_prompt_repeat_before_update", **cmp,
                  note="the second send of the first prompt may be served from the radix cache")
        if not cmp["identical"]:
            # soft: a radix-cache hit may be served through another graph or bucket on Neuron; the PR 20 checks
            # (C == A, B == D, the indirect cache check after the update) do not depend on it. Exit code stays 0.
            warnings.append("radix_repeat")
            res.write(status="warn", stage="radix_repeat",
                      message=("the first prompt gave different answers when sent twice on the base weights (a cache "
                               "hit may use other numerics); recorded, the run goes on to B, C, D: "
                               f"max_abs_logprob_diff {cmp['max_abs_logprob_diff']}, texts_identical "
                               f"{cmp['texts_identical']}")[:400])

        # ---- B ------------------------------------------------------------------------------------------
        stage = "B_update_to_alt"
        u = update("alt", "B")
        if not u["ok"]:
            raise CheckFailed(f"update to alt refused: {u['status']} {u['body'][:200]}")
        stage = "B_answers"
        B = ask_all(url, name, prompts)
        res.write(event="answers", which="B alt, in place", answers=brief(B))
        need_ok(B, "B")
        ab = compare(A, B)
        res.write(event="check", name="update_changed_answers", update_changed_answers=not ab["identical"],
                  prompts_changed=ab["prompts_differing"], of=ab["prompts"],
                  a_vs_b_max_abs_logprob_diff=ab["max_abs_logprob_diff"])
        if ab["identical"]:
            raise CheckFailed("answers after the update to alt equal the base answers: the update did nothing, "
                              "or the two checkpoints cannot be told apart")
        p0 = compare(A[:1], B[:1])
        checks["first_prompt_changed_by_update"] = not p0["identical"]
        res.write(event="check", name="first_prompt_changed_by_update", changed=not p0["identical"],
                  max_abs_logprob_diff=p0["max_abs_logprob_diff"],
                  note="indirect radix check, part 1: the first prompt was in the cache before the update; "
                       "an unchanged answer is a stale entry or a prompt the noise does not move")
        if p0["identical"]:
            soft_fail("radix_indirect", "the first prompt gives the same answer after the update to alt as before "
                                        "it (stale cache entry, or this prompt is unaffected)")

        # ---- C ------------------------------------------------------------------------------------------
        stage = "C_update_to_base"
        u = update("base", "C")
        if not u["ok"]:
            raise CheckFailed(f"update back to base refused: {u['status']} {u['body'][:200]}")
        stage = "C_answers"
        Cc = ask_all(url, name, prompts)
        res.write(event="answers", which="C base, in place again", answers=brief(Cc))
        need_ok(Cc, "C")
        ca = compare(A, Cc)
        checks["back_to_base_equals_A"] = ca["identical"]
        res.write(event="check", name="back_to_base_equals_A", **ca)
        if not ca["identical"]:
            raise CheckFailed(f"after updating back to base the answers differ from the first base answers: {ca}")

        # ---- smoke ----------------------------------------------------------------------------------------
        stage = "smoke_updates"
        res.write(event="smoke", note="smoke only, not a load test", updates=a.smoke_updates, threads=a.smoke_threads)
        ref = {"base": A, "alt": B}
        cur, smoke_ok, secs = "base", True, []
        for i in range(a.smoke_updates):
            to = "alt" if cur == "base" else "base"
            u = update(to, f"smoke {i + 1}")
            if not u["ok"]:
                soft_fail("smoke_updates", f"smoke update {i + 1} to {to} not ok: {u['status']} {u['body'][:200]}",
                          alive=srv.alive())
                smoke_ok = False      # no further updates on this server
                break
            cur = to
            secs.append(u["seconds"])
        res.write(event="smoke_updates", asked=a.smoke_updates, ok=len(secs), update_s=C.describe(secs),
                  seconds=secs, ends_on=cur if smoke_ok else "unknown")
        checks["smoke_updates_ok"] = smoke_ok
        if smoke_ok and a.smoke_updates > 0:
            stage = "smoke_updates_answers"
            S = ask_all(url, name, prompts)
            cs = compare(ref[cur], S) if all(x["status"] == 200 for x in S) else {"identical": False, "note": "a request failed"}
            res.write(event="check", name="after_smoke_updates_equals_reference", serving=cur, **cs)
            if not cs["identical"]:
                soft_fail("smoke_updates", f"after the {a.smoke_updates} smoke updates the answers differ from the "
                                           f"{cur} reference", max_abs_logprob_diff=cs.get("max_abs_logprob_diff"))
                smoke_ok = False
        if smoke_ok and a.smoke_threads > 0:
            stage = "smoke_under_load"
            to = "alt" if cur == "base" else "base"
            lock, inflight, done_reqs, stop = threading.Lock(), [0], [], threading.Event()

            def worker(w):
                k = 0
                while not stop.is_set():
                    q = prompts[(w + k) % len(prompts)]
                    k += 1
                    with lock:
                        inflight[0] += 1
                    x = ask_one(url, name, q, max_tokens=SMOKE_MAX_TOKENS, timeout=600)
                    with lock:
                        inflight[0] -= 1
                        done_reqs.append({"status": x["status"], "error": x["error"], "tokens": len(x["lp"])})

            threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(a.smoke_threads)]
            for t in threads:
                t.start()
            t0 = time.time()
            while time.time() - t0 < 30 and inflight[0] == 0:
                time.sleep(0.05)
            time.sleep(0.3)
            with lock:
                at_call = inflight[0]
            u = update(to, "smoke under load")
            with lock:
                at_ret = inflight[0]
            time.sleep(0.5)
            stop.set()
            for t in threads:
                t.join(timeout=700)
            alive_threads = sum(t.is_alive() for t in threads)
            errs = [r for r in done_reqs if r["status"] != 200]
            line = {"to": to, "update_ok": u["ok"], "update_status": u["status"], "update_seconds": u["seconds"],
                    "threads": a.smoke_threads, "in_flight_at_call": at_call, "in_flight_at_return": at_ret,
                    "requests_finished": len(done_reqs), "http_errors": len(errs),
                    "first_errors": [{"status": r["status"], "error": r["error"]} for r in errs[:3]],
                    "threads_still_running": alive_threads, "server_alive": srv.alive()}
            checks["smoke_under_load_http_errors"] = len(errs)
            if not u["ok"]:
                soft_fail("smoke_under_load", f"update under load not ok: {u['status']} {u['body'][:200]}", **line)
                smoke_ok = False
            else:
                cur = to
                if errs:
                    soft_fail("smoke_under_load", f"{len(errs)} HTTP error(s) among {len(done_reqs)} requests "
                                                  f"in flight around the update", **line)
                if alive_threads:
                    soft_fail("smoke_under_load", f"{alive_threads} request(s) did not return", **line)
                stage = "smoke_under_load_answers"
                S = ask_all(url, name, prompts)
                cs = compare(ref[cur], S) if all(x["status"] == 200 for x in S) else {"identical": False, "note": "a request failed"}
                line["answers_equal_reference_of_new_weights"] = cs["identical"]
                line["max_abs_logprob_diff"] = cs.get("max_abs_logprob_diff")
                if not cs["identical"]:
                    soft_fail("smoke_under_load", f"after the update under load the prompts do not give the {cur} "
                                                  f"answers", **{k: v for k, v in line.items() if k != "first_errors"})
            res.write(event="smoke_under_load", note="smoke only, not a load test; answers in flight are not "
                                                     "classed", **line)

        # ---- D --------------------------------------------------------------------------------------------
        if a.skip_fresh:
            res.write(event="check", name="inplace_equals_fresh", result="not_run", note="--skip-fresh")
            res.write(event="check", name="radix_cache_indirect", result="not_run", note="--skip-fresh")
        else:
            stage = "D_restart_alt"
            rs = srv.restart(str(alt))
            res.write(event="restart_alt", **rs)
            if rs["ready_s"] is None:
                raise CheckFailed("server did not become ready on the alt model")
            stage = "D_answers"
            D = ask_all(srv.base_url, srv.served_name(), prompts)
            res.write(event="answers", which="D alt, fresh start", answers=brief(D))
            need_ok(D, "D")
            bd = compare(B, D)
            checks["inplace_equals_fresh"] = bd["identical"]
            res.write(event="check", name="inplace_equals_fresh", **bd)
            if not bd["identical"]:
                soft_fail("fresh_vs_inplace", f"in-place update and fresh start serve different answers: "
                                              f"max |logprob diff| {bd['max_abs_logprob_diff']}, "
                                              f"{bd['prompts_differing']} of {bd['prompts']} prompts differ",
                          max_abs_logprob_diff=bd["max_abs_logprob_diff"])
            r0 = compare(B[:1], D[:1])
            checks["radix_cache_indirect"] = r0["identical"]
            res.write(event="check", name="radix_cache_indirect", first_prompt_after_update_equals_fresh=r0["identical"],
                      max_abs_logprob_diff=r0["max_abs_logprob_diff"],
                      note="indirect: the server reports no cached-token count. The first prompt was sent twice "
                           "before the update (cache holds it) and right after it (B); equality with the fresh "
                           "start D says no stale entry was served, it does not show the cache was flushed")
            if not r0["identical"]:
                soft_fail("radix_indirect", "the first prompt after the update differs from the fresh start's",
                          max_abs_logprob_diff=r0["max_abs_logprob_diff"])
        stage = "summary"
        res.write(event="summary", passed=not failures, failed_stages=sorted(set(failures)),
                  warnings=sorted(set(warnings)), checks=checks,
                  prompts=len(prompts), patch_dir=os.environ.get("SGL_PATCH_DIR"))
    except BaseException as e:  # noqa: BLE001
        failures.append(stage)
        res.fail(stage, e)
        if not isinstance(e, (CheckFailed, Exception)):
            raise
        if not isinstance(e, CheckFailed):
            traceback.print_exc()
    finally:
        if srv is not None:
            try:
                srv.stop()
            except Exception:  # noqa: BLE001
                pass
        res.done()
    if failures:
        return 1, f"75_pr20_check FAILED at {', '.join(sorted(set(failures)))}; see {res.path}"
    warn_txt = f" WITH WARNING {', '.join(sorted(set(warnings)))} (soft, see the warn line)" if warnings else ""
    return 0, (f"75_pr20_check PASSED{warn_txt}: C == A and B == D with max logprob difference 0.0 on {len(prompts)} "
               f"prompts, smoke ok; {res.path}")


def main():
    ap = argparse.ArgumentParser(description="PR 20 check: in-place update against fresh start, with a short smoke")
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, default=30075)
    ap.add_argument("--prompts", type=int, default=20, help=f"1 to {len(PROMPTS)}")
    ap.add_argument("--smoke-updates", type=int, default=10)
    ap.add_argument("--smoke-threads", type=int, default=8)
    ap.add_argument("--skip-fresh", action="store_true", help="leave out step D (restart on alt)")
    ap.add_argument("--manifest-json", default="", help="file written by patch_manifest.py; copied into the first line")
    ap.add_argument("--noise", type=float, default=0.01, help="only when --alt has to be made (mk_alt.py)")
    a = ap.parse_args()
    if not 1 <= a.prompts <= len(PROMPTS):
        ap.error(f"--prompts must be between 1 and {len(PROMPTS)}")
    if a.smoke_updates < 0 or a.smoke_threads < 0:
        ap.error("--smoke-updates and --smoke-threads must not be negative")
    try:
        code, last = run(a)
    except BaseException as e:  # noqa: BLE001
        print(f"75_pr20_check FAILED: {type(e).__name__}: {e}", flush=True)
        raise
    print(last, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
