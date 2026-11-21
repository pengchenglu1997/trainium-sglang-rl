"""Client for an OpenAI-compatible server (SGLang, SGLang-Neuron, vLLM, vllm-neuron).
Standard library only."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

THINK_OFF = {"enable_thinking": False}


def post(url, body, timeout=600):
    """POST JSON. Returns (http status or 0, parsed body or text, seconds)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, code = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read().decode("utf-8", "replace"), e.code
    except Exception as e:  # noqa: BLE001  (connection refused, timeout)
        return 0, f"{type(e).__name__}: {e}", time.time() - t
    try:
        return code, json.loads(raw), time.time() - t
    except ValueError:
        return code, raw[:2000], time.time() - t


def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")[:2000]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:2000]
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def chat(base_url, model, messages, temperature=0.0, max_tokens=256, n=1, seed=None,
         logprobs=False, top_logprobs=None, extra=None, timeout=600):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "chat_template_kwargs": THINK_OFF}
    if temperature is not None:      # None = leave the field out (the server's own default)
        body["temperature"] = temperature
        if temperature > 0:
            # Sample the full vocabulary at this temperature: it is what the trainer's loss
            # assumes, and a model's generation_config (Qwen3: top_k 20, top_p 0.95) would
            # otherwise be applied as the request default. SGLang-Neuron's "temperature"
            # sampler answers 400 to any filtered request (2026-11-20, 20_rl_probes_sgl).
            body.update({"top_p": 1.0, "top_k": -1, "min_p": 0.0})
    if n and n > 1:
        body["n"] = n
    if seed is not None:
        body["seed"] = seed
    if logprobs:
        body["logprobs"] = True
        if top_logprobs:
            body["top_logprobs"] = top_logprobs
    if extra:
        body.update(extra)
    code, out, sec = post(base_url.rstrip("/") + "/v1/chat/completions", body, timeout)
    res = {"status": code, "seconds": sec, "texts": [], "completion_tokens": None,
           "logprob_tokens": None, "error": None}
    if code == 200 and isinstance(out, dict) and out.get("choices"):
        res["texts"] = [(c.get("message") or {}).get("content") or "" for c in out["choices"]]
        res["completion_tokens"] = (out.get("usage") or {}).get("completion_tokens")
        lp = out["choices"][0].get("logprobs")
        if lp and lp.get("content"):
            res["logprob_tokens"] = len(lp["content"])
    else:
        res["error"] = out if isinstance(out, str) else json.dumps(out)[:600]
    return res


def sample_group(base_url, model, messages, k, temperature, max_tokens, mode="auto",
                 seed=None, timeout=600):
    """k sampled answers to one prompt.

    mode "n": one request with n=k. mode "parallel": k requests at once. mode "auto": try "n",
    fall back to "parallel" when the server refuses it (the mode used is returned)."""
    if mode in ("n", "auto"):
        r = chat(base_url, model, messages, temperature, max_tokens, n=k, seed=seed, timeout=timeout)
        if r["status"] == 200 and len(r["texts"]) == k:
            return {"mode": "n", "texts": r["texts"], "seconds": r["seconds"],
                    "completion_tokens": r["completion_tokens"], "errors": 0, "first_error": None}
        if mode == "n":
            return {"mode": "n", "texts": [], "seconds": r["seconds"], "completion_tokens": None,
                    "errors": 1, "first_error": r["error"]}
    t = time.time()
    with ThreadPoolExecutor(max_workers=k) as ex:
        rs = list(ex.map(lambda i: chat(base_url, model, messages, temperature, max_tokens,
                                        seed=None if seed is None else seed + i,
                                        timeout=timeout), range(k)))
    texts = [x["texts"][0] if x["status"] == 200 and x["texts"] else None for x in rs]
    bad = [x for x in rs if x["status"] != 200 or not x["texts"]]
    toks = [x["completion_tokens"] for x in rs if x["completion_tokens"] is not None]
    return {"mode": "parallel", "texts": texts, "seconds": time.time() - t,
            "completion_tokens": sum(toks) if toks else None, "errors": len(bad),
            "first_error": bad[0]["error"] if bad else None}


def wait_ready(base_url, model, timeout_s, poll_s=5, alive=None):
    """Seconds until the server answers one greedy 1-token chat request; None on timeout or when
    alive() says the server process has ended."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if alive is not None and not alive():
            return None
        r = chat(base_url, model, [{"role": "user", "content": "Say OK."}], temperature=0.0,
                 max_tokens=1, timeout=120)
        if r["status"] == 200:
            return time.time() - t0
        time.sleep(poll_s)
    return None
