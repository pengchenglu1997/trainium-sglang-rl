#!/usr/bin/env python3
"""dp_ slot: client half of the correctness method of ~/w/cc/cc_collect.py (2026-11-26), against a running server.
Same 20 PROMPTS (82_sampler_kernel_hw.PROMPTS), /v1/chat/completions, one request at a time, max_tokens 48,
logprobs true, thinking off, temperature 0, no seed. Writes {"answers": [...]} like out_<name>_t0.json.

    python dp_cc.py <base url> <served model name> <out json>

Last line "LLMBENCH_DONE cc ..." (so 76_serve_and_bench.py counts the step as done) or "LLMBENCH_FAILED cc ..." (exit 1).
"""
import importlib.util
import json
import sys
from pathlib import Path

KIT = Path.home() / "w" / "sglang_rl"
sys.path.insert(0, str(KIT))


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("hw82", KIT / "82_sampler_kernel_hw.py")
O, rc = H.O, H.rc


def chat(url, served, prompt, params, max_tokens=48):
    body = {"model": served, "messages": O.C.messages_for(prompt), "max_tokens": max_tokens,
            "logprobs": True, "chat_template_kwargs": rc.THINK_OFF, **params}
    code, out, sec = rc.post(url + "/v1/chat/completions", body, 300)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"ok": False, "status": code, "error": str(out)[:300], "body": body}
    ch = out["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content")) or []
    return {"ok": True, "status": code, "text": (ch.get("message") or {}).get("content"),
            "finish_reason": ch.get("finish_reason"), "seconds": round(sec, 3),
            "tokens": [c.get("token") for c in content], "logprobs": [c.get("logprob") for c in content],
            "seed": None, "params": params}


def main():
    url, served, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    params = {"temperature": 0.0}
    outs = [chat(url, served, q, params) for q in H.PROMPTS]
    out.write_text(json.dumps({"name": out.stem, "set": "t0", "params": params, "answers": outs}), encoding="utf-8")
    ok = sum(o["ok"] for o in outs)
    if ok != len(outs):
        print(f"LLMBENCH_FAILED cc {ok}/{len(outs)} ok, first error: "
              f"{next(o.get('error') for o in outs if not o['ok'])}", flush=True)
        return 1
    print(f"LLMBENCH_DONE cc csv={out} rows={len(outs)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
