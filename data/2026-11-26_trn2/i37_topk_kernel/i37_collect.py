#!/usr/bin/env python3
"""Line i37: start ONE SGLang-Neuron server (topk_topp sampler), send the 20 prompts of the cc_ correctness run at
temperature 0 with logprobs and top_logprobs, save every answer, stop the server.

    python i37_collect.py <name> <cores> <port> <cache source tag> <patch dir> "<K=V ...>" <top_logprobs>

The cache is a fresh sudo cp -a of ~/w/sgl_cache_s4_<source tag> into ~/w/sgl_cache_i37_<name> (source untouched).
Kit modules are imported read-only from ~/w/sglang_rl.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

KIT = Path.home() / "w" / "sglang_rl"
W = Path.home() / "w" / "i37"
RES = W / "results"
sys.path.insert(0, str(KIT))


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("hw82", KIT / "82_sampler_kernel_hw.py")
O, rc, server_ctl = H.O, H.rc, H.server_ctl
PROMPTS = H.PROMPTS
FLAGS = H.FLAGS
S06 = "/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh"
KERNEL_LINES = ["sampler by the kernel", "topk_topp sampler by the kernel"]


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(ev, **kv):
    kv = dict(t=utc(), **kv)
    with open(ev, "a", encoding="utf-8") as f:
        f.write(json.dumps(kv, default=str) + "\n")
    print(json.dumps(kv, default=str)[:300], flush=True)


def chat(srv, prompt, params, top, max_tokens):
    body = {"model": srv.served_name(), "messages": O.C.messages_for(prompt), "max_tokens": max_tokens,
            "logprobs": True, "chat_template_kwargs": rc.THINK_OFF, **params}
    if top:
        body["top_logprobs"] = top
    code, out, sec = rc.post(srv.base_url + "/v1/chat/completions", body, 300)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"ok": False, "status": code, "error": str(out)[:300]}
    ch = out["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content")) or []
    return {"ok": True, "status": code, "text": (ch.get("message") or {}).get("content"),
            "finish_reason": ch.get("finish_reason"), "seconds": round(sec, 3),
            "tokens": [c.get("token") for c in content], "logprobs": [c.get("logprob") for c in content],
            "top": [[[t.get("token"), t.get("logprob")] for t in (c.get("top_logprobs") or [])] for c in content],
            "params": params}


def main():
    name, cores, port, src, patch, xenv, top = sys.argv[1:8]
    port, top = int(port), int(top)
    max_tokens = int(os.environ.get("I37_MAX_TOKENS", "48"))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "server_logs").mkdir(exist_ok=True)
    ev = RES / f"events_{name}.jsonl"
    data_root = Path.home() / "w"
    cache = data_root / f"sgl_cache_i37_{name}"
    orig = data_root / f"sgl_cache_s4_{src}"
    if not cache.exists():
        if src == "none":   # empty cache: cold start
            cache.mkdir(parents=True)
        else:
            subprocess.run(["sudo", "-n", "cp", "-a", str(orig), str(cache)], check=True)
    nfiles = subprocess.run(["sudo", "-n", "find", str(cache), "-type", "f"], capture_output=True,
                            text=True).stdout.count("\n")
    env = {"DATA_ROOT": str(data_root), "MODELS_ROOT": str(data_root / "models"), "SGL_CACHE": str(cache),
           "SGL_CORES": cores, "SGL_SAMPLER": "topk_topp", "SGL_EXTRA": FLAGS, "SGL_SCRIPT": S06,
           "SGL_TOP_LOGPROBS": str(top) if top else "", "SGL_PATCH_DIR": patch, "SGL_ENV": xenv, "SGL_MOUNTS": ""}
    os.environ["MODELS_ROOT"] = env["MODELS_ROOT"]
    backend = dict(server_ctl.load_backend(KIT / "backends.json", "sglang_neuron"), ready_timeout_s=2000)
    srv = server_ctl.Server(backend, port, RES / "server_logs", name=f"i37_{name}", env=env)
    log(ev, event="open", name=name, cores=cores, port=port, cache_src=str(orig), cache=str(cache),
        cache_files_at_start=nfiles, env=env, max_tokens=max_tokens)
    status = "failed"
    try:
        r = srv.start(str(data_root / "models" / "Qwen3-0.6B"))
        log(ev, event="server_start", **r)
        if r.get("ready_s") is None:
            log(ev, event="not_ready", last_lines=O.tail_lines(r.get("log"), 30) if r.get("log") else [])
            return 1
        params = {"temperature": 0.0}
        outs = [chat(srv, q, params, top, max_tokens) for q in PROMPTS]
        again = [chat(srv, q, params, top, max_tokens) for q in PROMPTS[:3]]   # same server, repeat
        (RES / f"out_{name}_t0.json").write_text(json.dumps({"name": name, "set": "t0", "params": params,
                                                             "top_logprobs": top, "answers": outs,
                                                             "repeat_first3": again}), encoding="utf-8")
        log(ev, event="set_done", set="t0", ok=sum(o["ok"] for o in outs), of=len(outs),
            first_error=next((o.get("error") for o in outs if not o["ok"]), None))
        status = "ok"
        return 0
    except BaseException as e:  # noqa: BLE001
        log(ev, event="error", error=f"{type(e).__name__}: {e}"[:400])
        return 1
    finally:
        try:
            p = subprocess.run(["docker", "logs", f"i37_{name}"], capture_output=True, text=True, timeout=120)
            text = p.stdout + p.stderr
            (RES / "server_logs" / f"docker_i37_{name}.log").write_text(text, encoding="utf-8")
            counts = {k: sum(1 for x in text.splitlines() if k in x) for k in KERNEL_LINES}
            log(ev, event="log_lines", counts=counts,
                examples=[x[-250:] for x in text.splitlines() if "by the kernel" in x][:4])
        except Exception as e:  # noqa: BLE001
            log(ev, event="log_scan_error", error=str(e)[:200])
        nf = subprocess.run(["sudo", "-n", "find", str(cache), "-type", "f"], capture_output=True,
                            text=True).stdout.count("\n")
        log(ev, event="stop", seconds=srv.stop(), cache_files_at_end=nf)
        log(ev, event="end", status=status)
        print(f"I37_COLLECT_{'DONE' if status == 'ok' else 'FAILED'} {name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
