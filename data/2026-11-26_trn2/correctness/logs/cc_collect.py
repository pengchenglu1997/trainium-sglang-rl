#!/usr/bin/env python3
"""Correctness session 2026-11-26 (cc_): start ONE server (config by name), send request sets one at a time,
save every answer (tokens, logprobs, text) to ~/w/cc/results/out_<name>_<set>.json, optionally run GSM8K with
the kit's 21_gsm8k_acc.py, stop the server. Kit modules are imported read-only from ~/w/sglang_rl.

    python cc_collect.py <name> <cores> <port> <set,set,...>
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
CC = Path.home() / "w" / "cc"
RES = CC / "results"
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
S8 = "/opt/sglang-neuron/examples/qwen3_8b/run_server.sh"
P_LMH = str(KIT / "patches" / "sglang_neuron_lmhead_int_26213e7")
P_TOPK = str(KIT / "patches" / "sglang_neuron_int_topk")

# exact env of tonight's servers, from ~/w/results_s4/76_serve_b_<tag>.jsonl "config" events
CFG = {
    "lmh_off_a": ("q06_lmh_off", "Qwen3-0.6B", S06, "temperature", P_LMH, "SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0"),
    "lmh_off_b": ("q06_lmh_off", "Qwen3-0.6B", S06, "temperature", P_LMH, "SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=0"),
    "lmh_on":    ("q06_lmh_on",  "Qwen3-0.6B", S06, "temperature", P_LMH, "SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"),
    "koff_a":    ("q06_topk_koff", "Qwen3-0.6B", S06, "topk_topp", P_TOPK, ""),
    "koff_b":    ("q06_topk_koff", "Qwen3-0.6B", S06, "topk_topp", P_TOPK, ""),
    "kon":       ("q06_topk_kon",  "Qwen3-0.6B", S06, "topk_topp", P_TOPK, "RLKIT_SAMPLER_KERNEL_TOPK=1"),
    "all":       ("q06_all", "Qwen3-0.6B", S06, "temperature", P_TOPK,
                  "RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"),
    "rel":       ("q06_rel", "Qwen3-0.6B", S06, "temperature", "", ""),
    "rel_b":     ("q06_rel", "Qwen3-0.6B", S06, "temperature", "", ""),
    "q8_rel_b":  ("q8_rel", "Qwen3-8B", S8, "temperature", "", ""),
    "q8_all":    ("q8_all", "Qwen3-8B", S8, "temperature", P_TOPK,
                  "RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_LM_HEAD_PRETRANSPOSED=1"),
    "q8_rel":    ("q8_rel", "Qwen3-8B", S8, "temperature", "", ""),
}
FULL = {"top_p": 1.0, "top_k": -1, "min_p": 0.0}
SETS = {   # name: (params, seeded)
    "t0": ({"temperature": 0.0}, False),
    "t08": (dict(temperature=0.8, **FULL), True),
    "rl": ({"temperature": 0.8, "top_k": 50, "top_p": 0.95, "min_p": 0.0}, True),
    "minp": ({"temperature": 0.8, "top_k": -1, "top_p": 1.0, "min_p": 0.1}, True),
}
SEED0 = 1000          # seed of prompt i = SEED0 + i, the same on every server
KERNEL_LINES = ["sampler by the kernel", "topk_topp sampler by the kernel"]


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(ev, **kv):
    kv = dict(t=utc(), **kv)
    with open(ev, "a", encoding="utf-8") as f:
        f.write(json.dumps(kv, default=str) + "\n")
    print(json.dumps(kv, default=str)[:300], flush=True)


def chat(srv, prompt, params, seed=None, max_tokens=48):
    body = {"model": srv.served_name(), "messages": O.C.messages_for(prompt), "max_tokens": max_tokens,
            "logprobs": True, "chat_template_kwargs": rc.THINK_OFF, **params}
    if seed is not None:
        body["seed"] = seed
    code, out, sec = rc.post(srv.base_url + "/v1/chat/completions", body, 300)
    if code != 200 or not isinstance(out, dict) or not out.get("choices"):
        return {"ok": False, "status": code, "error": str(out)[:300], "body": body}
    ch = out["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content")) or []
    return {"ok": True, "status": code, "text": (ch.get("message") or {}).get("content"),
            "finish_reason": ch.get("finish_reason"), "seconds": round(sec, 3),
            "tokens": [c.get("token") for c in content], "logprobs": [c.get("logprob") for c in content],
            "seed": seed, "params": params}


def main():
    name, cores, port, sets = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",")
    src, model, script, sampler, patch, xenv = CFG[name]
    RES.mkdir(parents=True, exist_ok=True)
    ev = RES / f"events_{name}.jsonl"
    data_root = Path.home() / "w"
    cache = data_root / f"sgl_cache_cc_{name}"
    orig = data_root / f"sgl_cache_s4_{src}"
    if not cache.exists():
        # the originals hold root-owned 0600 files written by the containers: read them with sudo (source untouched)
        subprocess.run(["sudo", "-n", "cp", "-a", str(orig), str(cache)], check=True)
    nfiles = int(subprocess.run(["sudo", "-n", "find", str(cache), "-type", "f"], capture_output=True,
                                text=True).stdout.count("\n"))
    norig = int(subprocess.run(["sudo", "-n", "find", str(orig), "-type", "f"], capture_output=True,
                               text=True).stdout.count("\n"))
    env = {"DATA_ROOT": str(data_root), "MODELS_ROOT": str(data_root / "models"), "SGL_CACHE": str(cache),
           "SGL_CORES": cores, "SGL_SAMPLER": sampler, "SGL_EXTRA": FLAGS, "SGL_SCRIPT": script,
           "SGL_TOP_LOGPROBS": "", "SGL_PATCH_DIR": patch, "SGL_ENV": xenv, "SGL_MOUNTS": ""}
    os.environ["MODELS_ROOT"] = env["MODELS_ROOT"]
    backend = dict(server_ctl.load_backend(KIT / "backends.json", "sglang_neuron"), ready_timeout_s=1500)
    srv = server_ctl.Server(backend, port, RES / "server_logs", name=f"cc_{name}", env=env)
    log(ev, event="open", name=name, cores=cores, port=port, sets=sets, cache_src=str(orig), cache=str(cache),
        cache_files_at_start=nfiles, cache_files_in_original=norig, env=env)
    status = "failed"
    try:
        t0 = time.time()
        r = srv.start(str(data_root / "models" / model))
        log(ev, event="server_start", **r)
        if r.get("ready_s") is None:
            log(ev, event="not_ready", last_lines=O.tail_lines(r.get("log"), 30) if r.get("log") else [])
            return 1
        for s in sets:
            if s == "gsm8k":
                cmd = [sys.executable, str(KIT / "21_gsm8k_acc.py"), "--base-url", srv.base_url,
                       "--model", srv.served_name(), "--data", str(data_root / "models" / "gsm8k" / "test.jsonl"),
                       "--results", str(RES), "--tag", f"cc_{name}", "--limit", "200", "--temperature", "0"]
                p = subprocess.run(cmd, cwd=str(CC), capture_output=True, text=True, timeout=3600,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                log(ev, event="gsm8k", cmd=cmd, exit=p.returncode, stdout_tail=p.stdout[-500:],
                    stderr_tail=p.stderr[-800:])
                continue
            params, seeded = SETS[s]
            outs = [chat(srv, q, params, (SEED0 + i) if seeded else None) for i, q in enumerate(PROMPTS)]
            extra = {}
            if seeded:   # seed effective? prompt 5 again with its seed, and with another seed
                extra["same_seed_again"] = chat(srv, PROMPTS[5], params, SEED0 + 5)
                extra["other_seed"] = chat(srv, PROMPTS[5], params, SEED0 + 5 + 100000)
            (RES / f"out_{name}_{s}.json").write_text(json.dumps({"name": name, "set": s, "params": params,
                                                                   "answers": outs, **extra}), encoding="utf-8")
            log(ev, event="set_done", set=s, ok=sum(o["ok"] for o in outs), of=len(outs),
                first_error=next((o.get("error") for o in outs if not o["ok"]), None))
        status = "ok"
        return 0
    except BaseException as e:  # noqa: BLE001
        log(ev, event="error", error=f"{type(e).__name__}: {e}"[:400])
        return 1
    finally:
        try:
            text = subprocess.run(["docker", "logs", f"cc_{name}"], capture_output=True, text=True,
                                  timeout=120).stdout
            text += subprocess.run(["docker", "logs", f"cc_{name}"], capture_output=True, text=True,
                                   timeout=120).stderr
            counts = {k: sum(1 for x in text.splitlines() if k in x) for k in KERNEL_LINES}
            low = text.lower()
            log(ev, event="log_lines", counts=counts, pretransposed_mentions=low.count("pretranspos"),
                examples=[x[-250:] for x in text.splitlines() if "by the kernel" in x or "pretranspos" in x.lower()][:6])
        except Exception as e:  # noqa: BLE001
            log(ev, event="log_scan_error", error=str(e)[:200])
        log(ev, event="stop", seconds=srv.stop())
        log(ev, event="end", status=status)
        log(ev, done=True)
        print(f"CC_COLLECT_{'DONE' if status == 'ok' else 'FAILED'} {name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
