#!/usr/bin/env python3
"""Adapter-only weight sync into a patched SGLang-Neuron server: is what it serves the same as a
fresh start on the merged checkpoint?

    (environment as 48_inplace_server_test.py, patch with _experiment_apply_adapter)
    python 50_adapter_server_test.py --base DIR --work DIR_UNDER_MODELS_ROOT --results DIR --tag adapter

Made on the host first (seeded, PEFT file layout, rank 16 on the trainer's seven projections):
  <work>/adapter1, <work>/adapter2   adapter_config.json + adapter_model.safetensors (+ the base's
                                     config and tokenizer files, which SGLang reads from the path)
  <work>/adapter0                    the same layout with B = 0 (no change to the base)
  <work>/merged1                     the base checkpoint with adapter1 merged in on the host
Steps: start on the base (answers A); update to adapter1 (B1); update to adapter2 (B2); update to
adapter1 again (B1'); update to adapter0 (A'); restart on merged1 (D).
Checks: B1 differs from A; B1' equals B1 exactly (adapters replace, they do not add up);
A' equals A exactly (the base comes back); B1 equals D (adapter sync serves what a restart on
the merged weights serves). Equality is the text and every token logprob."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROMPTS = [
    "A farmer has 17 rows of 23 apple trees. He cuts down 41 trees. How many trees are left?",
    "Tom reads 12 pages a day for 9 days and then 30 pages more. How many pages did he read?",
    "A box holds 24 pencils. A school buys 15 boxes and gives away 80 pencils. How many remain?",
]


def build(base: Path, work: Path, rank: int, alpha: int, std: float):
    import torch
    from safetensors.torch import load_file, save_file
    sd = {}
    for f in sorted(base.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    small = [f for f in base.iterdir() if f.is_file() and f.suffix != ".safetensors"]
    keys = [k for k in sorted(sd) if k.endswith(".weight") and k.split(".")[-2] in TARGETS and sd[k].dim() == 2]

    def adapter(seed, zero_b=False):
        g = torch.Generator().manual_seed(seed)
        out = {}
        for k in keys:
            o, i = sd[k].shape
            stem = "base_model.model." + k[:-len(".weight")]
            out[stem + ".lora_A.weight"] = (std * torch.randn(rank, i, generator=g)).to(torch.float32)
            b = torch.zeros(o, rank) if zero_b else std * torch.randn(o, rank, generator=g)
            out[stem + ".lora_B.weight"] = b.to(torch.float32)
        return out

    def write(name, tensors):
        d = work / name
        d.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
        (d / "adapter_config.json").write_text(json.dumps(
            {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": alpha,
             "lora_dropout": 0.0, "bias": "none", "target_modules": TARGETS,
             "base_model_name_or_path": str(base)}, indent=1))
        for f in small:
            shutil.copy2(f, d / f.name)
        return sum(t.numel() * t.element_size() for t in tensors.values())

    a1, a2 = adapter(101), adapter(202)
    nbytes = write("adapter1", a1)
    write("adapter2", a2)
    write("adapter0", adapter(303, zero_b=True))
    scale = alpha / rank
    merged = dict(sd)
    for k in keys:
        stem = "base_model.model." + k[:-len(".weight")]
        delta = scale * (a1[stem + ".lora_B.weight"] @ a1[stem + ".lora_A.weight"])
        merged[k] = (sd[k].float() + delta).to(sd[k].dtype)
    md = work / "merged1"
    md.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(md / "model.safetensors"), metadata={"format": "pt"})
    for f in small:
        shutil.copy2(f, md / f.name)
    return {"adapted_matrices": len(keys), "adapter_bytes": nbytes,
            "base_bytes": sum(t.numel() * t.element_size() for t in sd.values())}


def ask(base_url, model):
    outs = []
    for q in PROMPTS:
        body = {"model": model, "messages": C.messages_for(q), "max_tokens": 48, "temperature": 0.0,
                "logprobs": True, "chat_template_kwargs": {"enable_thinking": False}}
        code, out, sec = rc.post(base_url.rstrip("/") + "/v1/chat/completions", body, timeout=600)
        if code != 200 or not isinstance(out, dict):
            outs.append({"status": code, "text": None, "lp": [], "error": str(out)[:300]})
            continue
        ch = out["choices"][0]
        outs.append({"status": 200, "text": ch["message"]["content"],
                     "lp": [t.get("logprob") for t in ((ch.get("logprobs") or {}).get("content") or [])],
                     "tok": [t.get("token") for t in ((ch.get("logprobs") or {}).get("content") or [])]})
    return outs


def cmp(x, y):
    eq = all(a["text"] is not None and a["text"] == b["text"] for a, b in zip(x, y))
    d = [abs(p - q) for a, b in zip(x, y) for p, q in zip(a["lp"], b["lp"]) if p is not None and q is not None]
    return {"same_text": eq, "max_logprob_diff": round(max(d), 6) if d else None}


def brief(x):
    return [{"status": a["status"], "tokens": len(a["lp"]),
             "sum_logprob": round(sum(p for p in a["lp"] if p is not None), 4), "error": a.get("error")} for a in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--work", required=True, help="a folder under MODELS_ROOT for the adapters and merged1")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, default=30013)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--std", type=float, default=0.01)
    a = ap.parse_args()
    res = C.Results(a.results, f"50_adapter_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=f"srv_{a.tag}")
    try:
        base, work = Path(a.base), Path(a.work)
        t = time.time()
        info = build(base, work, a.rank, a.alpha, a.std)
        res.write(event="built", seconds=round(time.time() - t, 1), rank=a.rank, alpha=a.alpha, std=a.std,
                  share_of_base_bytes=round(info["adapter_bytes"] / info["base_bytes"], 5), **info)
        st = srv.start(str(base))
        res.write(event="start_base", **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready on the base model")
        name = srv.served_name()
        got = {"A": ask(srv.base_url, name)}
        res.write(event="answers", which="A base", answers=brief(got["A"]))
        for label, folder in (("B1", "adapter1"), ("B2", "adapter2"), ("B1 again", "adapter1"), ("A again", "adapter0")):
            u = srv.update_weights_from_disk(str(work / folder))
            res.write(event="update", to=folder, **u)
            if not u["ok"]:
                raise RuntimeError(f"update to {folder} refused")
            got[label] = ask(srv.base_url, name)
            res.write(event="answers", which=label, answers=brief(got[label]))
        res.write(event="check", what="adapter1 changes the answers", **cmp(got["A"], got["B1"]))
        res.write(event="check", what="adapter2 differs from adapter1", **cmp(got["B1"], got["B2"]))
        res.write(event="check", what="adapter1 again equals adapter1 (replace, not add)", **cmp(got["B1"], got["B1 again"]))
        res.write(event="check", what="zero adapter gives the base back", **cmp(got["A"], got["A again"]))
        rs = srv.restart(str(work / "merged1"))
        res.write(event="restart_merged1", **rs)
        if rs["ready_s"] is not None:
            got["D"] = ask(srv.base_url, srv.served_name())
            res.write(event="answers", which="D merged1, fresh start", answers=brief(got["D"]))
            res.write(event="check", what="adapter sync equals a fresh start on the merged checkpoint",
                      **cmp(got["B1"], got["D"]))
            # Greedy decoding can fork at a near-tie: after the fork the two answers are different texts and their
            # logprobs are no longer comparable. Per prompt: the first position where the tokens differ, the largest
            # logprob difference BEFORE it, and the two logprobs at the fork.
            for i, (x, y) in enumerate(zip(got["B1"], got["D"])):
                n = min(len(x["tok"]), len(y["tok"]))
                fork = next((j for j in range(n) if x["tok"][j] != y["tok"][j]), None)
                upto = n if fork is None else fork
                before = [abs(x["lp"][j] - y["lp"][j]) for j in range(upto)
                          if x["lp"][j] is not None and y["lp"][j] is not None]
                res.write(event="fork", prompt=i, tokens=n, first_different_token=fork,
                          max_logprob_diff_before_fork=round(max(before), 6) if before else None,
                          mean_logprob_diff_before_fork=round(sum(before) / len(before), 6) if before else None,
                          logprob_at_fork=None if fork is None else [x["lp"][fork], y["lp"][fork]])
    except BaseException as e:  # noqa: BLE001
        res.fail("adapter", e)
        raise
    finally:
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
