#!/usr/bin/env python3
"""Does a patched SGLang-Neuron server take new weights in place, and are they the right ones?

    MODELS_ROOT=... SGL_CACHE=... SGL_PATCH_DIR=<kit>/patches/sglang_neuron_v0.1.0_inplace \
    SGL_EXTRA="..." python 48_inplace_server_test.py --base DIR --alt DIR --results DIR --tag inplace

Steps, each written as one line:
  1. make --alt if it is missing: a copy of --base whose matrices carry small seeded noise
  2. start the server on --base, greedy answers with token logprobs to fixed prompts      (A)
  3. POST /update_weights_from_disk to --alt, time it; greedy answers                       (B)
  4. POST /update_weights_from_disk back to --base, time it; greedy answers                 (C)
  5. restart the server on --alt (the ordinary way to new weights); greedy answers          (D)
Checks: B differs from A (the update did something); C equals A (in place both ways);
B equals D (the in-place update serves what a fresh start on the same checkpoint serves).
"Equals" is the text and the largest difference of token logprobs."""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402

PROMPTS = [
    "A farmer has 17 rows of 23 apple trees. He cuts down 41 trees. How many trees are left?",
    "Tom reads 12 pages a day for 9 days and then 30 pages more. How many pages did he read?",
    "A box holds 24 pencils. A school buys 15 boxes and gives away 80 pencils. How many remain?",
]


def make_alt(base: Path, alt: Path, noise: float):
    import torch
    from safetensors.torch import load_file, save_file
    alt.mkdir(parents=True, exist_ok=True)
    changed = 0
    for f in base.iterdir():
        if f.is_dir():
            continue
        if f.suffix == ".safetensors":
            sd = load_file(str(f))
            g = torch.Generator().manual_seed(7)
            out = {}
            for k in sorted(sd):
                w = sd[k]
                if w.dim() >= 2 and w.is_floating_point():
                    out[k] = (w.float() + noise * w.float().std() * torch.randn(w.shape, generator=g)).to(w.dtype)
                    changed += 1
                else:
                    out[k] = w
            save_file(out, str(alt / f.name), metadata={"format": "pt"})
        else:
            shutil.copy2(f, alt / f.name)
    return changed


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
        lp = [t.get("logprob") for t in ((ch.get("logprobs") or {}).get("content") or [])]
        outs.append({"status": 200, "text": ch["message"]["content"], "lp": lp})
    return outs


def same(x, y):
    """(texts equal on every prompt, largest |logprob difference| over tokens at equal positions)."""
    eq = all(a["text"] is not None and a["text"] == b["text"] for a, b in zip(x, y))
    d = [abs(p - q) for a, b in zip(x, y) for p, q in zip(a["lp"], b["lp"]) if p is not None and q is not None]
    return eq, (round(max(d), 5) if d else None)


def brief(x):
    return [{"status": a["status"], "chars": len(a["text"] or ""), "tokens": len(a["lp"]),
             "sum_logprob": round(sum(p for p in a["lp"] if p is not None), 4),
             "error": a.get("error")} for a in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="sglang_neuron")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, default=30005)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--skip-fresh", action="store_true", help="leave out step 5")
    a = ap.parse_args()
    res = C.Results(a.results, f"48_inplace_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=f"srv_{a.tag}")
    try:
        base, alt = Path(a.base), Path(a.alt)
        if not any(alt.glob("*.safetensors")):
            t = time.time()
            n = make_alt(base, alt, a.noise)
            res.write(event="alt_made", matrices_changed=n, noise=a.noise, seconds=round(time.time() - t, 1))
        st = srv.start(str(base))
        res.write(event="start_base", **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready on the base model")
        name = srv.served_name()
        A = ask(srv.base_url, name)
        res.write(event="answers", which="A base, fresh start", answers=brief(A))
        u1 = srv.update_weights_from_disk(str(alt))
        res.write(event="update", to="alt", **u1)
        B = ask(srv.base_url, name)
        res.write(event="answers", which="B alt, in place", answers=brief(B))
        u2 = srv.update_weights_from_disk(str(base))
        res.write(event="update", to="base", **u2)
        Cc = ask(srv.base_url, name)
        res.write(event="answers", which="C base, in place again", answers=brief(Cc))
        ab, cb = same(A, B), same(A, Cc)
        res.write(event="check", update_changed_answers=not ab[0], a_vs_b_max_logprob_diff=ab[1],
                  back_to_base_same_text=cb[0], back_to_base_max_logprob_diff=cb[1])
        if not a.skip_fresh:
            rs = srv.restart(str(alt))
            res.write(event="restart_alt", **rs)
            if rs["ready_s"] is not None:
                D = ask(srv.base_url, srv.served_name())
                res.write(event="answers", which="D alt, fresh start", answers=brief(D))
                bd = same(B, D)
                res.write(event="check", inplace_equals_fresh_text=bd[0], inplace_vs_fresh_max_logprob_diff=bd[1])
    except BaseException as e:  # noqa: BLE001
        res.fail("inplace", e)
        raise
    finally:
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
