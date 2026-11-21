#!/usr/bin/env python3
"""Do the trainer and the server compute the same log-probabilities for the same sampled tokens,
before and after an adapter-only weight sync? (The "trainer vs inference logprob difference" that
RL stacks watch: it is near zero only when both sides hold the same weights.)

    (environment as 50_adapter_server_test.py; the trainer runs on its own NeuronCore)
    NEURON_RT_VISIBLE_CORES=20 python 51_logprob_parity.py --base DIR --work DIR_UNDER_MODELS_ROOT \
        --train-jsonl gsm8k/train.jsonl --results DIR --tag parity [--n 24]

  1. server on the base, trainer on the base (LoRA B = 0): the server samples one answer per
     question at temperature 1 (full vocabulary) and returns token logprobs; the trainer scores
     the same tokens.                                                       -> "base / base"
  2. the trainer's LoRA tensors are set to seeded values, saved as a PEFT adapter, and sent to
     the server with update_weights_from_disk (adapter path of the patch). New samples from the
     server, scored by the trainer.                                         -> "adapter / adapter"
  3. control: the samples of step 1 (base server) scored by the trainer that now holds the
     adapter: what a mismatch of weights looks like in this number.         -> "base / adapter"
Per comparison: answers compared, tokens compared, mean and max absolute difference per token.
An answer is left out when re-tokenising its text does not give the server's tokens one by one."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402
from trainer import Trainer  # noqa: E402


def sample(base_url, model, questions, seed0, max_tokens):
    out = []
    for i, q in enumerate(questions):
        body = {"model": model, "messages": C.messages_for(q), "max_tokens": max_tokens, "temperature": 1.0,
                "top_p": 1.0, "top_k": -1, "min_p": 0.0, "seed": seed0 + i, "logprobs": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        code, o, _ = rc.post(base_url.rstrip("/") + "/v1/chat/completions", body, timeout=600)
        if code != 200 or not isinstance(o, dict):
            out.append(None)
            continue
        ch = o["choices"][0]
        cont = (ch.get("logprobs") or {}).get("content") or []
        out.append({"q": q, "text": ch["message"]["content"], "lp": [t.get("logprob") for t in cont],
                    "toks": [t.get("token") for t in cont], "finish": ch.get("finish_reason")})
    return out


def score(tr, samples):
    """Per sample: the trainer's log-probability (temperature 1) of each answer token, or None."""
    torch = tr.torch
    tr.model.eval()
    rows = []
    for s in samples:
        if s is None or not s["lp"]:
            rows.append(None)
            continue
        p = tr.tok(tr.prompt_text(s["q"]), add_special_tokens=False)["input_ids"]
        a = tr.tok(s["text"], add_special_tokens=False)["input_ids"]
        n = len(s["lp"])
        toks = s.get("toks") or []
        # the ids scored must be the tokens the server sampled: the text re-tokenised has to give the
        # server's token strings one by one (2026-11-20: without this one token was off by 29)
        same = len(toks) == n and all(tr.tok.decode([x]) == t for x, t in zip(a, toks))
        if len(a) == n - 1 and s.get("finish") == "stop":
            a = a + [tr.tok.eos_token_id]
        if not same or len(a) != n or len(p) > tr.max_prompt or len(p) + len(a) > tr.max_len:
            rows.append(None)
            continue
        rows.append((p + a, len(p), len(a)))
    res = [None] * len(samples)
    idx = [i for i, r in enumerate(rows) if r is not None]
    pad = tr.tok.pad_token_id
    for s0 in range(0, len(idx), tr.micro_bs):
        chunk = idx[s0:s0 + tr.micro_bs]
        ids = torch.full((tr.micro_bs, tr.max_len), pad, dtype=torch.long)
        att = torch.zeros((tr.micro_bs, tr.max_len), dtype=torch.long)
        for j, i in enumerate(chunk):
            seq = rows[i][0]
            ids[j, :len(seq)] = torch.tensor(seq)
            att[j, :len(seq)] = 1
        if len(chunk) < tr.micro_bs:
            att[len(chunk):, 0] = 1
        ids_d, att_d = ids.to(tr.device), att.to(tr.device)
        with torch.no_grad():
            logits = tr.model(input_ids=ids_d, attention_mask=att_d).logits
            logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            tok_lp = logp.gather(-1, ids_d[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()
        tr.sync()
        for j, i in enumerate(chunk):
            _, n_p, n_a = rows[i]
            res[i] = [float(x) for x in tok_lp[j, n_p - 1:n_p - 1 + n_a]]
    return res


def compare(samples, scores):
    diffs, used = [], 0
    for s, t in zip(samples, scores):
        if s is None or t is None:
            continue
        used += 1
        diffs += [abs(a - b) for a, b in zip(s["lp"], t) if a is not None]
    if not diffs:
        return {"answers": 0, "tokens": 0}
    diffs.sort()
    return {"answers": used, "tokens": len(diffs), "mean_abs_diff": round(sum(diffs) / len(diffs), 5),
            "median_abs_diff": round(diffs[len(diffs) // 2], 5), "p95_abs_diff": round(diffs[int(0.95 * len(diffs))], 5),
            "max_abs_diff": round(diffs[-1], 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, default=30014)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--std", type=float, default=0.01)
    a = ap.parse_args()
    res = C.Results(a.results, f"51_parity_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=f"srv_{a.tag}")
    try:
        qs = [r[1] for r in C.load_gsm8k(a.train_jsonl, limit=400)][-a.n:]
        tr = Trainer(a.base, device="neuron", lr=1e-4, max_prompt=320, max_completion=a.max_tokens + 8, micro_bs=4)
        st = srv.start(a.base)
        res.write(event="start", n=len(qs), **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready")
        name = srv.served_name()
        s_base = sample(srv.base_url, name, qs, 5000, a.max_tokens)
        res.write(event="parity", server="base", trainer="base", **compare(s_base, score(tr, s_base)))
        torch = tr.torch
        g = torch.Generator().manual_seed(909)
        with torch.no_grad():
            for p in tr.params:
                p.data.copy_((a.std * torch.randn(p.shape, generator=g)).to(p.dtype))
        tr.sync()
        ad = Path(a.work) / f"parity_adapter_{a.tag}"
        tr.save_adapter(ad)
        for f in Path(a.base).iterdir():
            if f.is_file() and f.suffix != ".safetensors":
                shutil.copy2(f, ad / f.name)
        u = srv.update_weights_from_disk(str(ad))
        res.write(event="update", files=sorted(x.name for x in ad.iterdir())[:8], **u)
        if not u["ok"]:
            raise RuntimeError("adapter update refused")
        s_ad = sample(srv.base_url, name, qs, 6000, a.max_tokens)
        res.write(event="parity", server="adapter", trainer="adapter", **compare(s_ad, score(tr, s_ad)))
        res.write(event="parity", server="base", trainer="adapter", note="control: weights do not match",
                  **compare(s_base, score(tr, s_base)))
    except BaseException as e:  # noqa: BLE001
        res.fail("parity", e)
        raise
    finally:
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
