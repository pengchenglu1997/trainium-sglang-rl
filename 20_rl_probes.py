#!/usr/bin/env python3
"""What an RL rollout needs from a running server, one probe per line.

    python 20_rl_probes.py --base-url http://127.0.0.1:30000 --model NAME --results DIR --tag sgl

Each probe records the HTTP status, the seconds, and the first 300 characters of an error body;
for sampled requests the number of distinct answers. Nothing is changed on the server: the
weight-update and LoRA calls are made with a path that does not exist, to read the refusal."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402

Q = "A farmer has 17 rows of 23 apple trees. He cuts down 41 trees. How many trees are left?"
# SGLang's native endpoints for new weights and adapters (upstream names).
WEIGHT_ENDPOINTS = [
    ("/update_weights_from_disk", {"model_path": "/nonexistent/probe"}),
    ("/update_weights_from_tensor", {"serialized_named_tensors": []}),
    ("/update_weights_from_distributed", {"names": [], "dtypes": [], "shapes": []}),
    ("/init_weights_update_group", {"master_address": "127.0.0.1", "master_port": 1,
                                    "rank_offset": 0, "world_size": 1,
                                    "group_name": "probe", "backend": "nccl"}),
    ("/destroy_weights_update_group", {"group_name": "probe"}),
    ("/get_weights_by_name", {"name": "model.embed_tokens.weight", "truncate_size": 1}),
    ("/load_lora_adapter", {"lora_name": "probe", "lora_path": "/nonexistent/probe"}),
    ("/unload_lora_adapter", {"lora_name": "probe"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    res = C.Results(a.results, f"20_rl_probes_{a.tag}")
    m = C.messages_for(Q)
    gold = 17 * 23 - 41

    def rec(name, r, **more):
        res.write(event="probe", probe=name, status=r["status"], seconds=round(r["seconds"], 2),
                  answers=len(r["texts"]), distinct=len(set(r["texts"])),
                  correct=sum(C.is_correct(t, gold) for t in r["texts"]),
                  completion_tokens=r["completion_tokens"], logprob_tokens=r["logprob_tokens"],
                  error=(r["error"] or "")[:300] or None, **more)
        return r

    try:
        for path in ("/health", "/v1/models", "/get_model_info", "/get_server_info"):
            code, body = rc.get(a.base_url.rstrip("/") + path)
            res.write(event="probe", probe="get " + path, status=code,
                      body_chars=len(body) if code == 200 else None,
                      error=None if code == 200 else body[:300])
        rec("greedy", rc.chat(a.base_url, a.model, m, 0.0, 256))
        rec("no temperature field", rc.chat(a.base_url, a.model, m, None, 256))
        rec("sampled", rc.chat(a.base_url, a.model, m, a.temperature, 256))
        rec("sampled top_p 0.95", rc.chat(a.base_url, a.model, m, a.temperature, 256,
                                          extra={"top_p": 0.95}))
        rec(f"sampled n={a.k}", rc.chat(a.base_url, a.model, m, a.temperature, 256, n=a.k))
        rec(f"sampled n={a.k} with seed", rc.chat(a.base_url, a.model, m, a.temperature, 256,
                                                  n=a.k, seed=7))
        s1 = rec("sampled seed 7, first", rc.chat(a.base_url, a.model, m, a.temperature, 256, seed=7))
        s2 = rec("sampled seed 7, again", rc.chat(a.base_url, a.model, m, a.temperature, 256, seed=7))
        s3 = rec("sampled seed 8", rc.chat(a.base_url, a.model, m, a.temperature, 256, seed=8))
        ok = all(x["status"] == 200 and x["texts"] for x in (s1, s2, s3))
        res.write(event="probe", probe="seed honoured",
                  same_seed_same_text=(s1["texts"] == s2["texts"]) if ok else None,
                  other_seed_other_text=(s1["texts"] != s3["texts"]) if ok else None)
        rec("greedy logprobs", rc.chat(a.base_url, a.model, m, 0.0, 64, logprobs=True))
        rec("sampled logprobs", rc.chat(a.base_url, a.model, m, a.temperature, 64, logprobs=True))
        rec("sampled logprobs top 5", rc.chat(a.base_url, a.model, m, a.temperature, 64,
                                              logprobs=True, top_logprobs=5))
        with ThreadPoolExecutor(max_workers=a.k) as ex:
            rs = list(ex.map(lambda i: rc.chat(a.base_url, a.model, m, a.temperature, 256), range(a.k)))
        texts = [r["texts"][0] for r in rs if r["status"] == 200 and r["texts"]]
        res.write(event="probe", probe=f"{a.k} sampled requests at once",
                  ok=len(texts), distinct=len(set(texts)),
                  correct=sum(C.is_correct(t, gold) for t in texts),
                  seconds=round(max(r["seconds"] for r in rs), 2),
                  error=next(((r["error"] or "")[:300] for r in rs if r["status"] != 200), None))
        g = rc.sample_group(a.base_url, a.model, m, a.k, a.temperature, 256, mode="auto", seed=11)
        res.write(event="probe", probe="group of answers as the RL loop asks", mode=g["mode"],
                  answers=sum(t is not None for t in g["texts"]), errors=g["errors"],
                  seconds=round(g["seconds"], 2), error=(g["first_error"] or "")[:300] or None)
        for path, body in WEIGHT_ENDPOINTS:
            code, out, sec = rc.post(a.base_url.rstrip("/") + path, body, timeout=60)
            text = out if isinstance(out, str) else __import__("json").dumps(out)
            res.write(event="probe", probe="post " + path, status=code, seconds=round(sec, 2),
                      body=text[:300])
        # the server must still answer after the refused calls
        rec("greedy after the refused calls", rc.chat(a.base_url, a.model, m, 0.0, 64))
    except BaseException as e:  # noqa: BLE001
        res.fail("probes", e)
        raise
    finally:
        res.done()


if __name__ == "__main__":
    main()
