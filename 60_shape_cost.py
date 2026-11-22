#!/usr/bin/env python3
"""What do the compiled shapes cost when the real load does not fill them?

    (environment of srv_sglang_neuron.sh)
    python 60_shape_cost.py --model DIR --train-jsonl FILE --results DIR --tag A --cores 12-15 \
        --extra "--context-length 1024 ... --cuda-graph-bs-decode 32" [--env "RLKIT_DECODE_LADDER=0"]

One server with the given launch flags. For each number of concurrent requests c in --levels, two passes of c
sampled answers of up to --max-new-tokens tokens to GSM8K questions (the first pass of the first level is a warm-up
and is marked). Written per pass: wall seconds, tokens of every answer, the longest answer, and
    step_ms       = wall / longest answer      (the batch runs until its longest answer ends)
    row_fill      = completion tokens / (c x longest answer)   (share of the c rows that held a live answer)
The reader compares step_ms across c (does a step with few live rows cost less?) and across servers started with
other shapes (one decode bucket or the default set; context 1024, 4096, 8192)."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--env", default="")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--levels", default="32,16,8,4,2,1")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    a = ap.parse_args()
    res = C.Results(a.results, f"60_shape_cost_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=f"shape_{a.tag}",
                            env={"SGL_CORES": a.cores, "SGL_EXTRA": a.extra, "SGL_ENV": a.env})
    rows = C.load_gsm8k(a.train_jsonl, limit=2000)[-32:]
    try:
        st = srv.start(a.model)
        res.write(event="server_start", extra=a.extra, env=a.env, **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready")
        url, name = srv.base_url, srv.served_name()
        first = True
        for c in [int(x) for x in a.levels.split(",")]:
            for p in range(3 if first else 2):
                def one(i):
                    return rc.chat(url, name, C.messages_for(rows[i][1]), temperature=0.7,
                                   max_tokens=a.max_new_tokens, seed=6000000 + i, timeout=900)
                t = time.time()
                with ThreadPoolExecutor(max_workers=c) as ex:
                    outs = list(ex.map(one, range(c)))
                sec = time.time() - t
                tok = [o["completion_tokens"] or 0 for o in outs]
                longest = max(tok) if tok else 0
                res.write(event="pass", concurrent=c, n=p, warm_up=bool(first and p == 0), seconds=round(sec, 3),
                          completion_tokens=sum(tok), longest=longest, tokens=sorted(tok),
                          step_ms=round(1000.0 * sec / longest, 2) if longest else None,
                          row_fill=round(sum(tok) / (c * longest), 3) if longest else None,
                          tokens_per_s=round(sum(tok) / sec, 1), errors=sum(o["status"] != 200 for o in outs),
                          first_error=next((o["error"] for o in outs if o["status"] != 200), None))
            first = False
    except BaseException as e:  # noqa: BLE001
        res.fail("shape", e)
        raise
    finally:
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
