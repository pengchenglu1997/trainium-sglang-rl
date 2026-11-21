#!/usr/bin/env python3
"""Sampled tokens per second of a set of rollout servers, as the RL loop asks for them.

    (environment of the patched SGLang-Neuron server; SGL_EXTRA holds the common launch flags)
    python 55_rollout_bench.py --model DIR --train-jsonl FILE --results DIR --tag tp1x4 \
        --replica-cores "24;25;26;27" --extra "--tp-size 1" [--questions 64] [--k 8] [--per-replica 4]

Starts one server per entry of --replica-cores (the first alone, it may have to compile), sends
--questions groups of K sampled answers round-robin (--per-replica groups in flight per server,
so K x per-replica requests each), three passes, and reports seconds, completion tokens and
tokens per second for each pass, per replica count and per core. Then stops the servers."""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402


def cores_in(spec):
    n = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--replica-cores", required=True)
    ap.add_argument("--extra", default="", help="launch flags added after SGL_EXTRA for these servers")
    ap.add_argument("--port", type=int, default=30060)
    ap.add_argument("--questions", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--per-replica", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    res = C.Results(a.results, f"55_rollout_bench_{a.tag}")
    cores = a.replica_cores.split(";")
    extra = (os.environ.get("SGL_EXTRA", "") + " " + a.extra).strip()
    backend = server_ctl.load_backend(a.backends_json, a.backend)
    servers = [server_ctl.Server(backend, a.port + i, Path(a.results) / "server_logs", name=f"bench_{a.tag}_{i}",
                                 env={"SGL_CORES": c, "SGL_EXTRA": extra}) for i, c in enumerate(cores)]
    try:
        n_cores = sum(cores_in(c) for c in cores)
        res.write(event="start", replicas=len(servers), cores=n_cores, replica_cores=a.replica_cores, extra=extra)
        st = servers[0].start(a.model)
        res.write(event="server_start", replica=0, **st)
        if st["ready_s"] is None:
            raise RuntimeError("first server did not become ready")
        if len(servers) > 1:
            with ThreadPoolExecutor(max_workers=len(servers) - 1) as ex:
                sts = list(ex.map(lambda s: s.start(a.model), servers[1:]))
            for i, s in enumerate(sts, start=1):
                res.write(event="server_start", replica=i, **s)
                if s["ready_s"] is None:
                    raise RuntimeError(f"replica {i} did not become ready")
        rows = C.load_gsm8k(a.train_jsonl, limit=2000)[-a.questions:]
        for p in range(a.passes):
            def one(ir):
                i, r = ir
                s = servers[i % len(servers)]
                return rc.sample_group(s.base_url, s.served_name(), C.messages_for(r[1]), a.k, a.temperature,
                                       a.max_new_tokens, mode="parallel", seed=9000000 + p * 100000 + i * a.k, timeout=900)
            t = time.time()
            with ThreadPoolExecutor(max_workers=a.per_replica * len(servers)) as ex:
                groups = list(ex.map(one, list(enumerate(rows))))
            sec = time.time() - t
            tok = sum(g["completion_tokens"] or 0 for g in groups)
            missing = sum(len(g["texts"]) != a.k or any(x is None for x in g["texts"]) for g in groups)
            res.write(event="pass", n=p, questions=len(rows), answers=len(rows) * a.k, seconds=round(sec, 2),
                      completion_tokens=tok, tokens_per_s=round(tok / sec, 1),
                      tokens_per_s_per_core=round(tok / sec / n_cores, 1), groups_with_missing_answers=missing,
                      first_error=next((g["first_error"] for g in groups if g["first_error"]), None))
    except BaseException as e:  # noqa: BLE001
        res.fail("bench", e)
        raise
    finally:
        for s in servers:
            try:
                s.stop()
            except Exception:  # noqa: BLE001
                pass
        res.done()


if __name__ == "__main__":
    main()
