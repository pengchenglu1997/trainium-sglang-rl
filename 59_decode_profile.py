#!/usr/bin/env python3
"""Where does a decode step of the rollout server go? Records the Neuron runtime's system profile
(docs/guides/profiling.md of sglang-neuron: POST /start_profile with activity NEURON) around one rollout pass.

    (environment of srv_sglang_neuron.sh)
    python 59_decode_profile.py --model DIR --train-jsonl FILE --results DIR --tag p1 --cores 16-19

One server; a warm pass of 32 sampled answers of up to 256 tokens (one decode batch), then the profile window around
a second, identical pass. The sessions are written by the server under /cache/profile_<tag> (the compile-cache folder
of the host, $SGL_CACHE/profile_<tag>). This script records the wall time and tokens of both passes and the
/start_profile and /stop_profile answers; the sessions are read afterwards with neuron-explorer."""
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


def one_pass(url, name, rows, seed0):
    def one(ir):
        i, r = ir
        return rc.chat(url, name, C.messages_for(r[1]), temperature=0.7, max_tokens=256, seed=seed0 + i, timeout=900)
    t = time.time()
    with ThreadPoolExecutor(max_workers=len(rows)) as ex:
        outs = list(ex.map(one, list(enumerate(rows))))
    sec = time.time() - t
    tok = [o["completion_tokens"] or 0 for o in outs]
    return {"seconds": round(sec, 3), "requests": len(rows), "completion_tokens": sum(tok),
            "longest_answer_tokens": max(tok) if tok else 0, "errors": sum(o["status"] != 200 for o in outs),
            "first_error": next((o["error"] for o in outs if o["status"] != 200), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--backend", default="sglang_neuron_inplace")
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--port", type=int, default=30090)
    ap.add_argument("--requests", type=int, default=32)
    a = ap.parse_args()
    res = C.Results(a.results, f"59_decode_profile_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=f"prof_{a.tag}", env={"SGL_CORES": a.cores})
    rows = C.load_gsm8k(a.train_jsonl, limit=2000)[-a.requests:]
    try:
        st = srv.start(a.model)
        res.write(event="server_start", **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready")
        url, name = srv.base_url, srv.served_name()
        res.write(event="pass", which="warm, no profile", **one_pass(url, name, rows, 5000000))
        res.write(event="pass", which="second, no profile", **one_pass(url, name, rows, 5100000))
        out_dir = f"/cache/profile_{a.tag}"
        code, out, sec = rc.post(url + "/start_profile", {"activities": ["NEURON"], "output_dir": out_dir}, timeout=300)
        res.write(event="start_profile", status=code, seconds=round(sec, 2), body=str(out)[:300], output_dir=out_dir)
        if code == 200:
            res.write(event="pass", which="profiled", **one_pass(url, name, rows, 5100000))
            code, out, sec = rc.post(url + "/stop_profile", {}, timeout=900)
            res.write(event="stop_profile", status=code, seconds=round(sec, 2), body=str(out)[:300])
    except BaseException as e:  # noqa: BLE001
        res.fail("profile", e)
        raise
    finally:
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
        res.done()


if __name__ == "__main__":
    main()
