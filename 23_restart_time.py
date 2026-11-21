#!/usr/bin/env python3
"""Seconds to stop a server and bring it back on the same weights, N times.

    python 23_restart_time.py --backend sglang_neuron --model DIR --results DIR --tag sgl \
        [--repeats 2] [--port 30000] [--leave-running]

The first start of a model compiles its graphs; later starts read the compile cache. The loop
of 30_rl_loop.py restarts on new weights; this is the same restart on unchanged weights, so the
two can be compared. With --leave-running the server stays up for the next step of the session
(the name printed is what the stop command needs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402
import server_ctl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--backends-json", default=str(Path(__file__).with_name("backends.json")))
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--name", default=None)
    ap.add_argument("--leave-running", action="store_true")
    a = ap.parse_args()
    res = C.Results(a.results, f"23_restart_{a.tag}")
    srv = server_ctl.Server(server_ctl.load_backend(a.backends_json, a.backend), a.port,
                            Path(a.results) / "server_logs", name=a.name or f"srv_{a.tag}")
    ok = False
    try:
        st = srv.start(a.model)
        res.write(event="first_start", **st)
        if st["ready_s"] is None:
            raise RuntimeError("server did not become ready")
        for i in range(a.repeats):
            r = srv.restart(a.model)
            res.write(event="restart", n=i + 1, **r)
            if r["ready_s"] is None:
                raise RuntimeError(f"server did not come back at restart {i + 1}")
        ok = True
        res.write(event="summary", first_start_s=st["ready_s"], served=srv.served_name(),
                  base_url=srv.base_url, name=srv.name, left_running=bool(a.leave_running))
        print(f"SERVER {srv.name} {srv.base_url} {srv.served_name()}", flush=True)
    except BaseException as e:  # noqa: BLE001
        res.fail("restart", e)
        raise
    finally:
        if not (ok and a.leave_running):
            try:
                srv.stop()
            except Exception:  # noqa: BLE001
                pass
        res.done()


if __name__ == "__main__":
    main()
