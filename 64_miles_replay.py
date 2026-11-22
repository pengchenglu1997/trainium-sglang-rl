#!/usr/bin/env python3
"""Replays, against a real server URL, the HTTP calls one Miles disk-delta weight update sends to a rollout engine.

    python 64_miles_replay.py --base-url http://127.0.0.1:30000 --ckpt /local/ckpt [--weight-version 1]
           [--target-version 1] [--source-dir /shared/delta] [--pause-mode retract] [--baseline]
           [--results DIR] [--tag T]

What this is: OUR replay of the call sequence that Miles's own code issues (radixark/miles, files
miles/backends/training_utils/weight_update/protocols/delta.py and miles/backends/sglang_utils/sglang_api_client.py,
read 22 November; see 64_miles_sequence.md for the line numbers). Miles itself is not installed or run. The replay
sends the same routes, methods and JSON fields, in the same order, and applies Miles's own success test to each answer
(HTTP status below 400 and no "success": false in a JSON body). It does NOT write the delta files and does not
materialise a checkpoint: --ckpt must already be a loadable checkpoint directory of the served model, as the server
sees the path (Miles leaves that to the engine's /pull_weights; see the sequence note).

Differences from Miles, on purpose: after a failed step the replay goes on, so one run shows every route's answer
(each row says whether Miles would have stopped there), and /continue_generation is always sent, so a server that
accepted the pause is not left paused.

Results (JSON lines through rl_common.Results): one row per call with the step name, method, route, HTTP status,
seconds, whether Miles would accept the answer, and the keys of the answer; then a summary row with the seconds of
the whole sequence, the seconds between the pause call and the end of continue_generation (the window in which the
server admits no request), and the first step Miles would have stopped at; then {"done": true}."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402

FLUSH_TRIES = 60       # Miles: sglang_api_client.py flush_cache, "for _ in range(60)" with a 1 s sleep
FLUSH_SLEEP_S = 1.0


def sequence(ckpt, source_dir, target_version, weight_version, pause_mode, baseline, verify):
    """The ordered calls of one update. Each: name, method, route, body (None for GET), miles_checks_success.

    Order and payloads follow Miles: _capture_baseline (first update only, pull with target_version 0), then
    _reload_engines: pull_weights, pause_generation, flush_cache (not in_place), update_weights_from_disk,
    continue_generation. `verify` adds the GET /model_info that Miles's updater sends only under --ci-test
    (verify_engine_version)."""
    pull = {"local_checkpoint_dir": ckpt, "source_dir": source_dir}
    steps = []
    if baseline:
        steps.append(dict(name="pull_weights(baseline)", method="POST", route="/pull_weights",
                          body=dict(pull, target_version=0), checks_success=True))
    steps.append(dict(name="pull_weights", method="POST", route="/pull_weights",
                      body=dict(pull, target_version=target_version), checks_success=True))
    steps.append(dict(name="pause_generation", method="POST", route="/pause_generation",
                      body={"mode": pause_mode}, checks_success=False))
    if pause_mode != "in_place":
        steps.append(dict(name="flush_cache", method="GET", route="/flush_cache", body=None,
                          checks_success=False))
    steps.append(dict(name="update_weights_from_disk", method="POST", route="/update_weights_from_disk",
                      body={"model_path": ckpt, "weight_version": str(weight_version)}, checks_success=True))
    steps.append(dict(name="continue_generation", method="POST", route="/continue_generation", body={},
                      checks_success=False))
    if verify:
        steps.append(dict(name="model_info", method="GET", route="/model_info", body=None, checks_success=False,
                          expect_version=str(weight_version)))
    return steps


def miles_accepts(status, body, checks_success, expect_version=None):
    """Miles's own test: _make_request raises on HTTP >= 400; check_weight_sync_results (applied to the pull and
    the reload answers) raises on a JSON object whose "success" is false. For the /model_info read-back,
    verify_engine_version raises unless str(weight_version) of the answer equals the version just sent
    (updater.py verify_engine_version), and get_weight_version needs HTTP 200 with a weight_version key."""
    if not (isinstance(status, int) and 200 <= status < 400):
        return False
    if checks_success and isinstance(body, dict) and body.get("success") is False:
        return False
    if expect_version is not None:
        if status != 200 or not isinstance(body, dict) or "weight_version" not in body:
            return False
        if str(body["weight_version"]) != expect_version:
            return False
    return True


def get_whole(url, timeout):
    """GET returning (status or 0, parsed JSON when the body is JSON, else the text). rollout_client.get cuts the body
    at 2000 characters, which would break a long /model_info answer, so the replay reads it whole."""
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            status, raw = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def run_call(base, step, timeout, flush_tries):
    url = base.rstrip("/") + step["route"]
    t0 = time.time()
    if step["name"] == "flush_cache":
        # Miles: GET until HTTP 200, one second apart, up to flush_tries times; a TimeoutError after that
        tries, status, body = 0, 0, ""
        while tries < flush_tries:
            tries += 1
            status, body = get_whole(url, timeout)
            if status == 200:
                break
            time.sleep(FLUSH_SLEEP_S)
        return status, body, time.time() - t0, tries
    if step["method"] == "GET":
        status, body = get_whole(url, timeout)
        return status, body, time.time() - t0, 1
    status, body, sec = rc.post(url, step["body"], timeout=timeout)
    return status, body, sec, 1


def replay(base_url, ckpt, results, source_dir="/shared/delta", target_version=1, weight_version=1,
           pause_mode="retract", baseline=False, verify=True, timeout=600, flush_tries=FLUSH_TRIES):
    """Runs the sequence; writes rows to `results` (an rl_common.Results); returns the list of row dicts."""
    steps = sequence(ckpt, source_dir, target_version, weight_version, pause_mode, baseline, verify)
    rows, first_stop, paused_at, window = [], None, None, None
    t_all = time.time()
    for i, step in enumerate(steps, 1):
        if step["name"] == "pause_generation":
            paused_at = time.time()
        status, body, sec, tries = run_call(base_url, step, timeout, flush_tries)
        ok = miles_accepts(status, body, step["checks_success"], step.get("expect_version"))
        if step["name"] == "continue_generation" and paused_at is not None:
            window = time.time() - paused_at
        if not ok and first_stop is None:
            first_stop = step["name"]
        row = {"event": "call", "step": i, "name": step["name"], "method": step["method"], "route": step["route"],
               "payload_fields": sorted(step["body"]) if step["body"] else [], "status": status,
               "seconds": round(sec, 4), "miles_accepts": ok, "tries": tries,
               "after_miles_would_have_stopped": first_stop is not None and first_stop != step["name"],
               "answer_keys": sorted(body) if isinstance(body, dict) else None,
               "success": body.get("success") if isinstance(body, dict) else None,
               "detail": (str(body.get("message") or body.get("detail") or "")[:160]
                          if isinstance(body, dict) else str(body)[:160])}
        if step["name"] == "model_info" and isinstance(body, dict):
            row["weight_version_answered"] = body.get("weight_version")
        if status == 0:
            results.fail(step["name"], RuntimeError(str(body)))
        results.write(**row)
        rows.append(row)
    if paused_at is not None and window is None:
        window = time.time() - paused_at
    results.write(event="summary", calls=len(rows), total_seconds=round(time.time() - t_all, 4),
                  pause_to_continue_seconds=None if window is None else round(window, 4),
                  miles_would_stop_at=first_stop, miles_would_succeed=first_stop is None)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint directory as the server sees it")
    ap.add_argument("--source-dir", default="/shared/delta",
                    help="only the pull_weights payload; nothing is read or written there")
    ap.add_argument("--target-version", type=int, default=1)
    ap.add_argument("--weight-version", type=int, default=1)
    ap.add_argument("--pause-mode", default="retract", choices=["abort", "retract", "in_place"])
    ap.add_argument("--baseline", action="store_true", help="also send the first-update pull with target_version 0")
    ap.add_argument("--no-verify", action="store_true", help="leave out the GET /model_info of Miles's ci-test")
    ap.add_argument("--flush-tries", type=int, default=FLUSH_TRIES)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--tag", default="run")
    a = ap.parse_args(argv)
    res = C.Results(a.results, f"64_miles_replay_{a.tag}")
    try:
        replay(a.base_url, a.ckpt, res, a.source_dir, a.target_version, a.weight_version, a.pause_mode,
               a.baseline, not a.no_verify, a.timeout, a.flush_tries)
    except Exception as e:  # noqa: BLE001
        res.fail("replay", e)
        res.done()
        raise
    res.done()
    print(f"wrote {res.path}")


if __name__ == "__main__":
    main()
