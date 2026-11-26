#!/usr/bin/env python3
"""Did the adapter updates of a server run go through the NKI kernel? (extra stop rule of H2, R7)

    python 79_kernel_ran_check.py <server log file> [<server log file> ...] --updates N [--ranks R]
    python 79_kernel_ran_check.py --adapter-result <50_adapter_*.jsonl> [--updates N]

Mode 1 (server logs, the default use). Reads the log lines the patch writes (strings below, quoted from the patch
files with file:line) and prints ONE JSON line
  {"updates_seen": .., "by_kernel": .., "kernel_tensors_min": .., "fallback_lines": .., "pass": true|false, ...}
and exits 1 when pass is false. pass needs ALL of:
  * updates_seen == N                    (the number of completed in-place updates in the logs)
  * by_kernel == N * R                   (every update has the "by the kernel" line, once per process that logs it)
  * kernel_tensors_min > 0               (that line says how many live tensors the kernel wrote)
  * fallback_lines == 0                  (no "kernel failed" line, no host-path line, no layout-check refusal, no
                                          "in-place update ... failed" line)
R is the number of log lines one update writes (one per tensor-parallel process if every rank's log reaches the file,
else 1). --ranks R fixes it; default: 1 if the completed-update lines number N, --tp-size (default 4) times N if they
number that, anything else is a failure.
Extra keys: lines_per_update, how (the kernel call forms seen), factors (modes seen), host_tensors_max (tensors a kernel
update still merged on the host), refused_seen (that line's "refused" text), reasons (why pass is false).

Mode 2 (--adapter-result). Evaluates the result file of 50_adapter_server_test.py for H2 and prints ONE JSON line
{"adapter_test": .., "checks": {..}, "pass": ..}. pass needs: no "status": "fail" row, last row {"done": true}, the
server ready on the base, 4 "update" rows all ok, every answer of every phase HTTP 200 with tokens, and the four
checks of the plan's H2 row: adapter1 changes the answers, adapter2 differs from adapter1, adapter1 again equals
adapter1 (same text, largest logprob difference 0.0), zero adapter gives the base back (same text, 0.0). The 0.0 is
tested on the key max_logprob_diff_raw (unrounded); a result file without it falls back to max_logprob_diff, which
50_adapter_server_test.py rounds to 6 decimals, and a note "rounded to 6 decimals" goes to stderr. The check
"adapter sync equals a fresh start on the merged checkpoint" is reported as info and does NOT decide pass: the plan
tests that explanation in H4 (a restart on the checkpoint the kernel wrote), and the host-merged checkpoint of this
test is rounded by another arithmetic.

Standard library only. Exit 0 pass, 1 fail, 2 usage error.
"""
import argparse
import json
import re
import sys

# ---- strings of the patch (kit folder patches/sglang_neuron_v0.1.0_inplace) --------------------------------------------
# host.py:382-386  logger.info(...) after a kernel update, one line:
#   "[neuron] EXPERIMENT adapter update from %s by the kernel (%s call, %s factors): adapter %d bytes, %d live tensors by
#    the kernel, %d put back to the base on the device, %d on the host (%d bytes in all; refused %s); layout probe ..."
KERNEL_LINE = re.compile(
    r"adapter update from (?P<path>\S+) by the kernel \((?P<how>\S+) call, (?P<mode>\S+) factors\): "
    r"adapter (?P<abytes>\d+) bytes, (?P<ktensors>\d+) live tensors by the kernel, "
    r"(?P<restored>\d+) put back to the base on the device, (?P<host>\d+) on the host "
    r"\((?P<allbytes>\d+) bytes in all; refused (?P<refused>.*?)\); layout probe")
# host.py:375-377  logger.warning(...) when the kernel call or its read-back check raised; the whole update is redone
# on the host and kernel mode is off for the server:
#   "[neuron] EXPERIMENT adapter update by the kernel failed (%s: %s); from the host from now on"
KERNEL_FAILED = "adapter update by the kernel failed ("
# host.py:443  logger.info(...) at the end of the host/device path (reached when kernel mode is off or failed; it is
# NOT written after a successful kernel update, which returns at host.py:393):
#   "[neuron] EXPERIMENT adapter update from %s on the %s (write <mode>): adapter ..."
HOST_PATH_LINE = re.compile(r"adapter update from \S+ on the (?P<where>.+?) \(write (?P<wm>\w+)\): adapter")
# rlkit_kernel_path.py:290-291  logger.warning(...) when the layout probe of a kind of parameter fails:
#   "[neuron] EXPERIMENT kernel adapter path: layout check failed for %s (%s); such parameters are merged on the host"
LAYOUT_FAILED = "kernel adapter path: layout check failed for"
# general_plugin.py:923  logger.info(...) after ModelHost.update_weights_in_place returned (any path):
#   "[neuron] EXPERIMENT in-place update from %s: drain %.3fs, write %.3fs"
UPDATE_DONE = re.compile(r"EXPERIMENT in-place update from \S+: drain [0-9.]+s, write [0-9.]+s")
# general_plugin.py:921-922  logger.exception(...) when the update raised:
#   "[neuron] EXPERIMENT in-place update from %s failed"
UPDATE_FAILED = re.compile(r"EXPERIMENT in-place update from \S+ failed")


def scan_logs(paths):
    c = {"done": 0, "kernel": [], "host_path": 0, "kernel_failed": 0, "layout_failed": 0, "update_failed": 0}
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if UPDATE_DONE.search(line):
                    c["done"] += 1
                if UPDATE_FAILED.search(line):
                    c["update_failed"] += 1
                if KERNEL_FAILED in line:
                    c["kernel_failed"] += 1
                if LAYOUT_FAILED in line:
                    c["layout_failed"] += 1
                if HOST_PATH_LINE.search(line):
                    c["host_path"] += 1
                m = KERNEL_LINE.search(line)
                if m:
                    c["kernel"].append(m.groupdict())
    return c


def check_logs(paths, n_updates, ranks, tp_size=4):
    c = scan_logs(paths)
    ks = c["kernel"]
    fallback = c["kernel_failed"] + c["host_path"] + c["layout_failed"] + c["update_failed"]
    reasons = []
    per = ranks
    if per is None:
        # UNVERIFIED which of these a server log shows: one line per update (one scheduler process logs) or one per
        # tensor-parallel process. Only these two are accepted; any other count is a failure, not a guess.
        if c["done"] == n_updates:
            per = 1
        elif c["done"] == n_updates * tp_size:
            per = tp_size
        else:
            per = 1
            reasons.append(f"completed-update lines {c['done']} is neither --updates {n_updates} nor "
                           f"{n_updates} x tp-size {tp_size}")
    updates_seen = c["done"] // per if per else 0
    tensors = [int(k["ktensors"]) for k in ks]
    tmin = min(tensors) if tensors else 0
    if updates_seen != n_updates or c["done"] != n_updates * per:
        reasons.append(f"updates seen {updates_seen} (completed-update lines {c['done']}, {per} per update), wanted {n_updates}")
    if len(ks) != n_updates * per:
        reasons.append(f"'by the kernel' lines {len(ks)}, wanted {n_updates * per}")
    if not ks:
        reasons.append("the kernel never ran: no 'by the kernel' line")
    elif tmin <= 0:
        reasons.append("a 'by the kernel' line reports 0 live tensors written by the kernel")
    if fallback:
        reasons.append(f"fallback lines {fallback} (kernel failed {c['kernel_failed']}, host path {c['host_path']}, "
                       f"layout refused {c['layout_failed']}, update failed {c['update_failed']})")
    out = {"updates_seen": updates_seen, "by_kernel": len(ks), "kernel_tensors_min": tmin, "fallback_lines": fallback,
           "pass": not reasons, "lines_per_update": per, "updates_wanted": n_updates,
           "how": sorted({k["how"] for k in ks}), "factors": sorted({k["mode"] for k in ks}),
           "host_tensors_max": max([int(k["host"]) for k in ks], default=0),
           "refused_seen": sorted({k["refused"] for k in ks if k["refused"] not in ("{}", "")}),
           "reasons": reasons}
    return out


# ---- result file of 50_adapter_server_test.py ------------------------------------------------------------------------
# 50_adapter_server_test.py:154-157 and :163-164 write event "check" with these "what" strings
MUST_DIFFER = ["adapter1 changes the answers", "adapter2 differs from adapter1"]
MUST_EQUAL = ["adapter1 again equals adapter1 (replace, not add)", "zero adapter gives the base back"]
INFO_ONLY = "adapter sync equals a fresh start on the merged checkpoint"


def check_adapter(path, n_updates):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    checks, reasons = {}, []
    if not rows or rows[-1] != {"done": True}:
        reasons.append("the file does not end with {\"done\": true}")
    if any(r.get("status") == "fail" for r in rows):
        reasons.append("a row has \"status\": \"fail\": " + "; ".join(
            f"{r.get('stage')}: {r.get('error_type')}: {str(r.get('message'))[:120]}" for r in rows if r.get("status") == "fail"))
    sb = [r for r in rows if r.get("event") == "start_base"]
    if not sb or sb[0].get("ready_s") is None:
        reasons.append("the server did not become ready on the base model")
    ups = [r for r in rows if r.get("event") == "update"]
    bad_up = [r.get("to") for r in ups if not r.get("ok")]
    checks["updates"] = {"seen": len(ups), "wanted": n_updates, "not_ok": bad_up, "seconds": [r.get("seconds") for r in ups]}
    if len(ups) != n_updates or bad_up:
        reasons.append(f"updates: {len(ups)} seen (wanted {n_updates}), not ok: {bad_up}")
    answers = [r for r in rows if r.get("event") == "answers"]
    bad_ans = [r.get("which") for r in answers
               if not r.get("answers") or any(a.get("status") != 200 or not a.get("tokens") for a in r["answers"])]
    checks["answers"] = {"phases": [r.get("which") for r in answers], "bad_phases": bad_ans}
    if len(answers) < 5 or bad_ans:
        reasons.append(f"answers: {len(answers)} phases (wanted at least 5 before the restart), bad: {bad_ans}")
    by_what = {r.get("what"): r for r in rows if r.get("event") == "check"}
    for what in MUST_DIFFER:
        r = by_what.get(what)
        ok = r is not None and r.get("same_text") is False
        checks[what] = {"found": r is not None, "same_text": None if r is None else r.get("same_text"), "ok": ok}
        if not ok:
            reasons.append(f"check '{what}' failed or missing")
    for what in MUST_EQUAL:
        r = by_what.get(what)
        # the unrounded key of 50_adapter_server_test.py; an older result file only has the key rounded to 6 decimals
        raw_key = r is not None and "max_logprob_diff_raw" in r
        if r is not None and not raw_key:
            print(f"note: check '{what}': no max_logprob_diff_raw in the result file; max_logprob_diff is rounded to 6 "
                  f"decimals (a difference below 5e-7 shows as 0.0)", file=sys.stderr)
        diff = None if r is None else r.get("max_logprob_diff_raw" if raw_key else "max_logprob_diff")
        ok = r is not None and r.get("same_text") is True and diff == 0.0
        checks[what] = {"found": r is not None, "same_text": None if r is None else r.get("same_text"),
                        "max_logprob_diff": None if r is None else r.get("max_logprob_diff"),
                        "max_logprob_diff_raw": None if not raw_key else r.get("max_logprob_diff_raw"),
                        "tested_key": "max_logprob_diff_raw" if raw_key else "max_logprob_diff (rounded to 6 decimals)",
                        "ok": ok}
        if not ok:
            reasons.append(f"check '{what}' failed or missing")
    r = by_what.get(INFO_ONLY)
    checks["info_only: " + INFO_ONLY] = None if r is None else {"same_text": r.get("same_text"),
                                                                "max_logprob_diff": r.get("max_logprob_diff")}
    return {"adapter_test": path, "checks": checks, "pass": not reasons, "reasons": reasons}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="server log file(s) of the run")
    ap.add_argument("--updates", type=int, help="number of adapter updates the run made (4 for 50_adapter_server_test.py)")
    ap.add_argument("--ranks", type=int, default=None,
                    help="log lines one update writes (default: 1 or --tp-size, whichever the logs show)")
    ap.add_argument("--tp-size", type=int, default=4, help="tensor-parallel size of the run (default 4)")
    ap.add_argument("--adapter-result", help="the 50_adapter_*.jsonl file: evaluate the adapter test's checks")
    a = ap.parse_args()
    if not a.logs and not a.adapter_result:
        ap.error("give server log files or --adapter-result")
    if a.logs and a.updates is None:
        ap.error("--updates N is required with server logs")
    ok = True
    try:
        if a.logs:
            res = check_logs(a.logs, a.updates, a.ranks, a.tp_size)
            print(json.dumps(res))
            ok = ok and res["pass"]
        if a.adapter_result:
            res = check_adapter(a.adapter_result, a.updates if a.updates is not None else 4)
            print(json.dumps(res))
            ok = ok and res["pass"]
    except (OSError, ValueError) as e:
        print(json.dumps({"pass": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}))
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
