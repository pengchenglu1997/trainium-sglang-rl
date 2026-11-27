"""g06: from server_prof_<tag>_001.log, the 'Decode batch' log lines just before profiler_start and just after
profiler_stop (TP0), to bound live rows (#running-req) and KV tokens (#token) around the profile window.
Also prints the passes of s4_profile_<tag>.jsonl (seconds, requests, completion tokens). args: tags."""
import sys, os, re, json
from verify_common import ROOT

LOG = os.path.join(ROOT, "records", "server_logs")
REC = os.path.join(ROOT, "records")
for t in sys.argv[1:]:
    f = os.path.join(LOG, "server_prof_%s_001.log" % t)
    lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(lines) if "profiler_start" in l]
    stops = [i for i, l in enumerate(lines) if "profiler_stop" in l]
    print("=" * 10, t, "profiler_start lines", len(starts), "stop lines", len(stops))
    for a, b in zip(starts, stops):
        before = [l for l in lines[:a] if "TP0] Decode batch" in l][-2:]
        after = [l for l in lines[b:] if "TP0] Decode batch" in l][:2]
        print(" start:", lines[a][:40], "| stop:", lines[b][:40])
        for l in before + ["   ..."] + after:
            m = re.search(r"\[(.*?) TP0\].*#running-req: (\d+), #token: (\d+)", l)
            print("   ", (m.group(1), "running", m.group(2), "tokens", m.group(3)) if m else l)
    r = os.path.join(REC, "s4_profile_%s.jsonl" % t)
    for l in open(r):
        j = json.loads(l)
        if j.get("event") == "pass":
            print("  pass", j["which"], "s", j["seconds"], "req", j["requests"], "tok", j["completion_tokens"],
                  "min/max ans", j["shortest_answer_tokens"], j["longest_answer_tokens"], "temp", j.get("temperature"),
                  "extra", j.get("request_extra"), "errors", j["errors"])
