"""Group the recorded neuronx-cc command lines of one cold start by compile (top workdir hash) and map each compile
to the warm phase the server log was in when it started (TP0 "[neuron] warm ..." lines, and the graph family).
Usage: python cc_table.py <cc_cmdlines.txt> <server log>"""
import re
import sys
from datetime import datetime

cc_path, log_path = sys.argv[1], sys.argv[2]
line_re = re.compile(r"^(\S+Z) pid=(\d+) cwd=(\S*) CMD: (.*)$")
compiles = {}
for n, line in enumerate(open(cc_path, encoding="utf-8", errors="replace"), 1):
    m = line_re.match(line.rstrip("\n"))
    if not m:
        continue
    t, pid, cwd, cmd = m.groups()
    mm = re.search(r"/neuronx-cc/([0-9a-f]+)", cwd)
    key = mm.group(1) if mm else f"pid{pid}"
    cmd = cmd.strip()
    c = compiles.setdefault(key, {"t": t, "line": n, "cmds": set()})
    c["cmds"].add(cmd)

warm = []
wre = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] (.*)$")
for n, line in enumerate(open(log_path, encoding="utf-8", errors="replace"), 1):
    m = wre.match(line)
    if m and ("[neuron] warm" in m.group(2) or "compile" in m.group(2).lower()):
        warm.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), n, m.group(2)[:120]))


def phase(t):
    tt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
    cur = None
    for w in warm:
        if w[0] <= tt:
            cur = w
    return cur


def args_of(cmd):
    i = cmd.find(" compile ")
    return cmd[i + 1:] if i >= 0 else cmd


print(f"compiles recorded: {len(compiles)}")
distinct = {}
for key, c in sorted(compiles.items(), key=lambda kv: kv[1]["t"]):
    cmds = sorted(args_of(x) for x in c["cmds"])
    p = phase(c["t"])
    ptxt = f"log line {p[1]}: {p[2]}" if p else "before first warm line"
    for a in set(cmds):
        distinct.setdefault(a, []).append((c["t"], c["line"], ptxt))
    print(f"{c['t']} cc_cmdlines line {c['line']} workdir {key[:12]} | {ptxt}")
print()
print(f"distinct argument strings: {len(distinct)}")
for a, uses in distinct.items():
    print(f"--- used by {len(uses)} compiles; first at cc_cmdlines line {uses[0][1]} ({uses[0][2]})")
    print(a)
print()
print("warm lines (TP0):")
for w in warm:
    print(f"  log line {w[1]}: {w[0]} {w[2]}")
