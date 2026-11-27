#!/usr/bin/env python3
"""Compare the saved answers of cc_collect.py pair by pair; one summary line per comparison to
~/w/cc/results/summary.jsonl (rewritten each run). Method of 83_sampler_check2.pair_stats: a fork is the first
position where the token lists differ (or the shorter length); logprob differences are taken before the fork."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RES = Path.home() / "w" / "cc" / "results"
KIT = Path.home() / "w" / "sglang_rl"
PAIRS = [
    ("check1", "lmh_off_a", "lmh_on", ["t0", "t08"], "switch"),
    ("check1", "lmh_off_a", "lmh_off_b", ["t0", "t08"], "A/A control"),
    ("check2", "koff_a", "kon", ["t0", "rl", "minp", "t08"], "switch"),
    ("check2", "koff_a", "koff_b", ["t0", "rl", "minp", "t08"], "A/A control"),
    ("check3", "rel", "all", ["t0"], "switch"),
    ("check3", "rel", "rel_b", ["t0"], "A/A control (extra)"),
    ("check4", "q8_rel", "q8_all", ["t0"], "switch"),
    ("check4", "q8_rel", "q8_rel_b", ["t0"], "A/A control (extra)"),
]


def load(name, s):
    p = RES / f"out_{name}_{s}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def stats(xs, ys):
    same, same_text, forks, per, maxes, n, bit, errors = 0, 0, [], [], [], 0, 0, 0
    for i, (a, b) in enumerate(zip(xs, ys)):
        if not (a["ok"] and b["ok"]):
            errors += 1
            per.append({"i": i, "error": True})
            continue
        m = min(len(a["tokens"]), len(b["tokens"]))
        first = next((k for k in range(m) if a["tokens"][k] != b["tokens"][k]), None)
        if first is None and len(a["tokens"]) == len(b["tokens"]):
            same += 1
            fork = None
        else:
            fork = m if first is None else first
            forks.append([i, fork])
        same_text += a["text"] == b["text"]
        upto = m if first is None else first
        d = [abs(float(x) - float(y)) for x, y in zip(a["logprobs"][:upto], b["logprobs"][:upto])
             if x is not None and y is not None]
        mx = max(d) if d else None
        if d:
            maxes.append(mx)
            n += len(d)
            bit += sum(1 for x in d if x == 0.0)
        per.append({"i": i, "fork": fork, "len_a": len(a["tokens"]), "len_b": len(b["tokens"]), "max_d": mx,
                    "bit_equal": sum(1 for x in d if x == 0.0), "compared": len(d),
                    "lp_at_fork_a": a["logprobs"][fork] if fork is not None and fork < len(a["logprobs"]) else None,
                    "lp_at_fork_b": b["logprobs"][fork] if fork is not None and fork < len(b["logprobs"]) else None})
    return {"requests": len(xs), "errors": errors, "same_tokens": same, "same_text": same_text,
            "fork_positions_i_pos": forks, "tokens_compared_before_fork": n, "bit_equal_logprobs": bit,
            "max_abs_dlogprob_before_fork": max(maxes) if maxes else None,
            "per_prompt": per}


def seed_check(d):
    a = d["answers"][5]
    s, o = d.get("same_seed_again"), d.get("other_seed")
    if not s or not o or not (a["ok"] and s["ok"] and o["ok"]):
        return None
    return {"same_seed_same_tokens": a["tokens"] == s["tokens"], "other_seed_differs": a["tokens"] != o["tokens"]}


def main():
    t = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    for check, x, y, sets, kind in PAIRS:
        for s in sets:
            dx, dy = load(x, s), load(y, s)
            row = {"t": t, "check": check, "kind": kind, "a": x, "b": y, "set": s}
            if dx is None or dy is None:
                row["missing"] = [n for n, d in ((x, dx), (y, dy)) if d is None]
            else:
                row["params"] = dx["params"]
                row["seed_rule"] = "seed = 1000 + prompt index" if s != "t0" else None
                row.update(stats(dx["answers"], dy["answers"]))
                if s != "t0":
                    row["seed_check"] = {x: seed_check(dx), y: seed_check(dy)}
            lines.append(row)
    for x, y in (("rel", "all"), ("rel", "rel_b")):
        fa, fb = RES / f"21_gsm8k_cc_{x}.jsonl", RES / f"21_gsm8k_cc_{y}.jsonl"
        row = {"t": t, "check": "check3", "kind": "gsm8k", "a": x, "b": y}
        for k, f in ((x, fa), (y, fb)):
            if f.is_file():
                ev = [json.loads(z) for z in f.read_text(encoding="utf-8").splitlines() if z.strip()]
                row[k] = [e for e in ev if e.get("event") in ("start", "pass")] + [e for e in ev if e.get("done")][:1]
            else:
                row[k] = "missing"
        if fa.is_file() and fb.is_file():
            p = subprocess.run([sys.executable, str(KIT / "21_gsm8k_acc.py"), "compare", "--a", str(fa), "--b", str(fb)],
                               capture_output=True, text=True)
            row["paired_compare_stdout"] = p.stdout.strip()
        lines.append(row)
    (RES / "summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    for r in lines:
        if r.get("kind") == "gsm8k":
            print(json.dumps({k: v for k, v in r.items() if k != "t"})[:1500])
            continue
        short = {k: r.get(k) for k in ("check", "kind", "a", "b", "set", "missing", "requests", "errors",
                                        "same_tokens", "same_text", "fork_positions_i_pos",
                                        "tokens_compared_before_fork", "bit_equal_logprobs",
                                        "max_abs_dlogprob_before_fork", "seed_check") if k in r}
        print(json.dumps(short))


if __name__ == "__main__":
    main()
